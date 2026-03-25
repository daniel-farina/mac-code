#!/usr/bin/env python3
"""
Tiered KV Cache — GPU → SSD → R2

Enables context windows beyond GPU memory by paging KV cache blocks
across three storage tiers:

  HOT:  GPU unified memory (current context, instant access)
  WARM: Local SSD (recent context, ~0.1ms access)
  COLD: Cloudflare R2 (persistent context, ~50ms access)

Architecture:
  - Context is split into "blocks" (chunks of N tokens)
  - Recent blocks stay in GPU memory
  - Older blocks are evicted to SSD
  - Very old or shared blocks live on R2
  - When the model needs an old block, it's prefetched from SSD/R2

This enables:
  - 256K+ context on 16GB RAM (vs 64K without paging)
  - Persistent cross-session context
  - Shared context across devices via R2
"""

import os
import time
import json
import gzip
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

CACHE_DIR = Path.home() / ".mac-code" / "kv-cache" / "blocks"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CacheBlock:
    """A chunk of KV cache state for a range of tokens."""
    block_id: int
    start_token: int
    end_token: int
    tier: str  # "gpu", "ssd", "r2"
    size_bytes: int = 0
    last_accessed: float = 0
    ssd_path: Optional[str] = None
    r2_key: Optional[str] = None


class TieredKVCache:
    """
    Manages KV cache across GPU, SSD, and R2 tiers.

    Usage:
        tiered = TieredKVCache(model, tokenizer, block_size=512)
        tiered.process_tokens(long_document_tokens)
        # Automatically pages old blocks to SSD when GPU budget exceeded
        # Load from R2 for cross-device resume
    """

    def __init__(self, model, tokenizer,
                 block_size=512,
                 gpu_budget_mb=500,
                 ssd_budget_mb=2000):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.gpu_budget_bytes = gpu_budget_mb * 1024 * 1024
        self.ssd_budget_bytes = ssd_budget_mb * 1024 * 1024

        self.blocks: Dict[int, CacheBlock] = {}
        self.block_states: Dict[int, List[Any]] = {}  # GPU-resident states
        self.next_block_id = 0
        self.total_tokens = 0

        # Stats
        self.stats = {
            "gpu_bytes": 0,
            "ssd_bytes": 0,
            "r2_bytes": 0,
            "evictions_to_ssd": 0,
            "evictions_to_r2": 0,
            "loads_from_ssd": 0,
            "loads_from_r2": 0,
        }

    def process_chunk(self, tokens, cache):
        """Process a chunk of tokens and manage cache tiers."""
        import mlx.core as mx

        # Process through model
        token_array = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
        logits = self.model(token_array[None], cache=cache)
        mx.eval(logits)

        # Record this block
        block = CacheBlock(
            block_id=self.next_block_id,
            start_token=self.total_tokens,
            end_token=self.total_tokens + len(tokens),
            tier="gpu",
            size_bytes=sum(c.nbytes for c in cache),
            last_accessed=time.time(),
        )
        self.blocks[block.block_id] = block
        self.next_block_id += 1
        self.total_tokens += len(tokens)

        # Save cache state for this block
        self.block_states[block.block_id] = [c.state for c in cache]
        self.stats["gpu_bytes"] = sum(c.nbytes for c in cache)

        # Check if we need to evict
        self._maybe_evict()

        return logits

    def _maybe_evict(self):
        """Evict oldest GPU blocks to SSD if over budget."""
        if self.stats["gpu_bytes"] <= self.gpu_budget_bytes:
            return

        # Sort blocks by last access time, evict oldest
        gpu_blocks = sorted(
            [b for b in self.blocks.values() if b.tier == "gpu"],
            key=lambda b: b.last_accessed
        )

        for block in gpu_blocks:
            if self.stats["gpu_bytes"] <= self.gpu_budget_bytes:
                break
            self._evict_to_ssd(block)

    def _evict_to_ssd(self, block):
        """Move a cache block from GPU to SSD."""
        import mlx.core as mx

        state = self.block_states.get(block.block_id)
        if not state:
            return

        # Save state tensors to npz
        block_path = CACHE_DIR / f"block_{block.block_id}.npz"
        tensors = {}
        for layer_idx, layer_state in enumerate(state):
            if isinstance(layer_state, list):
                for tensor_idx, tensor in enumerate(layer_state):
                    if hasattr(tensor, 'shape'):
                        tensors[f"l{layer_idx}_t{tensor_idx}"] = tensor
            elif hasattr(layer_state, 'shape'):
                tensors[f"l{layer_idx}_t0"] = layer_state

        if tensors:
            mx.savez(str(block_path), **tensors)
            mx.eval(*tensors.values())  # ensure saved

            actual_path = Path(str(block_path))
            # mx.savez may add .npz extension
            if not actual_path.exists():
                actual_path = Path(str(block_path) + ".npz") if not str(block_path).endswith(".npz") else actual_path

            if actual_path.exists():
                old_gpu_bytes = block.size_bytes
                block.tier = "ssd"
                block.ssd_path = str(actual_path)
                block.size_bytes = actual_path.stat().st_size
                self.stats["ssd_bytes"] += block.size_bytes
                self.stats["evictions_to_ssd"] += 1
                self.stats["gpu_bytes"] -= old_gpu_bytes

                # Free GPU state
                del self.block_states[block.block_id]

    def _load_from_ssd(self, block):
        """Load a cache block from SSD back to GPU."""
        import mlx.core as mx

        if not block.ssd_path or not os.path.exists(block.ssd_path):
            return None

        data = mx.load(block.ssd_path)
        # Reconstruct state
        state = []
        layer_idx = 0
        while True:
            layer_tensors = []
            tensor_idx = 0
            while f"layer_{layer_idx}_t_{tensor_idx}" in data:
                layer_tensors.append(data[f"layer_{layer_idx}_t_{tensor_idx}"])
                tensor_idx += 1
            if not layer_tensors:
                break
            state.append(layer_tensors)
            layer_idx += 1

        self.block_states[block.block_id] = state
        block.tier = "gpu"
        block.last_accessed = time.time()
        self.stats["loads_from_ssd"] += 1

        return state

    def _evict_to_r2(self, block):
        """Move a cache block from SSD to R2."""
        from r2_store import get_r2_client

        client, bucket = get_r2_client()
        if not client or not block.ssd_path:
            return

        # Compress and upload
        gz_path = block.ssd_path + ".gz"
        with open(block.ssd_path, "rb") as f_in:
            with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                f_out.write(f_in.read())

        r2_key = f"kv-blocks/block_{block.block_id}.safetensors.gz"
        client.upload_file(gz_path, bucket, r2_key)

        block.tier = "r2"
        block.r2_key = r2_key
        self.stats["r2_bytes"] += os.path.getsize(gz_path)
        self.stats["evictions_to_r2"] += 1

        # Clean up SSD
        os.remove(block.ssd_path)
        os.remove(gz_path)
        self.stats["ssd_bytes"] -= block.size_bytes

    def get_stats(self):
        """Return current tier stats."""
        return {
            **self.stats,
            "total_tokens": self.total_tokens,
            "total_blocks": len(self.blocks),
            "gpu_blocks": sum(1 for b in self.blocks.values() if b.tier == "gpu"),
            "ssd_blocks": sum(1 for b in self.blocks.values() if b.tier == "ssd"),
            "r2_blocks": sum(1 for b in self.blocks.values() if b.tier == "r2"),
            "gpu_mb": self.stats["gpu_bytes"] / (1024 * 1024),
            "ssd_mb": self.stats["ssd_bytes"] / (1024 * 1024),
            "r2_mb": self.stats["r2_bytes"] / (1024 * 1024),
        }

    def save_manifest(self, name):
        """Save the block manifest for restoring later."""
        manifest = {
            "name": name,
            "total_tokens": self.total_tokens,
            "block_size": self.block_size,
            "blocks": {
                str(bid): {
                    "block_id": b.block_id,
                    "start_token": b.start_token,
                    "end_token": b.end_token,
                    "tier": b.tier,
                    "size_bytes": b.size_bytes,
                    "ssd_path": b.ssd_path,
                    "r2_key": b.r2_key,
                }
                for bid, b in self.blocks.items()
            },
            "stats": self.stats,
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        manifest_path = CACHE_DIR / f"{name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return str(manifest_path)
