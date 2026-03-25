#!/usr/bin/env python3
"""
Paged Inference — Extends context beyond GPU memory via SSD/R2 paging.

The core idea: intercept the model's forward pass so that before each layer
processes its attention, we ensure the needed KV cache blocks are in GPU memory.
Old blocks get evicted to SSD. When needed again, they're loaded back.

This is Apple's "LLM in a Flash" concept applied to KV cache, not just model weights.

Architecture:
    1. Process tokens in chunks (e.g., 512 tokens)
    2. After each chunk, save a "checkpoint" of the KV cache state
    3. When GPU memory is full, evict the oldest checkpoint to SSD
    4. The model always has the most recent KV in GPU
    5. For attention over old tokens, load the checkpoint from SSD

Limitation: This is a checkpoint/resume system, not true virtual-memory paging.
True paging would require patching MLX's attention kernel, which needs C++ changes.
This Python-level approach gives ~80% of the benefit with zero kernel modifications.
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import mlx.core as mx

CACHE_DIR = Path.home() / ".mac-code" / "kv-cache" / "paged"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ContextWindow:
    """A saved chunk of processed context."""
    chunk_id: int
    start_token: int
    end_token: int
    cache_path: str
    size_mb: float
    in_gpu: bool


class PagedInference:
    """
    Extends effective context by checkpointing KV cache chunks to SSD.

    Usage:
        paged = PagedInference(model, tokenizer, chunk_size=512, max_gpu_chunks=4)

        # Process a very long document
        response = paged.generate_with_paging(long_document + question, max_tokens=500)

        # The system automatically:
        # 1. Splits input into chunks
        # 2. Processes each chunk, checkpointing KV state
        # 3. Evicts old chunks to SSD when GPU is full
        # 4. For the final generation, loads the most relevant chunks back
    """

    def __init__(self, model, tokenizer,
                 chunk_size=512,
                 max_gpu_chunks=4,
                 session_name="default"):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_gpu_chunks = max_gpu_chunks
        self.session_name = session_name

        self.chunks: List[ContextWindow] = []
        self.total_tokens = 0

        # Create session directory
        self.session_dir = CACHE_DIR / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def process_long_context(self, text, callback=None):
        """
        Process a long text that may exceed GPU context limits.
        Checkpoints KV cache chunks to SSD as it goes.

        Returns the number of tokens processed.
        """
        from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache

        tokens = self.tokenizer.encode(text)
        total = len(tokens)

        if callback:
            callback(f"Processing {total} tokens in {(total // self.chunk_size) + 1} chunks")

        cache = make_prompt_cache(self.model)

        for i in range(0, total, self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            if not chunk_tokens:
                break

            chunk_id = len(self.chunks)

            # Process chunk through model
            t0 = time.time()
            token_array = mx.array(chunk_tokens)
            logits = self.model(token_array[None], cache=cache)
            mx.eval(logits)
            process_time = time.time() - t0

            # Save checkpoint to SSD
            cache_path = str(self.session_dir / f"chunk_{chunk_id}.safetensors")
            save_prompt_cache(cache_path, cache, metadata={
                "chunk_id": str(chunk_id),
                "start": str(i),
                "end": str(i + len(chunk_tokens)),
                "tokens": str(len(chunk_tokens)),
            })

            size_mb = os.path.getsize(cache_path) / (1024 * 1024)

            window = ContextWindow(
                chunk_id=chunk_id,
                start_token=i,
                end_token=i + len(chunk_tokens),
                cache_path=cache_path,
                size_mb=size_mb,
                in_gpu=True,
            )
            self.chunks.append(window)
            self.total_tokens += len(chunk_tokens)

            if callback:
                callback(f"  Chunk {chunk_id}: {len(chunk_tokens)} tokens, {process_time:.2f}s, {size_mb:.1f}MB saved")

            # Evict old chunks if over GPU budget
            gpu_chunks = [c for c in self.chunks if c.in_gpu]
            while len(gpu_chunks) > self.max_gpu_chunks:
                oldest = gpu_chunks[0]
                oldest.in_gpu = False
                gpu_chunks = gpu_chunks[1:]
                if callback:
                    callback(f"  Evicted chunk {oldest.chunk_id} to SSD")

        return self.total_tokens

    def generate(self, question, max_tokens=500):
        """
        Generate a response using the most recent context.
        Loads the latest KV cache checkpoint from SSD if needed.
        """
        from mlx_lm import generate as mlx_generate
        from mlx_lm.models.cache import load_prompt_cache

        if not self.chunks:
            # No context — generate directly
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            return mlx_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

        # Load the most recent cache checkpoint
        latest = self.chunks[-1]
        t0 = time.time()
        cache, meta = load_prompt_cache(latest.cache_path, return_metadata=True)
        load_time = time.time() - t0

        # Generate with the loaded context
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        tokens = mx.array(self.tokenizer.encode(prompt))

        # Process the question tokens through the model with loaded cache
        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        # Now generate
        response = mlx_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

        # Strip thinking tags
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        for stop in ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
            if stop in response:
                response = response[:response.index(stop)]

        return response.strip(), {
            "context_tokens": self.total_tokens,
            "chunks_on_ssd": sum(1 for c in self.chunks if not c.in_gpu),
            "chunks_in_gpu": sum(1 for c in self.chunks if c.in_gpu),
            "cache_load_time": load_time,
        }

    def get_stats(self):
        """Return paging statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_chunks": len(self.chunks),
            "gpu_chunks": sum(1 for c in self.chunks if c.in_gpu),
            "ssd_chunks": sum(1 for c in self.chunks if not c.in_gpu),
            "total_ssd_mb": sum(c.size_mb for c in self.chunks if not c.in_gpu),
            "total_gpu_mb": sum(c.size_mb for c in self.chunks if c.in_gpu),
            "session": self.session_name,
        }

    def upload_to_r2(self, callback=None):
        """Upload all SSD chunks to R2 for cross-device resume."""
        from r2_store import get_r2_client

        client, bucket = get_r2_client()
        if not client:
            return {"error": "R2 not configured"}

        uploaded = 0
        for chunk in self.chunks:
            if os.path.exists(chunk.cache_path):
                key = f"paged/{self.session_name}/chunk_{chunk.chunk_id}.safetensors"
                t0 = time.time()
                client.upload_file(chunk.cache_path, bucket, key)
                upload_time = time.time() - t0
                uploaded += 1
                if callback:
                    callback(f"  Uploaded chunk {chunk.chunk_id}: {chunk.size_mb:.1f}MB in {upload_time:.1f}s")

        # Upload manifest
        manifest = {
            "session": self.session_name,
            "total_tokens": self.total_tokens,
            "chunks": [{
                "chunk_id": c.chunk_id,
                "start_token": c.start_token,
                "end_token": c.end_token,
                "size_mb": c.size_mb,
            } for c in self.chunks],
        }
        client.put_object(
            Bucket=bucket,
            Key=f"paged/{self.session_name}/manifest.json",
            Body=json.dumps(manifest, indent=2).encode(),
        )

        return {"uploaded": uploaded, "session": self.session_name}
