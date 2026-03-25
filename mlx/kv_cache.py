#!/usr/bin/env python3
"""
KV Cache manager — save, load, compress, decompress.
Enables persistent context across sessions and devices.
"""

import json
import time
import gzip
import hashlib
from pathlib import Path
from datetime import datetime

CACHE_DIR = Path.home() / ".mac-code" / "kv-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_kv_cache(kv_tensors, name, metadata=None):
    """Save KV cache tensors to disk with metadata."""
    try:
        import mlx.core as mx

        cache_path = CACHE_DIR / name
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        tensor_path = cache_path / "kv_cache.npz"
        mx.savez(str(tensor_path), **{f"layer_{i}": t for i, t in enumerate(kv_tensors)})

        # Save metadata
        meta = {
            "name": name,
            "created": datetime.now().isoformat(),
            "num_layers": len(kv_tensors),
            "total_bytes": sum(t.nbytes for t in kv_tensors),
            "dtype": str(kv_tensors[0].dtype) if kv_tensors else "unknown",
        }
        if metadata:
            meta.update(metadata)

        with open(cache_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    except ImportError:
        # Fallback: save as numpy
        import numpy as np

        cache_path = CACHE_DIR / name
        cache_path.mkdir(parents=True, exist_ok=True)

        tensor_path = cache_path / "kv_cache.npz"
        np.savez_compressed(str(tensor_path), *kv_tensors)

        meta = {
            "name": name,
            "created": datetime.now().isoformat(),
            "num_layers": len(kv_tensors),
            "format": "numpy",
        }
        if metadata:
            meta.update(metadata)

        with open(cache_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return meta


def load_kv_cache(name):
    """Load KV cache tensors from disk."""
    cache_path = CACHE_DIR / name

    if not cache_path.exists():
        return None, None

    # Load metadata
    meta_path = cache_path / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Load tensors
    tensor_path = cache_path / "kv_cache.npz"

    try:
        import mlx.core as mx
        data = mx.load(str(tensor_path))
        tensors = [data[f"layer_{i}"] for i in range(metadata.get("num_layers", len(data)))]
        return tensors, metadata
    except ImportError:
        import numpy as np
        data = np.load(str(tensor_path))
        tensors = [data[f"arr_{i}"] for i in range(len(data.files))]
        return tensors, metadata


def compress_kv_cache(name):
    """Compress saved KV cache with gzip for R2 upload."""
    cache_path = CACHE_DIR / name / "kv_cache.npz"
    compressed_path = CACHE_DIR / name / "kv_cache.npz.gz"

    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as f_in:
        with gzip.open(str(compressed_path), "wb", compresslevel=6) as f_out:
            f_out.write(f_in.read())

    original = cache_path.stat().st_size
    compressed = compressed_path.stat().st_size
    ratio = original / compressed if compressed > 0 else 0

    return {
        "original_bytes": original,
        "compressed_bytes": compressed,
        "ratio": ratio,
        "path": str(compressed_path),
    }


def decompress_kv_cache(name):
    """Decompress gzipped KV cache after R2 download."""
    compressed_path = CACHE_DIR / name / "kv_cache.npz.gz"
    cache_path = CACHE_DIR / name / "kv_cache.npz"

    if not compressed_path.exists():
        return False

    with gzip.open(str(compressed_path), "rb") as f_in:
        with open(cache_path, "wb") as f_out:
            f_out.write(f_in.read())

    return True


def list_cached_contexts():
    """List all saved KV cache contexts."""
    contexts = []
    for path in CACHE_DIR.iterdir():
        if path.is_dir():
            meta_path = path / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                # Check file sizes
                npz = path / "kv_cache.npz"
                gz = path / "kv_cache.npz.gz"
                meta["disk_size_mb"] = npz.stat().st_size / (1024*1024) if npz.exists() else 0
                meta["compressed_mb"] = gz.stat().st_size / (1024*1024) if gz.exists() else 0
                contexts.append(meta)
    return sorted(contexts, key=lambda x: x.get("created", ""), reverse=True)


def delete_cached_context(name):
    """Delete a saved KV cache context."""
    import shutil
    cache_path = CACHE_DIR / name
    if cache_path.exists():
        shutil.rmtree(cache_path)
        return True
    return False
