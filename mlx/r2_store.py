#!/usr/bin/env python3
"""
Cloudflare R2 storage for persistent KV cache.
Free tier: 10GB storage, no egress fees.

Setup:
    1. Create R2 bucket at https://dash.cloudflare.com → R2
    2. Create API token with read/write access
    3. Create config:

    mkdir -p ~/.mac-code
    cat > ~/.mac-code/r2-config.json << 'EOF'
    {
        "endpoint": "https://<account_id>.r2.cloudflarestorage.com",
        "access_key": "your-access-key",
        "secret_key": "your-secret-key",
        "bucket": "mac-code-kv"
    }
    EOF

Usage from agent:
    /save-context my-project     → compress + upload to R2
    /load-context my-project     → download from R2 + decompress
    /list-contexts               → show all saved contexts
    /share-context my-project    → generate signed URL
"""

import json
import os
import time
import gzip
import shutil
from pathlib import Path

CONFIG_PATH = Path.home() / ".mac-code" / "r2-config.json"
CACHE_DIR = Path.home() / ".mac-code" / "kv-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_r2_client():
    """Get boto3 S3 client configured for Cloudflare R2."""
    import boto3

    config = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())

    endpoint = config.get("endpoint") or os.environ.get("R2_ENDPOINT")
    access_key = config.get("access_key") or os.environ.get("R2_ACCESS_KEY")
    secret_key = config.get("secret_key") or os.environ.get("R2_SECRET_KEY")
    bucket = config.get("bucket") or os.environ.get("R2_BUCKET", "mac-code-kv")

    if not all([endpoint, access_key, secret_key]):
        return None, None

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return client, bucket


def is_configured():
    """Check if R2 is configured."""
    client, bucket = get_r2_client()
    return client is not None


def compress_cache(name):
    """Gzip compress a saved cache file."""
    src = CACHE_DIR / f"{name}.safetensors"
    dst = CACHE_DIR / f"{name}.safetensors.gz"

    if not src.exists():
        return None

    with open(src, "rb") as f_in:
        with gzip.open(str(dst), "wb", compresslevel=6) as f_out:
            f_out.write(f_in.read())

    original = src.stat().st_size
    compressed = dst.stat().st_size

    return {
        "original_mb": original / (1024 * 1024),
        "compressed_mb": compressed / (1024 * 1024),
        "ratio": original / compressed if compressed > 0 else 0,
        "path": str(dst),
    }


def decompress_cache(name):
    """Decompress gzipped cache file."""
    src = CACHE_DIR / f"{name}.safetensors.gz"
    dst = CACHE_DIR / f"{name}.safetensors"

    if not src.exists():
        return False

    with gzip.open(str(src), "rb") as f_in:
        with open(dst, "wb") as f_out:
            f_out.write(f_in.read())

    return True


def upload_context(name):
    """Compress and upload KV cache to R2."""
    client, bucket = get_r2_client()
    if not client:
        return {"error": "R2 not configured. Run /r2-setup for instructions."}

    # Check cache exists locally
    cache_file = CACHE_DIR / f"{name}.safetensors"
    if not cache_file.exists():
        return {"error": f"No local cache found: {name}"}

    # Compress
    comp = compress_cache(name)
    if not comp:
        return {"error": "Compression failed"}

    # Upload compressed file
    key = f"kv-cache/{name}.safetensors.gz"
    t0 = time.time()
    client.upload_file(comp["path"], bucket, key)
    upload_time = time.time() - t0

    # Upload metadata
    meta = {
        "name": name,
        "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_mb": comp["original_mb"],
        "compressed_mb": comp["compressed_mb"],
        "compression_ratio": comp["ratio"],
    }

    # Read local metadata if exists
    meta_file = CACHE_DIR / f"{name}.meta.json"
    if meta_file.exists():
        local_meta = json.loads(meta_file.read_text())
        meta.update(local_meta)

    client.put_object(
        Bucket=bucket,
        Key=f"kv-cache/{name}.meta.json",
        Body=json.dumps(meta, indent=2).encode(),
        ContentType="application/json",
    )

    return {
        "name": name,
        "compressed_mb": comp["compressed_mb"],
        "upload_time": upload_time,
        "speed_mbps": comp["compressed_mb"] / upload_time if upload_time > 0 else 0,
    }


def download_context(name):
    """Download KV cache from R2 and decompress."""
    client, bucket = get_r2_client()
    if not client:
        return {"error": "R2 not configured"}

    gz_path = CACHE_DIR / f"{name}.safetensors.gz"
    meta_path = CACHE_DIR / f"{name}.meta.json"

    # Download compressed cache
    key = f"kv-cache/{name}.safetensors.gz"
    t0 = time.time()
    try:
        client.download_file(bucket, key, str(gz_path))
    except Exception as e:
        return {"error": f"Download failed: {e}"}
    download_time = time.time() - t0

    # Download metadata
    try:
        client.download_file(bucket, f"kv-cache/{name}.meta.json", str(meta_path))
    except Exception:
        pass

    # Decompress
    decompress_cache(name)
    total_time = time.time() - t0

    size_mb = gz_path.stat().st_size / (1024 * 1024) if gz_path.exists() else 0

    return {
        "name": name,
        "compressed_mb": size_mb,
        "download_time": download_time,
        "total_time": total_time,
        "speed_mbps": size_mb / download_time if download_time > 0 else 0,
    }


def list_remote_contexts():
    """List all contexts stored in R2."""
    client, bucket = get_r2_client()
    if not client:
        return []

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix="kv-cache/")
    except Exception:
        return []

    contexts = {}
    for obj in response.get("Contents", []):
        key = obj["Key"]
        name = key.replace("kv-cache/", "").replace(".safetensors.gz", "").replace(".meta.json", "")
        if name not in contexts:
            contexts[name] = {"name": name, "r2_size_mb": 0}
        if key.endswith(".gz"):
            contexts[name]["r2_size_mb"] = obj["Size"] / (1024 * 1024)
            contexts[name]["last_modified"] = obj["LastModified"].isoformat()

    return list(contexts.values())


def list_local_contexts():
    """List all locally saved contexts."""
    contexts = []
    for f in CACHE_DIR.glob("*.safetensors"):
        name = f.stem
        meta_file = CACHE_DIR / f"{name}.meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta["name"] = name
        meta["local_mb"] = f.stat().st_size / (1024 * 1024)
        gz = CACHE_DIR / f"{name}.safetensors.gz"
        meta["compressed_mb"] = gz.stat().st_size / (1024 * 1024) if gz.exists() else 0
        contexts.append(meta)
    return contexts


def delete_context(name, remote=False):
    """Delete a context locally and optionally from R2."""
    # Local
    for suffix in [".safetensors", ".safetensors.gz", ".meta.json"]:
        path = CACHE_DIR / f"{name}{suffix}"
        if path.exists():
            path.unlink()

    # Remote
    if remote:
        client, bucket = get_r2_client()
        if client:
            for suffix in [".safetensors.gz", ".meta.json"]:
                try:
                    client.delete_object(Bucket=bucket, Key=f"kv-cache/{name}{suffix}")
                except Exception:
                    pass

    return True


def share_context(name, expires=3600):
    """Generate a pre-signed URL for sharing a context."""
    client, bucket = get_r2_client()
    if not client:
        return None

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": f"kv-cache/{name}.safetensors.gz"},
        ExpiresIn=expires,
    )
    return url


def setup_instructions():
    """Return setup instructions for R2."""
    return """
To enable cloud context storage (free):

1. Go to https://dash.cloudflare.com → R2 → Create bucket
   Name it: mac-code-kv

2. Go to R2 → Manage API tokens → Create token
   Permissions: Object Read & Write
   Copy the Access Key ID and Secret Access Key

3. Create config file:

   mkdir -p ~/.mac-code
   cat > ~/.mac-code/r2-config.json << 'EOF'
   {
       "endpoint": "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com",
       "access_key": "YOUR_ACCESS_KEY",
       "secret_key": "YOUR_SECRET_KEY",
       "bucket": "mac-code-kv"
   }
   EOF

4. Test: /list-contexts

Free tier: 10GB storage, no egress fees, 1M writes/month.
"""
