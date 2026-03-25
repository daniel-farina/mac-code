#!/usr/bin/env python3
"""
Triple benchmark: llama.cpp vs MLX vs MLX+TurboQuant+R2
Same prompts, same hardware, real numbers.
"""

import json, time, os, sys, subprocess
import urllib.request

SERVER = "http://localhost:8000"

def llm_call(messages, max_tokens=200):
    """Call whatever server is running on :8000."""
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    elapsed = time.time() - t0
    
    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    timings = d.get("timings", {})
    usage = d.get("usage", {})
    tokens = usage.get("completion_tokens", 0) or len(content.split())
    server_speed = timings.get("predicted_per_second", 0)
    wall_speed = tokens / elapsed if elapsed > 0 else 0
    
    return {
        "content": content,
        "tokens": tokens,
        "elapsed": elapsed,
        "server_speed": server_speed,
        "wall_speed": wall_speed,
    }

def wait_for_server(timeout=60):
    for i in range(timeout // 2):
        try:
            r = urllib.request.urlopen(urllib.request.Request(f"{SERVER}/health"), timeout=2)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except:
            pass
        time.sleep(2)
    return False

def start_llama_9b():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "mlx_engine"], capture_output=True)
    time.sleep(3)
    subprocess.Popen([
        "llama-server",
        "--model", os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf"),
        "--port", "8000", "--host", "127.0.0.1",
        "--flash-attn", "on", "--ctx-size", "4096",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--n-gpu-layers", "99", "--reasoning", "off", "-np", "1", "-t", "4",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wait_for_server()

def start_mlx_9b():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "mlx_engine"], capture_output=True)
    time.sleep(3)
    subprocess.Popen([
        sys.executable, "mlx_engine.py", "--model", "9b", "--port", "8000",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    cwd=os.path.expanduser("~/Desktop/mac-code-mlx"))
    return wait_for_server(timeout=120)

TESTS = [
    ("Math", [{"role": "user", "content": "What is 17 * 24? Just the number."}], 20),
    ("Reasoning", [{"role": "user", "content": "A bat and ball cost $1.10. Bat costs $1 more than ball. Ball cost? Just the number."}], 20),
    ("Short gen", [{"role": "user", "content": "Explain gravity in 2 sentences."}], 100),
    ("Long gen", [{"role": "user", "content": "Explain how backpropagation works in neural networks in detail."}], 300),
    ("Classification", [
        {"role": "system", "content": "Classify: search, shell, or chat. One word only."},
        {"role": "user", "content": "find me videos on my desktop"},
    ], 5),
]

# ═══════════════════════════════════════════
print()
print("=" * 65)
print("  TRIPLE BENCHMARK: llama.cpp vs MLX vs MLX+TurboQuant+R2")
print("  Mac mini M4, 16GB, Qwen3.5-9B")
print("=" * 65)

results = {}

# ── Benchmark 1: llama.cpp ──
print()
print("━" * 65)
print("  [1/3] llama.cpp (Q4_K_M, Metal)")
print("━" * 65)
if start_llama_9b():
    # Warmup
    llm_call([{"role": "user", "content": "hi"}], max_tokens=5)
    
    run_results = []
    for name, messages, max_tok in TESTS:
        r = llm_call(messages, max_tok)
        speed = r["server_speed"] if r["server_speed"] > 0 else r["wall_speed"]
        print(f"  {name:15s}  {speed:5.1f} tok/s  {r['elapsed']:5.1f}s  {r['tokens']:3d} tok  {r['content'][:40]}")
        run_results.append({"name": name, "speed": speed, "elapsed": r["elapsed"], "tokens": r["tokens"]})
    results["llama.cpp"] = run_results
else:
    print("  FAILED to start")

# ── Benchmark 2: MLX ──
print()
print("━" * 65)
print("  [2/3] MLX (4-bit, Apple native)")
print("━" * 65)
if start_mlx_9b():
    # Warmup
    llm_call([{"role": "user", "content": "hi"}], max_tokens=5)
    
    run_results = []
    for name, messages, max_tok in TESTS:
        r = llm_call(messages, max_tok)
        speed = r["server_speed"] if r["server_speed"] > 0 else r["wall_speed"]
        print(f"  {name:15s}  {speed:5.1f} tok/s  {r['elapsed']:5.1f}s  {r['tokens']:3d} tok  {r['content'][:40]}")
        run_results.append({"name": name, "speed": speed, "elapsed": r["elapsed"], "tokens": r["tokens"]})
    results["MLX"] = run_results
else:
    print("  FAILED to start")

# ── Benchmark 3: MLX + TurboQuant + R2 (context resume) ──
print()
print("━" * 65)
print("  [3/3] MLX + TurboQuant + R2 (persistent context)")
print("━" * 65)

# MLX should still be running from benchmark 2
if wait_for_server(timeout=10):
    from turboquant import compress_kv_cache, decompress_kv_cache, serialize_compressed, load_compressed
    from r2_store import upload_context, download_context, compress_cache, decompress_cache, is_configured
    from mlx_lm import load as mlx_load
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
    import mlx.core as mx
    from pathlib import Path

    model, tokenizer = mlx_load('mlx-community/Qwen3.5-9B-MLX-4bit')
    cache_dir = Path.home() / ".mac-code" / "kv-cache"
    
    # Simulate: process a document, save context, upload to R2, then resume
    doc = "Explain neural networks, backpropagation, gradient descent, activation functions, loss functions, optimizers, batch normalization, and dropout in detail. " * 3
    tokens = mx.array(tokenizer.encode(doc))
    
    # A) Process from scratch (prefill)
    cache = make_prompt_cache(model)
    t0 = time.time()
    logits = model(tokens[None], cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t0
    print(f"  Prefill ({len(tokens.tolist())} tokens):    {prefill_time:.2f}s")
    
    # B) Save with TurboQuant compression
    original_states = [c.state for c in cache]
    t0 = time.time()
    compressed, stats = compress_kv_cache(original_states, bits=4, group_size=64)
    save_result = serialize_compressed(compressed, str(cache_dir / "bench-turbo"))
    compress_save_time = time.time() - t0
    
    disk_size = sum(f.stat().st_size for f in cache_dir.glob("bench-turbo*")) / (1024*1024)
    print(f"  TurboQuant save:          {compress_save_time:.2f}s ({disk_size:.1f} MB, {stats['ratio']:.1f}x compressed)")
    
    # C) Save uncompressed for comparison
    uncompressed_path = str(cache_dir / "bench-uncompressed.safetensors")
    t0 = time.time()
    save_prompt_cache(uncompressed_path, cache)
    uncompressed_save_time = time.time() - t0
    uncompressed_size = os.path.getsize(uncompressed_path) / (1024*1024)
    print(f"  Uncompressed save:        {uncompressed_save_time:.2f}s ({uncompressed_size:.1f} MB)")
    
    # D) Upload to R2 (uncompressed path — compress_cache handles gzip)
    if is_configured():
        # Upload uncompressed
        t0 = time.time()
        r2_result = upload_context("bench-uncompressed")
        r2_upload_time = time.time() - t0
        print(f"  R2 upload (gzip):         {r2_upload_time:.1f}s ({r2_result.get('compressed_mb', 0):.1f} MB)")
        
        # Download from R2
        # Delete local first
        for f in cache_dir.glob("bench-uncompressed*"):
            f.unlink()
        
        t0 = time.time()
        r2_result = download_context("bench-uncompressed")
        r2_download_time = time.time() - t0
        print(f"  R2 download + decompress: {r2_download_time:.1f}s")
        
        # Load from disk
        t0 = time.time()
        loaded_cache, meta = load_prompt_cache(str(cache_dir / "bench-uncompressed.safetensors"), return_metadata=True)
        load_time = time.time() - t0
        print(f"  Load from disk:           {load_time:.4f}s")
    else:
        print("  R2 not configured — skipping cloud test")
        r2_upload_time = 0
        r2_download_time = 0
        load_time = 0
    
    # E) Load TurboQuant compressed
    t0 = time.time()
    loaded_compressed = load_compressed(str(cache_dir / "bench-turbo"))
    restored = decompress_kv_cache(loaded_compressed)
    turbo_load_time = time.time() - t0
    print(f"  TurboQuant load+decomp:   {turbo_load_time:.3f}s")
    
    print()
    print(f"  Resume comparison:")
    print(f"    Reprocess from scratch: {prefill_time:.2f}s")
    print(f"    Load from SSD:          {load_time:.4f}s  ({prefill_time/max(load_time,0.001):.0f}x faster)")
    print(f"    Load from R2:           {r2_download_time:.1f}s   ({prefill_time/max(r2_download_time,0.001):.1f}x faster)")
    print(f"    TurboQuant from SSD:    {turbo_load_time:.3f}s  ({prefill_time/max(turbo_load_time,0.001):.0f}x faster)")

# ── Summary ──
print()
print("=" * 65)
print("  SUMMARY")
print("=" * 65)

if "llama.cpp" in results and "MLX" in results:
    print()
    print(f"  {'Test':15s}  {'llama.cpp':>10s}  {'MLX':>10s}  {'Winner':>8s}  {'Δ':>6s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    
    for i, (name, _, _) in enumerate(TESTS):
        lspeed = results["llama.cpp"][i]["speed"]
        mspeed = results["MLX"][i]["speed"]
        winner = "MLX" if mspeed > lspeed else "llama"
        delta = max(mspeed, lspeed) / max(min(mspeed, lspeed), 0.1)
        print(f"  {name:15s}  {lspeed:>9.1f}  {mspeed:>9.1f}  {winner:>8s}  {delta:>5.1f}x")

print()
print("=" * 65)
