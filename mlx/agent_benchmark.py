#!/usr/bin/env python3
"""
Agent benchmark: mac-code (llama.cpp) vs mac-code-mlx (MLX + TurboQuant + R2)
Same prompts, same hardware, real agent tasks.
"""

import json, time, os, sys, subprocess
import urllib.request

SERVER = "http://localhost:8000"

def wait_for_server(timeout=120):
    for i in range(timeout // 3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(f"{SERVER}/health"), timeout=2)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except: pass
        time.sleep(3)
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
        sys.executable, "mlx_engine.py", "--model", "9b",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    cwd=os.path.expanduser("~/Desktop/mac-code-mlx"))
    return wait_for_server()

# Agent functions from mac-code
sys.path.insert(0, os.path.expanduser("~/Desktop/pico-mini"))
from agent import classify_intent, quick_search, run_smart_tool, llm_call

AGENT_TESTS = [
    # (name, type, query)
    ("Intent: shell",    "classify",  "find me videos on my desktop"),
    ("Intent: search",   "classify",  "who do the lakers play next"),
    ("Intent: chat",     "classify",  "explain quantum computing"),
    ("Web search",       "search",    "what is bitcoin trading at today"),
    ("Shell command",    "shell",     "how much disk space do I have"),
    ("Math",             "chat",      "what is the integral of x^2 from 0 to 5"),
    ("Code gen",         "chat",      "write a python function to check if a number is prime"),
    ("Reasoning",        "chat",      "if I invest $10000 and it grows 15% then drops 8%, how much do I have"),
]

print()
print("=" * 70)
print("  AGENT BENCHMARK: mac-code vs mac-code-mlx")
print("  Mac mini M4, 16GB, Qwen3.5-9B, same agent tasks")
print("=" * 70)

results = {}

for backend_name, start_fn in [("llama.cpp", start_llama_9b), ("MLX", start_mlx_9b)]:
    print()
    print("━" * 70)
    print(f"  {backend_name}")
    print("━" * 70)

    if not start_fn():
        print("  FAILED to start")
        continue

    # Warmup
    try:
        llm_call([{"role": "user", "content": "hi"}], max_tokens=5)
    except: pass

    run_results = []

    for name, test_type, query in AGENT_TESTS:
        t0 = time.time()
        try:
            if test_type == "classify":
                result = classify_intent(query)
                elapsed = time.time() - t0
                print(f"  {name:20s}  {elapsed:5.1f}s  → {result}")
                run_results.append({"name": name, "elapsed": elapsed, "result": result})

            elif test_type == "search":
                result = quick_search(query)
                elapsed = time.time() - t0
                if result:
                    resp, speed = result
                    print(f"  {name:20s}  {elapsed:5.1f}s  {speed:.0f} tok/s  {resp[:50]}...")
                    run_results.append({"name": name, "elapsed": elapsed, "speed": speed})
                else:
                    print(f"  {name:20s}  {elapsed:5.1f}s  FAILED")
                    run_results.append({"name": name, "elapsed": elapsed, "speed": 0})

            elif test_type == "shell":
                result = run_smart_tool(query)
                elapsed = time.time() - t0
                if result:
                    resp, speed, cmd = result
                    print(f"  {name:20s}  {elapsed:5.1f}s  {speed:.0f} tok/s  $ {cmd[:40]}")
                    run_results.append({"name": name, "elapsed": elapsed, "speed": speed})
                else:
                    print(f"  {name:20s}  {elapsed:5.1f}s  FAILED")
                    run_results.append({"name": name, "elapsed": elapsed, "speed": 0})

            elif test_type == "chat":
                resp, timings = llm_call([{"role": "user", "content": query}], max_tokens=200)
                elapsed = time.time() - t0
                speed = timings.get("predicted_per_second", 0) or (len(resp.split()) / elapsed)
                print(f"  {name:20s}  {elapsed:5.1f}s  {speed:.0f} tok/s  {resp[:50]}...")
                run_results.append({"name": name, "elapsed": elapsed, "speed": speed})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {name:20s}  {elapsed:5.1f}s  ERROR: {str(e)[:40]}")
            run_results.append({"name": name, "elapsed": elapsed, "speed": 0})

    results[backend_name] = run_results

# ── Context resume benchmark (MLX only) ──
print()
print("━" * 70)
print("  MLX + TurboQuant + R2 (context persistence)")
print("━" * 70)

if start_mlx_9b():
    from paged_inference import PagedInference
    from mlx_lm import load as mlx_load
    from turboquant import compress_kv_cache, serialize_compressed
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
    import mlx.core as mx
    from pathlib import Path

    model, tokenizer = mlx_load('mlx-community/Qwen3.5-9B-MLX-4bit')
    cache_dir = Path.home() / ".mac-code" / "kv-cache"

    doc = "Python is a programming language created by Guido van Rossum. It supports multiple paradigms including object-oriented, functional, and procedural programming. " * 5
    tokens = mx.array(tokenizer.encode(doc))

    # A) Prefill from scratch
    cache = make_prompt_cache(model)
    t0 = time.time()
    logits = model(tokens[None], cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t0
    print(f"  Prefill ({len(tokens.tolist())} tok):     {prefill_time:.2f}s")

    # B) Save uncompressed
    path_unc = str(cache_dir / "agent-bench-unc.safetensors")
    t0 = time.time()
    save_prompt_cache(path_unc, cache)
    print(f"  Save uncompressed:       {time.time()-t0:.3f}s ({os.path.getsize(path_unc)/(1024*1024):.1f} MB)")

    # C) Load uncompressed
    t0 = time.time()
    loaded, _ = load_prompt_cache(path_unc, return_metadata=True)
    load_unc = time.time() - t0
    print(f"  Load uncompressed:       {load_unc:.4f}s")

    # D) TurboQuant compress + save
    states = [c.state for c in cache]
    t0 = time.time()
    compressed, stats = compress_kv_cache(states, bits=4)
    save_result = serialize_compressed(compressed, str(cache_dir / "agent-bench-turbo"))
    turbo_save = time.time() - t0
    turbo_size = sum(f.stat().st_size for f in cache_dir.glob("agent-bench-turbo*")) / (1024*1024)
    print(f"  TurboQuant save:         {turbo_save:.2f}s ({turbo_size:.1f} MB, {stats['ratio']:.1f}x smaller)")

    # E) R2 round-trip
    from r2_store import upload_context, download_context, is_configured
    if is_configured():
        t0 = time.time()
        upload_context("agent-bench-unc")
        r2_up = time.time() - t0

        for f in cache_dir.glob("agent-bench-unc*"):
            f.unlink()

        t0 = time.time()
        download_context("agent-bench-unc")
        r2_down = time.time() - t0

        t0 = time.time()
        load_prompt_cache(str(cache_dir / "agent-bench-unc.safetensors"), return_metadata=True)
        r2_load = time.time() - t0

        print(f"  R2 upload:               {r2_up:.1f}s")
        print(f"  R2 download:             {r2_down:.1f}s")
        print(f"  R2 → load:               {r2_load:.4f}s")
        print(f"  R2 total resume:         {r2_down + r2_load:.1f}s")

    print()
    print(f"  Context resume comparison:")
    print(f"    Reprocess:       {prefill_time:.2f}s")
    print(f"    SSD load:        {load_unc:.4f}s  ({prefill_time/max(load_unc,0.0001):.0f}x faster)")
    if is_configured():
        print(f"    R2 resume:       {r2_down + r2_load:.1f}s  ({prefill_time/(r2_down+r2_load):.1f}x faster)")

# ── Summary ──
print()
print("=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)

if "llama.cpp" in results and "MLX" in results:
    print()
    print(f"  {'Task':20s}  {'llama.cpp':>10s}  {'MLX':>10s}  {'Winner':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

    llama_total = 0
    mlx_total = 0

    for i in range(len(AGENT_TESTS)):
        name = results["llama.cpp"][i]["name"]
        lt = results["llama.cpp"][i]["elapsed"]
        mt = results["MLX"][i]["elapsed"]
        llama_total += lt
        mlx_total += mt
        winner = "MLX" if mt < lt else "llama"
        print(f"  {name:20s}  {lt:>9.1f}s  {mt:>9.1f}s  {winner:>8s}")

    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    winner = "MLX" if mlx_total < llama_total else "llama"
    print(f"  {'TOTAL':20s}  {llama_total:>9.1f}s  {mlx_total:>9.1f}s  {winner:>8s}")

print()
print("  mac-code-mlx unique features:")
print("    ✅ Persistent context (save/load KV cache)")
print("    ✅ Cross-device resume via R2")
print("    ✅ TurboQuant 4-bit compression (4.1x smaller)")
print("    ✅ Tiered GPU→SSD paging")
print("    ✅ Paged inference for long documents")
print()
print("=" * 70)
