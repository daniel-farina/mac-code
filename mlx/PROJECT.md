# mac code MLX — Persistent KV Cache with Cloudflare R2

## Mission

Build the first MLX-native inference engine with cloud-backed KV cache for Apple Silicon. Persistent, shareable, cross-device AI context — for free.

## The Problem

Every local AI agent today has amnesia. Close the session, lose the context. Switch devices, start over. Process a codebase, wait 30+ seconds next time.

Cloud AI (Claude, ChatGPT) solves this with server-side session storage. But your data leaves your machine and you pay per token.

We want the best of both: **local inference with cloud-persistent context.**

## The Solution

```
┌─────────────────────────────────────────────────┐
│  mac code MLX agent                             │
│  Same agent as mac-code (search, shell, chat)   │
├─────────────────────────────────────────────────┤
│  MLX inference engine (Apple native)            │
│  25% faster than llama.cpp on Apple Silicon     │
├─────────────────────────────────────────────────┤
│  Tiered KV Cache                                │
│                                                 │
│  HOT:  GPU unified memory (current tokens)      │
│  WARM: SSD (overflow, ~0.1ms reads)             │
│  COLD: Cloudflare R2 (persistent, ~50ms reads)  │
│        Free for first 10GB, no egress fees      │
├─────────────────────────────────────────────────┤
│  Apple Silicon — M1/M2/M3/M4                    │
│  Unified memory + Metal GPU + fast SSD          │
└─────────────────────────────────────────────────┘
```

## What This Enables

### 1. Persistent Context
```
Monday:   Analyze entire codebase → KV cache saved to R2 (64MB, 1.3s upload)
Tuesday:  "load-context my-project" → resumes in 1.3s (not 30s reprocessing)
```

### 2. Shared Context Across Devices
```
Mac mini:    Process document → upload KV to R2
MacBook:     Download KV from R2 → continue conversation
Mac Studio:  Same context, bigger model, same R2 store
```

### 3. Team Shared Knowledge
```
Developer A: Analyzes the API codebase → saves to R2 as "api-context"
Developer B: Loads "api-context" → asks questions about the API
No reprocessing. Instant shared understanding.
```

### 4. Unlimited Context via Tiered Paging
```
Current:  64K tokens max (limited by GPU memory)
With R2:  Process 256K+ tokens, page cold KV blocks to R2
          Hot blocks stay in GPU, warm on SSD, cold on R2
```

## Architecture

### Phase 1: MLX Inference (replace llama.cpp)
- [ ] MLX server with OpenAI-compatible API on localhost:8000
- [ ] Support Qwen3.5-9B-MLX-4bit and Qwen3.5-35B-A3B-4bit
- [ ] Verify agent.py works with MLX backend (classify, search, shell)
- [ ] Benchmark: MLX vs llama.cpp on same tasks
- [ ] Quantized KV cache in MLX (equivalent to llama.cpp q4_0)

### Phase 2: KV Cache Save/Load
- [ ] Extract KV cache tensors from MLX model after inference
- [ ] Serialize KV cache to disk (safetensors or numpy format)
- [ ] Compress KV cache (q4_0 quantization → 3.5x smaller)
- [ ] Load KV cache from disk and inject into model state
- [ ] Verify: loaded KV cache produces same outputs as live KV
- [ ] Add /save-context and /load-context commands to agent.py

### Phase 3: Cloudflare R2 Integration
- [ ] R2 upload/download via boto3 (S3-compatible API)
- [ ] Compress before upload (q4_0 + optional gzip)
- [ ] Context manifest: JSON metadata (model, timestamp, token count, hash)
- [ ] /save-context <name> → compress → upload to R2
- [ ] /load-context <name> → download from R2 → decompress → inject
- [ ] /list-contexts → show all saved contexts with metadata
- [ ] /share-context <name> <user> → generate signed URL for sharing

### Phase 4: Tiered KV Paging
- [ ] Monitor KV cache size during inference
- [ ] When KV exceeds GPU budget, page cold blocks to SSD
- [ ] When SSD KV exceeds threshold, page to R2
- [ ] Prefetch from R2 when context is likely needed
- [ ] Benchmark: latency impact of R2 paging at different context sizes

### Phase 5: TurboQuant Integration
- [ ] Implement 3-bit KV compression (PolarQuant + QJL from TurboQuant paper)
- [ ] Compare vs q4_0: size reduction and quality
- [ ] Optimize R2 transfer size with 3-bit compression
- [ ] Target: 95% reduction in KV transfer payload

## Key Research References

- **Apple "LLM in a Flash"** — SSD paging via unified memory
  https://machinelearning.apple.com/research/efficient-large-language

- **Google TurboQuant** — Extreme KV cache compression with zero quality loss
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

- **LMCache** — Multi-tier KV caching with S3 support (GPU → CPU → S3)
  https://github.com/LMCache/LMCache

- **CacheGen** — KV cache compression + S3 storage/retrieval
  https://arxiv.org/abs/2310.07240

- **TinyLoRA** — 13-parameter fine-tuning via RL for self-improvement
  https://arxiv.org/html/2602.04118v1

## Technical Details

### KV Cache Sizes (Qwen3.5-9B at q4_0)

| Context | KV Size | Compressed (3-bit) | R2 Upload | R2 Download |
|---|---|---|---|---|
| 4K | 20 MB | 5 MB | 0.5s | 0.1s |
| 64K | 288 MB | 64 MB | 6.4s | 1.3s |
| 256K | 1,024 MB | 256 MB | 25.6s | 5.1s |

### R2 Costs
- First 10 GB: **Free**
- No egress fees (unlike AWS S3)
- Class A operations (writes): 1M free/month
- Class B operations (reads): 10M free/month
- For most users: **$0/month**

### MLX Advantages Over llama.cpp
- Python-native: direct tensor access for KV manipulation
- Zero-copy unified memory: no Metal overhead
- 25% faster inference on 9B model
- Lazy evaluation: operation fusion
- Native Apple Silicon: designed for M-series chips

## File Structure

```
mac-code-mlx/
├── PROJECT.md          ← This file
├── agent.py            ← CLI agent (forked from mac-code)
├── mlx_engine.py       ← MLX inference engine with KV cache access
├── kv_cache.py         ← KV cache save/load/compress/decompress
├── r2_store.py         ← Cloudflare R2 upload/download/list
├── chat.py             ← Lightweight streaming chat
├── dashboard.py        ← Server monitor
├── web/                ← Retro Mac web UI
├── setup.sh            ← One-command install
├── config.example.json ← Agent config
├── CLAUDE.md           ← Setup instructions for Claude Code
└── tests/
    ├── test_kv_cache.py
    ├── test_r2_store.py
    └── test_mlx_engine.py
```

## Development Priorities

1. **Get MLX inference working with agent.py** (Phase 1)
2. **KV cache save/load to disk** (Phase 2)
3. **R2 integration** (Phase 3)
4. **Benchmark everything** (ongoing)
5. **Tiered paging** (Phase 4 — research)
6. **TurboQuant 3-bit** (Phase 5 — research)

## Success Metrics

- [ ] MLX backend passes all 10/10 agent tests (search, shell, chat)
- [ ] KV cache save/load round-trips with zero quality loss
- [ ] R2 upload/download under 2 seconds for 64K context
- [ ] Agent resumes context from R2 faster than reprocessing
- [ ] Cross-device context sharing works between two Macs
- [ ] Published benchmarks: MLX+R2 vs llama.cpp vs cloud APIs
