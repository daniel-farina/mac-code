# 🍎 mac code

**Open-source AI agent on Apple Silicon. Two backends. Persistent memory. $0/month.**

---

## What It Does

An AI agent that runs locally on your Mac — web search, shell commands, file operations, code generation. Same things Claude Code does, but the LLM runs on your desk.

Two backends, one agent:

| Backend | Speed | Context | Persistent Memory | Best For |
|---|---|---|---|---|
| **llama.cpp** | 16-30 tok/s | 64K (9B) / 12K (35B) | No | 35B MoE via SSD paging |
| **MLX** | 20 tok/s (+25%) | 64K | **Yes — save/load/R2 sync** | Persistent context, cross-device |

Both run the same `agent.py`. Same slash commands. Same web search. Same shell tools.

---

## Quick Start

### Option A: llama.cpp (proven, supports 35B MoE)

```bash
brew install llama.cpp
pip3 install rich ddgs

# Download model
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

# Run agent (auto-starts llama-server if not running)
python3 agent.py
```

The agent auto-detects and starts the LLM server on first launch. You can also start it manually:

```bash
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4
```

### Option B: MLX (faster, persistent context)

```bash
pip3 install mlx-lm rich ddgs --break-system-packages

# Start MLX engine (downloads model on first run)
python3 mlx/mlx_engine.py

# Run agent
python3 agent.py
```

---

## Key Findings

### 35B Model via SSD Paging — 30 tok/s on $600 Hardware

Qwen3.5-35B-A3B (10.6 GB) doesn't fit in 16 GB RAM. macOS pages from the SSD. On NVIDIA, paging gives 1.6 tok/s. On Apple Silicon: **30 tok/s. 18.6x faster.**

### Tool Calling Works at 2.6 Bits Per Weight

The 35B at IQ2_M quantization was supposed to have broken instruction following. JSON function calls DO break. But our LLM-as-Router (simple text classification) works perfectly — 8/8 correct.

### Quantized KV Cache Doubles Context For Free

Two server flags (`--cache-type-k q4_0 --cache-type-v q4_0`) shrink KV cache from 1024 MB to 288 MB. The 9B goes from 32K to 64K context. Zero quality loss.

### MLX Is 25% Faster for Generation

Same model, same hardware: MLX 20 tok/s vs llama.cpp 16 tok/s on sustained generation.

### Persistent Context — Process Once, Resume Anywhere (MLX only)

Save KV cache to disk in 0.04s. Load in 0.0003s (6,677x faster than reprocessing). Upload to Cloudflare R2 for cross-device resume in 1.5s. TurboQuant compression makes storage 4x smaller.

---

## How It Works

The LLM classifies its own intent:

```
"find me videos on my desktop"  → LLM says "shell"  → generates find command → executes
"who do the lakers play next"   → LLM says "search" → rewrites query → DuckDuckGo → answers
"create a snake game"           → LLM says "code"   → writes file → opens in browser
"open gmail and read my emails" → LLM says "code"   → MCP: navigate → read → summarize
"explain quantum computing"     → LLM says "chat"   → streams directly
```

Four paths. No hardcoded rules. Upgrading the model upgrades every capability.

---

## Coding Agent

The agent can write, edit, and run code - like Claude Code or Codex, but local:

```
"create a tetris game"       → writes tetris.html (complete) → opens in browser
"add a pause button"         → reads file → makes surgical edit → reopens
"the score is wrong, fix it" → reads file → fixes the bug → auto-verifies syntax
```

Features:
- **READ:** reads files before editing (auto-triggered)
- **EDIT:** surgical search-and-replace on existing files (default for modifications)
- **FILE:** creates new files with complete code
- **RUN:** executes shell commands (open browser, run scripts, npm install)
- **Fuzzy matching** - EDIT tolerates whitespace differences, indentation mismatches
- **Multi-block edits** - multiple EDIT blocks in one response for coordinated changes
- **Auto-continue** - if response is truncated mid-code, automatically continues
- **Error loop** - if RUN or EDIT fails, feeds errors back to the LLM to fix
- **Syntax verification** - auto-checks Python, JS, HTML, Ruby, Go, Rust, Bash after edits
- **Project-aware** - scans file tree so the model knows what files exist
- **Sticky intent** - follow-up messages stay in code mode ("now also add X")

All responses stream token-by-token as they're generated.

---

## MCP Support

Connect the agent to external tools via [Model Context Protocol](https://modelcontextprotocol.io/). Any stdio-based MCP server works - browser automation, databases, APIs, etc. Uses the same config format as Claude Desktop.

### Quick Start with Open WebMCP (Browser Automation)

<img width="1898" height="1454" alt="image" src="https://github.com/user-attachments/assets/abfac894-a335-463b-96e8-bdd09ca200bb" />


[Open WebMCP](https://github.com/daniel-farina/open-web-mcp) is an open-source MCP bridge that lets the agent control any website through Chrome. Navigate pages, read content, fill forms, click buttons, run JavaScript - 23 tools total.

**Architecture:**

```
mac code (agent.py)
  | stdio JSON-RPC
Bridge Server (node bridge.js, port 3852)
  | WebSocket
Chrome Extension (content.js in every tab)
  | DOM access
Any Website
```

**Install:**

```bash
# Clone Open WebMCP
git clone https://github.com/daniel-farina/open-web-mcp.git
cd open-web-mcp/server && npm install

# Load Chrome extension
# 1. Go to chrome://extensions
# 2. Enable Developer mode
# 3. Click "Load unpacked" → select the extension/ directory
# Badge shows "OFF" until bridge connects, then turns green
```

**Configure mac code to use it:**

Create `~/.mac-code/mcp.json`:

```json
{
  "mcpServers": {
    "webmcp": {
      "command": "node",
      "args": ["/absolute/path/to/open-web-mcp/server/bridge.js"]
    }
  }
}
```

**Start the agent:**

```bash
python3 agent.py
```

```
  connecting to MCP servers...
  ✓ webmcp  23 tools
  23 MCP tools available
```

### How It Works

MCP tools are injected into the coding agent's system prompt. The LLM sees all available tools and decides when to use them. It chains calls automatically - navigate, read, click, read again - until the task is done:

```
"open gmail and list my emails"
  → MCP: navigate {"url": "https://gmail.com"}
  → MCP: read_page_text {}
  → LLM: "Here are your 11 recent emails..."

"click the first email and read it"
  → MCP: click_element {"text": "AliExpress"}
  → MCP: read_page_text {}
  → LLM: summarizes the email content

"fill out the contact form on this page"
  → MCP: get_form_fields {}
  → MCP: fill_input {"selector": "#name", "value": "..."}
  → MCP: click_element {"text": "Submit"}
```

### Open WebMCP Tools (23)

| Category | Tools |
|---|---|
| **Navigation** | `navigate`, `get_active_tab`, `list_tabs`, `open_tab`, `close_tab`, `switch_tab` |
| **Reading** | `read_page_text`, `read_page_html`, `query_selector`, `find_by_text`, `get_page_info`, `get_links`, `get_table_data`, `get_form_fields` |
| **Interaction** | `click_element`, `fill_input`, `type_text`, `select_option`, `hover_element`, `scroll_page` |
| **Advanced** | `execute_javascript`, `wait_for_element`, `screenshot` |

Features:
- **Tab management** - controlled tabs group in a red "WebMCP" tab group with visual badges
- **Smart tab reuse** - agent tracks last-used tab, won't hijack your active window
- **Multi-instance** - multiple MCP clients can share one extension via proxy mode
- **Custom port** - configurable via `--port` flag or extension popup (default 3852)

### Adding More MCP Servers

Any stdio-based MCP server works. Add it to `~/.mac-code/mcp.json`:

```json
{
  "mcpServers": {
    "webmcp": {
      "command": "node",
      "args": ["/path/to/open-web-mcp/server/bridge.js"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

The agent discovers tools from all servers at startup and makes them available via `MCP:` markers in the coding agent.

---

## MLX Backend - Persistent Context

The MLX backend adds features llama.cpp can't do:

```bash
# Save context after analyzing a codebase
curl -X POST localhost:8000/v1/context/save \
    -d '{"name":"my-project","prompt":"your codebase here"}'

# Next day — resume instantly (0.0003s vs minutes reprocessing)
curl -X POST localhost:8000/v1/context/load \
    -d '{"name":"my-project"}'

# Different Mac — download from R2 (1.5s)
curl -X POST localhost:8000/v1/context/download \
    -d '{"name":"my-project"}'
```

TurboQuant compresses context storage 4x (26.6 MB → 6.7 MB) with 0.993 cosine similarity.

See `mlx/PROJECT.md` for the full research roadmap.

---

## Commands

Type `/` to see all commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode (default) |
| `/raw` | Direct streaming, no tools |
| `/model 9b` | Switch to 9B (64K ctx) |
| `/model 35b` | Switch to 35B MoE (llama.cpp only) |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/stats` | Session statistics |
| `/cost` | Cost savings vs cloud |
| `/good` / `/bad` | Grade response (self-improvement logging) |
| `/improve` | View grading stats |
| `/clear` | Reset conversation |
| `/quit` | Exit |

---

## Benchmarks

### Agent Tasks — llama.cpp vs MLX (same prompts, same Mac mini M4)

| Task | llama.cpp | MLX | Winner |
|---|---|---|---|
| Shell command | 7.9s | **7.6s** | MLX |
| Math | 12.4s | **9.8s** | **MLX (21%)** |
| Code gen | 12.3s | **9.7s** | **MLX (21%)** |
| Reasoning | 12.3s | **10.0s** | **MLX (19%)** |
| Web search | **45.7s** | 48.3s | llama.cpp |

### Context Persistence (MLX only)

| Operation | Time |
|---|---|
| Reprocess 141 tokens | 1.01s |
| **SSD load** | **0.0003s (6,677x faster)** |
| R2 download + load | 1.5s |
| TurboQuant compress | 26.6 → 6.7 MB (4x) |

### SSD Paging — Apple Silicon vs NVIDIA

| Hardware | Speed | Cost |
|---|---|---|
| **Mac mini M4** | **30 tok/s** | **$0/month** |
| NVIDIA + NVMe | 1.6 tok/s | $0.44/hr |
| NVIDIA in-VRAM | 42.5 tok/s | $0.34/hr |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  agent.py — LLM-as-Router                        │
│  search / shell / code / chat                    │
│  + coding agent (FILE/EDIT/RUN/READ)             │
│  + MCP client (browser, APIs, databases)         │
├──────────┬───────────────────────────────────────┤
│ llama.cpp│  MLX backend                          │
│ backend  │  + KV cache save/load                 │
│          │  + TurboQuant 4-bit compression       │
│          │  + Cloudflare R2 sync                 │
│          │  + Paged inference (GPU→SSD)          │
├──────────┴───────────────────────────────────────┤
│  MCP servers (stdio JSON-RPC)                    │
│  browser · github · databases · any MCP server   │
├──────────────────────────────────────────────────┤
│  Apple Silicon — Unified Memory + SSD paging     │
└──────────────────────────────────────────────────┘
```

---

## Configuration

| File | What |
|---|---|
| `~/.mac-code/mcp.json` | MCP server connections (same format as Claude Desktop) |
| `~/.mac-code/history` | Command history (persists across sessions) |
| `~/.mac-code/logs/` | Interaction logs for self-improvement |
| `~/models/` | GGUF model files |

Environment variables:

| Variable | Default | What |
|---|---|---|
| `LLAMA_URL` | `http://localhost:8000` | LLM server URL |
| `MAC_CODE_MAX_ITER` | `100` | Max coding agent iterations per turn |

---

## Files

| File | What |
|---|---|
| `agent.py` | CLI agent - coding, MCP, search, shell, chat |
| `chat.py` | Streaming chat |
| `dashboard.py` | Server monitor |
| `web/` | Retro Mac web UI |
| `mlx/mlx_engine.py` | MLX inference server with context API |
| `mlx/kv_cache.py` | KV cache save/load/compress |
| `mlx/r2_store.py` | Cloudflare R2 integration |
| `mlx/turboquant.py` | 4-bit KV compression |
| `mlx/paged_inference.py` | Process docs beyond context limit |
| `mlx/PROJECT.md` | MLX research roadmap |

---

## Scaling

| Mac | RAM | What you can run |
|---|---|---|
| Any Mac (8GB) | 8 GB | 9B, 4K context |
| **Mac mini M4** | **16 GB** | **9B (64K) + 35B MoE (12K, SSD paging)** |
| Mac mini M4 Pro | 48 GB | 35B at Q4 + speculative decoding |
| Mac Studio Ultra | 192 GB | 397B frontier model |

Same `agent.py` at every level. Just swap the model.

---

## Research

This project builds on:
- **[Apple "LLM in a Flash"](https://machinelearning.apple.com/research/efficient-large-language)** — SSD paging via unified memory
- **[Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** — KV cache compression
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's native ML framework

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** - the models
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** - inference engine
- **[Unsloth](https://huggingface.co/unsloth)** - GGUF quantizations
- **[Open WebMCP](https://github.com/daniel-farina/open-web-mcp)** - browser automation via MCP
- **[Cloudflare R2](https://developers.cloudflare.com/r2/)** - free object storage
- **[Rich](https://github.com/Textualize/rich)** - terminal UI

## License

MIT
