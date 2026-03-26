#!/usr/bin/env python3
"""
mac code — claude code for your Mac
"""

import json, sys, os, time, subprocess, re, threading, queue
import urllib.request, random, readline
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding
from rich.columns import Columns

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")
MAX_ITERATIONS = int(os.environ.get("MAC_CODE_MAX_ITER", "100"))

# ── Self-improvement: failure logging ─────────────
LOGS_DIR = Path.home() / ".mac-code" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Session persistence ──────────────────────────
SESSIONS_DIR = Path.home() / ".mac-code" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def save_session(session_id, messages, metadata=None):
    """Save conversation to disk."""
    data = {
        "session_id": session_id,
        "messages": messages,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
        "work_dir": os.getcwd(),
    }
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def load_session(session_id):
    """Load a conversation from disk."""
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def list_sessions(limit=10):
    """List recent sessions sorted by date."""
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            first_msg = ""
            for m in data.get("messages", []):
                if m.get("role") == "user":
                    first_msg = m["content"][:60]
                    break
            sessions.append({
                "id": data["session_id"],
                "date": data.get("saved_at", "")[:16],
                "turns": len([m for m in data.get("messages", []) if m.get("role") == "user"]),
                "preview": first_msg,
                "work_dir": data.get("work_dir", ""),
            })
        except Exception:
            pass
        if len(sessions) >= limit:
            break
    return sessions

# ── Background job manager ────────────────────────
class BackgroundJobs:
    """Track background processes started by the agent (dev servers, builds, etc)."""

    def __init__(self):
        self.jobs = {}  # id -> {process, cmd, started, port, status}
        self._next_id = 0

    def start(self, cmd, cwd="."):
        """Start a background process. Returns job id."""
        self._next_id += 1
        jid = self._next_id
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            # Try to detect port from command
            port = None
            import re as _re
            port_match = _re.search(r'(?:--port|[-:])\s*(\d{4,5})', cmd)
            if port_match:
                port = int(port_match.group(1))

            self.jobs[jid] = {
                "process": proc,
                "cmd": cmd,
                "started": time.time(),
                "port": port,
                "pid": proc.pid,
                "cwd": cwd,
            }
            return jid
        except Exception as e:
            return None

    def stop(self, jid):
        """Stop a background job by id."""
        if jid in self.jobs:
            job = self.jobs[jid]
            try:
                job["process"].terminate()
                job["process"].wait(timeout=5)
            except Exception:
                job["process"].kill()
            del self.jobs[jid]
            return True
        return False

    def stop_all(self):
        for jid in list(self.jobs.keys()):
            self.stop(jid)

    def list_jobs(self):
        """Return list of running jobs."""
        alive = {}
        for jid, job in list(self.jobs.items()):
            if job["process"].poll() is None:
                alive[jid] = job
            else:
                del self.jobs[jid]
        self.jobs = alive
        return alive

    def get_output(self, jid, lines=20):
        """Read recent output from a job (non-blocking)."""
        if jid not in self.jobs:
            return None
        proc = self.jobs[jid]["process"]
        output = []
        import select
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            output.append(line.rstrip())
            if len(output) >= lines:
                break
        return output

    def render_status(self):
        """Render a one-line status for the prompt."""
        alive = self.list_jobs()
        if not alive:
            return ""
        parts = []
        for jid, job in alive.items():
            elapsed = int(time.time() - job["started"])
            port_str = f":{job['port']}" if job.get('port') else ""
            parts.append(f"[{jid}] {job['cmd'][:30]}{port_str} ({elapsed}s)")
        return " | ".join(parts)


bg_jobs = BackgroundJobs()

# ── readline: history + tab completion ────────────
HISTORY_FILE = Path.home() / ".mac-code" / "history"

def _setup_readline():
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    readline.set_history_length(1000)
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))
    except OSError:
        pass

def save_readline_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass

def log_interaction(query, intent, response, speed, grade=None, error=None):
    """Log every interaction for self-improvement training data."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "intent": intent,
        "response": response[:500] if response else None,
        "speed": speed,
        "grade": grade,  # "good", "bad", or None (ungraded)
        "error": error,
        "model": get_current_model() if 'get_current_model' in dir() else "unknown",
    }
    log_file = LOGS_DIR / f"interactions-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_failure_stats():
    """Show stats from logged interactions."""
    total = 0
    graded = {"good": 0, "bad": 0}
    intents = {"search": 0, "shell": 0, "chat": 0}
    errors = 0

    for log_file in LOGS_DIR.glob("interactions-*.jsonl"):
        for line in open(log_file):
            try:
                entry = json.loads(line)
                total += 1
                if entry.get("grade"):
                    graded[entry["grade"]] = graded.get(entry["grade"], 0) + 1
                if entry.get("intent"):
                    intents[entry["intent"]] = intents.get(entry["intent"], 0) + 1
                if entry.get("error"):
                    errors += 1
            except:
                pass

    return {"total": total, "graded": graded, "intents": intents, "errors": errors}
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── model configs ─────────────────────────────────
MODELS = {
    "9b": {
        "path": os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf"),
        "ctx": 65536,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -t 4",
        "name": "Qwen3.5-9B",
        "detail": "8.95B dense · Q4_K_M · 64K ctx",
        "good_for": "tool calling, long conversations, agent tasks",
    },
    "35b": {
        "path": os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"),
        "ctx": 8192,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -np 1 -t 4",
        "name": "Qwen3.5-35B-A3B",
        "detail": "MoE 34.7B · 3B active · IQ2_M · 8K ctx",
        "good_for": "reasoning, math, knowledge, fast answers",
    },
}

# ── smart routing ─────────────────────────────────
TOOL_KEYWORDS = [
    "search", "find", "look up", "google", "what time", "when do",
    "when is", "when does", "when are", "who do", "who is playing",
    "who plays", "who won", "what happened", "what is the score",
    "weather", "news", "latest", "schedule", "score", "tonight",
    "today", "tomorrow", "yesterday", "this week", "next game",
    "play next", "playing next", "results", "standings",
    "price", "stock", "market", "crypto", "bitcoin",
    "fetch", "download", "read file", "write file",
    "create file", "run", "execute", "list files", "show me",
    "open", "browse", "url", "http", "website",
    "how much", "where is", "directions", "recipe",
    "explore", "repo", "repository", "github", "tell me more",
    "more about", "what else", "continue", "go deeper",
]

def classify_intent(message):
    """Ask LLM to classify: 'search', 'shell', 'code', or 'chat'. One fast call (~1s)."""
    mcp_hint = ""
    if mcp_clients:
        mcp_hint = " Also use code for browser automation (get tab info, read page text, click elements, navigate browser, fill forms, take screenshot, interact with web pages)."
    try:
        result, _ = llm_call([
            {"role": "system", "content": f"""Classify the user's request into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- search: needs web search (news, scores, weather, prices, current events, looking up info online)
- shell: needs simple filesystem info or commands (find files, list directories, check disk space, quick one-line commands)
- code: needs to write, create, edit, fix, run, or deploy code/programs/files (build a game, create an app, write a script, make a website, edit code, fix a bug, add a feature, refactor, run a dev server, start an app, npm install, npm run dev, run this command, start the server, deploy, build).{mcp_hint}
- chat: general conversation, reasoning, math, explanations (no file creation or commands needed)

Reply with ONLY one word: search, shell, code, or chat"""},
            {"role": "user", "content": message},
        ], max_tokens=5, temperature=0.0)
        return result.strip().lower().split()[0]
    except Exception:
        return "chat"

def generate_shell_command(query, work_dir="."):
    """Ask LLM to generate the right shell command for a file/system task."""
    home = os.path.expanduser("~")
    result, _ = llm_call([
        {"role": "system", "content": f"""You are a macOS shell command generator. The user's home directory is {home}. Current working directory is {work_dir}.

Generate a single shell command that accomplishes the user's request. Output ONLY the command, nothing else. No explanation, no markdown, no backticks.

Examples:
- "find videos on my desktop" → find {home}/Desktop -type f \\( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \\)
- "what files are on my desktop" → ls -la {home}/Desktop
- "how much disk space do I have" → df -h /
- "show me python files in this project" → find . -name "*.py" -type f
- "read the readme" → cat README.md
- "what's running on port 8000" → lsof -i :8000
- "count lines of code" → find . -name "*.py" -exec wc -l {{}} +"""},
        {"role": "user", "content": query},
    ], max_tokens=100, temperature=0.0)
    return result.strip().strip('`').strip()

def run_smart_tool(query, work_dir="."):
    """Execute a shell command generated by the LLM, feed results back."""
    import subprocess as sp
    from datetime import datetime

    # Step 1: LLM generates the command (~1s)
    cmd = generate_shell_command(query, work_dir)

    # Step 2: Execute it
    try:
        result = sp.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=30, cwd=work_dir)
        output = result.stdout[:8000]
        if result.stderr:
            output += f"\n{result.stderr[:2000]}"
    except sp.TimeoutExpired:
        output = "Command timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {e}"

    # Step 3: LLM summarizes results (~2-3s)
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. You ran a shell command and got results. Present the results clearly to the user. If it's a file listing, format it nicely. If it's code, use formatting. Be helpful and concise."},
        {"role": "user", "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0), cmd

def run_file_tool(query, work_dir="."):
    """Execute file/exec operations directly in Python, feed results to LLM."""
    import subprocess as sp
    from datetime import datetime

    lower = query.lower()
    tool_output = ""
    tool_name = ""

    try:
        # List directory
        if any(kw in lower for kw in ["list files", "list dir", "ls ", "what's in"]):
            # Extract path or use work_dir
            path = work_dir
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isdir(expanded):
                    path = expanded
                    break
            entries = os.listdir(path)
            entries.sort()
            tool_name = f"list_dir({path})"
            tool_output = "\n".join(entries[:50])
            if len(entries) > 50:
                tool_output += f"\n... and {len(entries)-50} more"

        # Read file
        elif any(kw in lower for kw in ["read file", "show me", "look at", "cat ", "what's in"]):
            # Find file path in the query
            path = None
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isfile(expanded):
                    path = expanded
                    break
                # Try with work_dir
                joined = os.path.join(work_dir, token)
                if os.path.isfile(joined):
                    path = joined
                    break
            if path:
                with open(path, "r", errors="ignore") as f:
                    content = f.read(10000)
                tool_name = f"read_file({path})"
                tool_output = content
            else:
                tool_output = f"Could not find file in query: {query}"
                tool_name = "read_file(not found)"

        # Write file
        elif any(kw in lower for kw in ["write file", "write a file", "create file", "create a file",
                                          "create a new", "save file", "save to", "save this"]):
            # LLM decides what to write
            content, _ = llm_call([
                {"role": "system", "content": "The user wants to create/write a file. Generate ONLY the file content. No explanations."},
                {"role": "user", "content": query},
            ], max_tokens=2000)

            # Extract filename from query
            filename = None
            for token in query.split():
                if "." in token and not token.startswith("http"):
                    filename = token
                    break
            if not filename:
                filename = "output.txt"

            filepath = os.path.join(work_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            tool_name = f"write_file({filepath})"
            tool_output = f"Written {len(content)} bytes to {filepath}"

        # Execute command
        elif any(kw in lower for kw in ["execute", "run "]):
            # Extract command
            cmd = query
            for prefix in ["execute ", "run "]:
                if lower.startswith(prefix):
                    cmd = query[len(prefix):]
                    break

            result = sp.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=30, cwd=work_dir)
            tool_name = f"exec({cmd.strip()[:40]})"
            tool_output = result.stdout[:5000]
            if result.stderr:
                tool_output += f"\nSTDERR: {result.stderr[:1000]}"

        else:
            return None

    except Exception as e:
        tool_output = f"Error: {e}"
        tool_name = "error"

    # Feed tool output to LLM for final answer
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. You executed a tool and got results. Summarize the results clearly for the user. If it's code, format it nicely."},
        {"role": "user", "content": f"Tool: {tool_name}\nResult:\n{tool_output}\n\nOriginal question: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0), tool_name

def llm_call(messages, max_tokens=300, temperature=0.1):
    """Single LLM call, returns content + timings."""
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return d["choices"][0]["message"]["content"], d.get("timings", {})

def quick_search(query):
    """LLM rewrites query → DuckDuckGo search → LLM answers. ~5-8s total."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")

    # Step 1: LLM rewrites query into optimal search terms (~1s)
    try:
        search_query, _ = llm_call([
            {"role": "system", "content": f"Today is {today}. Rewrite the user's question into an optimal web search query that will find current, specific data (not articles about announcements). Include 'today' or 'tonight' and the full date for time-sensitive queries. Add words like 'scores', 'results', 'live', or 'now' when looking for current data. Output ONLY the search query string, nothing else."},
            {"role": "user", "content": query},
        ], max_tokens=30, temperature=0.0)
        search_query = search_query.strip().strip('"\'')
    except Exception:
        search_query = query

    # Step 2: DuckDuckGo search — text (15 results) + news (5 results)
    ddg = DDGS()
    all_results = []

    try:
        text_results = ddg.text(search_query, max_results=15)
        all_results.extend(text_results)
    except Exception:
        pass

    try:
        news_results = ddg.news(search_query, max_results=5)
        all_results.extend(news_results)
    except Exception:
        pass

    if not all_results:
        return None

    # Combine all snippets
    snippets = "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in all_results])

    # Check if snippets actually contain useful data or just meta descriptions
    # If total snippet text is mostly generic, fetch the best page
    import re as _re
    page_content = ""
    snippet_words = len(snippets.split())

    # Heuristic: check if snippets have actual specific data
    # Numbers with context (times, scores, prices) count. Generic "live scores available" doesn't.
    specific_patterns = _re.findall(r'\d{1,2}:\d{2}\s*(?:p\.m\.|a\.m\.|ET|PT)|\$[\d,.]+|\d+-\d+(?:\s*(?:win|loss|final))', snippets.lower())
    has_specifics = len(specific_patterns) >= 2  # need at least 2 specific data points

    if not has_specifics and all_results:
        # Snippets are weak — use Jina Reader to fetch the best page
        # Jina reads JS-rendered pages (ESPN, etc.) that urllib can't
        for r in all_results[:3]:
            url = r.get("href") or r.get("link", "")
            if not url:
                continue
            try:
                jina_url = f"https://r.jina.ai/{url}"
                req = urllib.request.Request(jina_url, headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/plain",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    text = resp.read(6000).decode("utf-8", errors="ignore")
                    if len(text) > 200:
                        page_content = text[:4000]
                        break
            except Exception:
                continue

    context = snippets
    if page_content:
        context += f"\n\nDetailed content from top result:\n{page_content}"

    # Step 3: LLM answers using results (~2-3s)
    content, timings = llm_call([
        {"role": "system", "content": f"Today is {today}. Answer the user's question using the search results below. Be specific, direct, and detailed. Extract dates, times, scores, names, numbers, prices, and facts. Present them clearly."},
        {"role": "user", "content": f"Search results:\n\n{context}\n\nQuestion: {query}"},
    ], max_tokens=1000)

    return content, timings.get("predicted_per_second", 0)

def get_current_model():
    """Check which model the running server has loaded."""
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "35b"
        elif "9B" in alias:
            return "9b"
    except Exception:
        pass
    return None

def swap_model(target_key):
    """Stop current server and start a new one with the target model."""
    cfg = MODELS[target_key]
    if not os.path.exists(cfg["path"]):
        return False, f"Model not found: {cfg['path']}"

    # Kill current server
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

    # Start new server
    cmd_list = [
        "llama-server",
        "--model", cfg["path"],
        "--port", "8000",
        "--host", "127.0.0.1",
        "--ctx-size", str(cfg["ctx"]),
    ] + cfg["flags"].split()
    subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for ready
    for i in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{SERVER}/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                return True, f"Switched to {cfg['name']} ({cfg['ctx']} ctx)"
        except Exception:
            pass

    return False, "Server failed to start"

def ensure_server_running():
    """Check if llama-server is running; if not, auto-start with 9B model."""
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=2) as r:
            d = json.loads(r.read())
        if d.get("status") == "ok":
            return True
    except Exception:
        pass

    cfg = MODELS.get("9b")
    if not cfg or not os.path.exists(cfg["path"]):
        # Try 35b
        cfg = MODELS.get("35b")
        if not cfg or not os.path.exists(cfg["path"]):
            return False

    console.print(f"  [dim]starting llama-server ({cfg['name']})...[/]")
    cmd_list = [
        "llama-server",
        "--model", cfg["path"],
        "--port", "8000", "--host", "127.0.0.1",
        "--ctx-size", str(cfg["ctx"]),
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    ] + cfg["flags"].split()

    try:
        subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        console.print("  [bold red]llama-server not found. Install: brew install llama.cpp[/]")
        return False

    for _ in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{SERVER}/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                console.print("  [dim]server ready.[/]")
                return True
        except Exception:
            pass

    console.print("  [bold red]server failed to start within 60s[/]")
    return False

# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')
def strip_ansi(text):
    return ANSI_RE.sub('', text)

# ── live working display ──────────────────────────
DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

class WorkingDisplay:
    def __init__(self):
        self.events = []
        self.phase = "thinking"
        self.frame = 0
        self.start_time = time.time()
        self.logs = []

    def add_log(self, line):
        clean = strip_ansi(line).strip()
        if not clean:
            return

        lower = clean.lower()
        new_phase = None
        detail = ""

        if "processing message" in lower:
            new_phase = "reading your message"
        elif "llm_request" in lower:
            new_phase = "thinking"
        elif "tool_call" in lower or "web_search" in lower:
            if "web_search" in lower or "duckduckgo" in lower:
                new_phase = "searching the web"
            elif "web_fetch" in lower or "fetch" in lower:
                new_phase = "fetching page"
            elif "exec" in lower:
                new_phase = "running command"
            elif "read_file" in lower:
                new_phase = "reading file"
            elif "write_file" in lower:
                new_phase = "writing file"
            else:
                new_phase = "using tools"
        elif "context_compress" in lower:
            new_phase = "compressing context"
        elif "turn_end" in lower:
            new_phase = "finishing up"

        if new_phase:
            self.phase = new_phase
            self.events.append((time.time() - self.start_time, new_phase, detail))

        # Keep last few interesting log lines
        if any(k in lower for k in ["llm_request", "tool_call", "tool_result", "turn_end", "web_search", "fetch", "exec"]):
            short = clean
            if ">" in short:
                short = short.split(">", 1)[-1].strip()
            if len(short) > 70:
                short = short[:67] + "..."
            self.logs.append(short)
            if len(self.logs) > 3:
                self.logs.pop(0)

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start_time
        spinner = DOTS[self.frame % len(DOTS)]

        t = Text()
        t.append(f"  {spinner} ", style="bold bright_cyan")
        t.append(self.phase, style="bold bright_cyan")
        t.append(f"  {elapsed:.0f}s", style="dim")
        t.append("\n")

        for log in self.logs[-3:]:
            t.append(f"    {log}\n", style="dim italic")

        return t

# ── detect model ───────────────────────────────────
def detect_model():
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "Qwen3.5-35B-A3B", "MoE 34.7B · 3B active · IQ2_M"
        elif "9B" in alias:
            return "Qwen3.5-9B", "8.95B dense · Q4_K_M"
        return alias.replace(".gguf", "").split("/")[-1], "local"
    except Exception:
        return "offline", ""

# ── streaming chat (raw mode) ─────────────────────
def stream_llm(messages):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        while True:
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return full, tokens, time.time() - start
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c
                except Exception:
                    pass

    return full, tokens, time.time() - start

def stream_chat(messages, max_tokens=1000, temperature=0.7):
    """Streaming LLM call for arbitrary messages - yields content chunks. Supports both SSE and non-streaming backends (MLX)."""
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        # Check content type - if not SSE, read as single JSON (MLX engine)
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            raw_body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw_body)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    yield content
            except Exception:
                pass
            return

        buf = ""
        while True:
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        yield c
                except Exception:
                    pass

def prepare_shell(query, work_dir="."):
    """Generate and execute a shell command. Returns (cmd, summary_messages)."""
    import subprocess as sp
    cmd = generate_shell_command(query, work_dir)

    try:
        result = sp.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=30, cwd=work_dir)
        output = result.stdout[:8000]
        if result.stderr:
            output += f"\n{result.stderr[:2000]}"
    except sp.TimeoutExpired:
        output = "Command timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {e}"

    today = datetime.now().strftime("%A, %B %d, %Y")
    msgs = [
        {"role": "system", "content": f"Today is {today}. You ran a shell command and got results. Present the results clearly to the user. If it's a file listing, format it nicely. If it's code, use formatting. Be helpful and concise."},
        {"role": "user", "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {query}"},
    ]
    return cmd, msgs

def prepare_search(query):
    """Rewrite query and search DuckDuckGo. Returns summary_messages or None."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    today = datetime.now().strftime("%A, %B %d, %Y")

    try:
        search_query, _ = llm_call([
            {"role": "system", "content": f"Today is {today}. Rewrite the user's question into an optimal web search query that will find current, specific data (not articles about announcements). Include 'today' or 'tonight' and the full date for time-sensitive queries. Add words like 'scores', 'results', 'live', or 'now' when looking for current data. Output ONLY the search query string, nothing else."},
            {"role": "user", "content": query},
        ], max_tokens=30, temperature=0.0)
        search_query = search_query.strip().strip('"\'')
    except Exception:
        search_query = query

    ddg = DDGS()
    all_results = []
    try:
        all_results.extend(ddg.text(search_query, max_results=15))
    except Exception:
        pass
    try:
        all_results.extend(ddg.news(search_query, max_results=5))
    except Exception:
        pass

    if not all_results:
        return None

    snippets = "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in all_results])

    import re as _re
    page_content = ""
    specific_patterns = _re.findall(r'\d{1,2}:\d{2}\s*(?:p\.m\.|a\.m\.|ET|PT)|\$[\d,.]+|\d+-\d+(?:\s*(?:win|loss|final))', snippets.lower())
    has_specifics = len(specific_patterns) >= 2

    if not has_specifics and all_results:
        for r in all_results[:3]:
            url = r.get("href") or r.get("link", "")
            if not url:
                continue
            try:
                jina_url = f"https://r.jina.ai/{url}"
                req = urllib.request.Request(jina_url, headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/plain",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    text = resp.read(6000).decode("utf-8", errors="ignore")
                    if len(text) > 200:
                        page_content = text[:4000]
                        break
            except Exception:
                continue

    context = snippets
    if page_content:
        context += f"\n\nDetailed content from top result:\n{page_content}"

    msgs = [
        {"role": "system", "content": f"Today is {today}. Answer the user's question using the search results below. Be specific, direct, and detailed. Extract dates, times, scores, names, numbers, prices, and facts. Present them clearly."},
        {"role": "user", "content": f"Search results:\n\n{context}\n\nQuestion: {query}"},
    ]
    return msgs

# ── MCP client ────────────────────────────────────

MCP_CONFIG_PATH = Path.home() / ".mac-code" / "mcp.json"

class MCPClient:
    """Minimal MCP client - connects to stdio MCP servers via newline-delimited JSON-RPC."""

    def __init__(self, name, command, args, env=None):
        self.name = name
        self.tools = []
        self._id = 0
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        self.process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._buf = b""
        self._initialize()

    def _next_id(self):
        self._id += 1
        return self._id

    def _send(self, method, params=None, timeout=15):
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode())
        self.process.stdin.flush()
        return self._read_response(msg["id"], timeout=timeout)

    def _send_notification(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode())
        self.process.stdin.flush()

    def _read_response(self, expected_id, timeout=15):
        import select
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if we already have a complete line in buffer
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Skip notifications (no id field)
                if "id" not in data:
                    continue
                if data.get("id") == expected_id:
                    return data
            # Read more data
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 0.5))
            if not ready:
                continue
            chunk = self.process.stdout.read1(4096) if hasattr(self.process.stdout, 'read1') else self.process.stdout.read(1)
            if not chunk:
                return None
            self._buf += chunk
        return None

    def _initialize(self):
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mac-code", "version": "1.0"}
        })
        self._send_notification("notifications/initialized")
        return resp

    def list_tools(self):
        resp = self._send("tools/list", timeout=15)
        if resp and "result" in resp:
            self.tools = resp["result"].get("tools", [])
        return self.tools

    def call_tool(self, name, arguments=None):
        resp = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        if resp and "result" in resp:
            result = resp["result"]
            # Extract text from MCP content array
            if "content" in result:
                texts = []
                for item in result["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return "\n".join(texts)
            return json.dumps(result, indent=2)
        if resp and "error" in resp:
            return f"Error: {resp['error'].get('message', str(resp['error']))}"
        return "Error: No response from MCP server"

    def close(self):
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            self.process.kill()


# Global MCP clients
mcp_clients = {}

def load_mcp_servers():
    """Load MCP server configs from ~/.mac-code/mcp.json and connect."""
    global mcp_clients
    if not MCP_CONFIG_PATH.exists():
        return
    try:
        with open(MCP_CONFIG_PATH) as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        for name, cfg in servers.items():
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            env = cfg.get("env")
            if not command:
                continue
            try:
                client = MCPClient(name, command, args, env)
                tools = client.list_tools()
                mcp_clients[name] = client
                console.print(f"  [bold bright_green]\u2713[/] [bold]{name}[/]  {len(tools)} tools")
            except Exception as e:
                console.print(f"  [bold red]\u2717[/] [bold]{name}[/]  {e}")
    except Exception as e:
        console.print(f"  [dim]mcp config error: {e}[/]")

def get_mcp_tool_descriptions():
    """Build tool description string from all connected MCP servers."""
    if not mcp_clients:
        return ""
    lines = ["""
MCP (browser & external tools):
You have access to MCP tools that let you control the user's browser and interact with web pages.
These tools run in the user's actual Chrome browser via an extension.

To call an MCP tool, use:
MCP: tool_name
ARGS: {"param": "value"}

You will see the result, then decide your next action. You can chain multiple MCP calls.

HOW TO USE MCP TOOLS EFFECTIVELY:
1. Start by navigating: use 'navigate' to open a URL, or 'get_active_tab' to see what's already open
2. Read the page: use 'read_page_text' to see what's on the page, or 'query_selector'/'find_by_text' to find specific elements
3. Interact: use 'click_element' to click buttons/links, 'fill_input'/'type_text' for forms
4. Chain calls: after each action, read the page again to see what changed, then decide next step
5. Be patient: web pages load dynamically - use 'wait_for_element' if needed

Example workflow - reading emails:
  MCP: navigate -> MCP: read_page_text -> MCP: click_element (on an email) -> MCP: read_page_text (read content)

Available MCP tools:"""]
    for server_name, client in mcp_clients.items():
        for tool in client.tools:
            name = tool["name"]
            desc = tool.get("description", "")
            props = tool.get("inputSchema", {}).get("properties", {})
            required = tool.get("inputSchema", {}).get("required", [])
            param_parts = []
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", pinfo.get("type", ""))
                param_parts.append(f"{pname}: {pdesc}{req}")
            params = ", ".join(param_parts) if param_parts else "none"
            lines.append(f"- {name}: {desc} -- params: {params}")
    return "\n".join(lines)

def call_mcp_tool(tool_name, arguments):
    """Find and call an MCP tool across all connected servers."""
    for client in mcp_clients.values():
        for tool in client.tools:
            if tool["name"] == tool_name:
                return client.call_tool(tool_name, arguments)
    return f"Error: MCP tool '{tool_name}' not found"

def close_mcp_servers():
    for client in mcp_clients.values():
        client.close()
    mcp_clients.clear()


# ── unified diff applier ──────────────────────────

def apply_udiff(file_path, patch_text, work_dir="."):
    """Apply a unified diff patch to a file. Returns (success, old_content, new_content, error)."""
    if not os.path.isabs(file_path):
        file_path = os.path.join(work_dir, file_path)
    file_path = os.path.expanduser(file_path)

    try:
        with open(file_path, "r") as f:
            original_lines = f.readlines()
    except FileNotFoundError:
        original_lines = []

    old_content = "".join(original_lines)
    result_lines = list(original_lines)
    offset = 0  # track line shifts from previous hunks

    # Parse hunks from the diff
    hunk_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    lines = patch_text.split("\n")
    i = 0

    # Skip header lines (--- and +++)
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1

    while i < len(lines):
        m = hunk_re.match(lines[i])
        if not m:
            i += 1
            continue

        old_start = int(m.group(1)) - 1  # 0-indexed
        i += 1

        # Collect hunk lines
        remove_lines = []
        add_lines = []
        context_before = 0
        in_prefix = True

        while i < len(lines):
            line = lines[i]
            if line.startswith("@@") or line.startswith("diff ") or line.startswith("---") or line.startswith("+++"):
                break
            if line.startswith("-"):
                remove_lines.append(line[1:])
                in_prefix = False
            elif line.startswith("+"):
                add_lines.append(line[1:])
                in_prefix = False
            elif line.startswith(" ") or line == "":
                if in_prefix:
                    context_before += 1
                if line.startswith(" "):
                    # Context line - not removed or added
                    pass
            i += 1

        # Apply the hunk
        apply_at = old_start + context_before + offset
        n_remove = len(remove_lines)

        # Verify the lines match (fuzzy)
        if n_remove > 0 and apply_at < len(result_lines):
            expected = [l.rstrip("\n") for l in remove_lines]
            actual = [l.rstrip("\n") for l in result_lines[apply_at:apply_at + n_remove]]
            if expected != actual:
                # Try fuzzy match - strip whitespace
                exp_stripped = [l.strip() for l in expected]
                act_stripped = [l.strip() for l in actual]
                if exp_stripped != act_stripped:
                    return False, old_content, old_content, f"Hunk failed at line {apply_at + 1}: expected {expected[:2]} but found {actual[:2]}"

        # Replace
        new_lines_with_newline = [l + "\n" for l in add_lines]
        result_lines[apply_at:apply_at + n_remove] = new_lines_with_newline
        offset += len(add_lines) - n_remove

    new_content = "".join(result_lines)

    if new_content == old_content:
        return False, old_content, old_content, "Patch produced no changes"

    # Write the patched file
    dname = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dname, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(new_content)

    return True, old_content, new_content, None


# ── coding agent ──────────────────────────────────

CODE_SYSTEM = """You are a coding agent on macOS. You write, edit, and run code directly on the user's computer.

CRITICAL: To modify an existing file, ALWAYS use READ: then DIFF:. NEVER regenerate an entire file with FILE: just to change a few lines. FILE: is ONLY for brand-new files.

Available operations:

READ: path/to/file.ext
  Always read a file before editing it. You need the file contents to write accurate diffs.
  IMPORTANT: When you READ a file, STOP and wait for the contents. Do NOT guess changes in the same response.

DIFF: path/to/file.ext
```diff
--- a/path/to/file.ext
+++ b/path/to/file.ext
@@ -10,3 +10,4 @@
 context line (unchanged, starts with space)
-old line to remove
+new line to add
+another new line
```
  DEFAULT for all modifications. Uses standard unified diff format.
  Rules for DIFF:
  - Lines starting with space are CONTEXT (unchanged, must match the file exactly)
  - Lines starting with - are REMOVED
  - Lines starting with + are ADDED
  - Include 1-3 context lines before and after changes for accurate placement
  - The @@ header shows line numbers: @@ -oldstart,count +newstart,count @@
  - You can have multiple @@ hunks in one DIFF block for changes in different parts of the file

EDIT: path/to/file.ext
<<<SEARCH
exact text to find
===REPLACE
replacement text
>>>
  Fallback: simple search-and-replace. Use only if DIFF is too complex.

FILE: path/to/file.ext
```
complete content
```
  ONLY for creating brand-new files that don't exist yet.

RUN: command
  Execute a shell command.

Rules:
- To MODIFY existing code: READ first, then DIFF. Use unified diff format by default.
- Use multiple @@ hunks in one DIFF block for changes in different parts of the same file.
- Keep EDIT search text short (3-8 lines) and unique so it matches reliably.
- For NEW files: use FILE with complete working code. No placeholders or TODOs.
- For web apps: FILE to create, then RUN: open file.html
- For Vite/npm projects: Use scaffold commands instead of writing files manually:
  RUN: yes | npm create vite@latest myapp -- --template vanilla
  RUN: cd myapp && npm install
  Then EDIT the generated files to customize. IMPORTANT: always pipe yes or use --yes to avoid interactive prompts that hang.
- When the user says "turn this into X" or "make it do Y", modify the EXISTING project - do NOT create a new one.
- When the user says "fix it", READ the error output, diagnose the root cause, and try a DIFFERENT approach.
- After starting a dev server, tell the user the URL (http://localhost:PORT).
- For Python: FILE to create, then RUN: python3 file.py
- If too long for one response, end with CONTINUE.
- Keep explanations brief. Focus on code.
- For complex changes: make one focused EDIT at a time. You can do more EDITs in subsequent iterations.
- NEVER output bare code blocks without FILE: or EDIT: markers. All code must use markers.

CRITICAL - CSS LAYOUTS:
- When asked for a grid layout, ALWAYS add the CSS. Either inline style or a style block.
- 3-column grid: style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px"
- 2-column grid: style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px"
- Responsive: use repeat(auto-fill, minmax(280px, 1fr)) for auto-adapting columns
- Cards in a row: use display:grid or display:flex on the PARENT container, not on the cards themselves

CRITICAL - SAFE EDITS:
- When editing inside a <script> block, do NOT include </script> in your replacement unless it was in your search text.
- Keep EDITs small and focused. Insert new functions BETWEEN existing ones, don't replace entire blocks.
- To add a function: search for the line BEFORE where you want to insert, and replace it with that same line PLUS your new function.
- Example of SAFE insertion:
  SEARCH: let allCoins = [];
  REPLACE: let allCoins = [];
           function myNewFunction() { ... }
  This ADDS code after the existing line without removing anything.

CRITICAL - LOGIC COMPLETENESS:
- When adding UI elements (buttons, inputs, dropdowns), you MUST also add the JavaScript event listeners and handler functions in the SAME response.
- Never add HTML without wiring up the JS. An unconnected button is a bug.
- After adding interactive features, verify by adding a small EDIT that connects the event: addEventListener, onclick, onchange, etc.
- If you add a function reference, make sure the function is defined. If you add an event listener, make sure the element exists.
- Think through the data flow: where does data come from? Where is it stored? How do UI events trigger updates?

CRITICAL - JAVASCRIPT TEMPLATE LITERALS:
- In template literals, $$ before {variable} shows a literal dollar sign. To show "$70,000", use: '$' + value or put the $ inside the format function, NOT $$
- WRONG: `$$${price}` (shows $$70000)
- RIGHT: `$${price.toLocaleString()}` shows $70000 but ONLY if price is a string. Better: '$' + price.toLocaleString()
- Safest approach: create a formatPrice function that returns '$' + formatted number, then use ${formatPrice(value)} in the template.

CRITICAL - MULTI-FILE PROJECTS:
- When JS references a DOM element (getElementById, querySelector), that element MUST exist in the HTML.
- If you add JS that needs new HTML elements, edit BOTH the JS file AND the HTML file in the same response.
- READ all related files first (HTML + JS + CSS), then make coordinated EDITs across all files that need changes.
- Never edit just one file when the change spans multiple files."""


def parse_code_ops(text):
    """Parse LLM response for FILE:, EDIT:, RUN:, READ: markers."""
    operations = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("FILE:"):
            path = line[5:].strip().strip("`")
            i += 1
            content_lines = []
            in_block = False
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("```") and not in_block:
                    in_block = True
                    i += 1
                    continue
                elif stripped.startswith("```") and in_block:
                    i += 1
                    break
                elif in_block:
                    content_lines.append(lines[i])
                    i += 1
                else:
                    if stripped == "":
                        i += 1
                        continue
                    break
            if content_lines:
                operations.append({
                    "op": "file", "path": path,
                    "content": "\n".join(content_lines),
                })
            continue

        elif line.startswith("EDIT:"):
            path = line[5:].strip().strip("`")
            i += 1
            # Collect one or more search/replace blocks under this EDIT:
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("<<<"):
                    # Start a new search/replace block
                    search_lines = []
                    replace_lines = []
                    phase = "search"
                    i += 1
                    while i < len(lines):
                        stripped2 = lines[i].strip()
                        if stripped2.startswith("<<<"):
                            if phase == "search" and search_lines:
                                phase = "replace"
                            else:
                                phase = "search"
                                search_lines = []
                            i += 1
                            continue
                        elif stripped2 in ("===", "---", "===REPLACE", "=== REPLACE", "REPLACE:"):
                            phase = "replace"
                            i += 1
                            continue
                        elif stripped2 == ">>>":
                            i += 1
                            break
                        elif phase == "search":
                            search_lines.append(lines[i])
                        elif phase == "replace":
                            replace_lines.append(lines[i])
                        i += 1
                    if search_lines or replace_lines:
                        operations.append({
                            "op": "edit", "path": path,
                            "search": "\n".join(search_lines),
                            "replace": "\n".join(replace_lines),
                        })
                elif stripped.startswith(("EDIT:", "FILE:", "RUN:", "READ:", "MCP:")):
                    break  # Next operation starts
                elif stripped == "":
                    i += 1
                    continue
                else:
                    i += 1
                    continue
            continue

        elif line.startswith("DIFF:"):
            path = line[5:].strip().strip("`")
            i += 1
            # Collect the diff content (inside ``` block or raw)
            diff_lines = []
            in_block = False
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("```") and not in_block:
                    in_block = True
                    i += 1
                    continue
                elif stripped.startswith("```") and in_block:
                    i += 1
                    break
                elif in_block or stripped.startswith("---") or stripped.startswith("@@") or stripped.startswith("+") or stripped.startswith("-") or stripped.startswith(" "):
                    diff_lines.append(lines[i])
                    if not in_block and stripped == "" and diff_lines:
                        break
                else:
                    if diff_lines:
                        break
                i += 1
            if diff_lines:
                operations.append({"op": "diff", "path": path, "patch": "\n".join(diff_lines)})
            continue

        elif line.startswith("RUN:"):
            cmd = line[4:].strip().strip("`")
            if cmd:
                operations.append({"op": "run", "cmd": cmd})

        elif line.startswith("READ:"):
            path = line[5:].strip().strip("`")
            if path:
                operations.append({"op": "read", "path": path})

        elif line.startswith("MCP:"):
            tool_name = line[4:].strip().strip("`")
            i += 1
            args = {}
            # Look for ARGS: on next non-blank line
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("ARGS:"):
                    args_str = stripped[5:].strip()
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        pass
                    i += 1
                    break
                elif stripped == "":
                    i += 1
                    continue
                else:
                    break
            if tool_name:
                operations.append({"op": "mcp", "tool": tool_name, "args": args})
            continue

        i += 1

    return operations


def execute_code_op(op, work_dir):
    """Execute a single code operation and return result dict."""
    import subprocess as sp

    if op["op"] == "file":
        path = op["path"]
        if not os.path.isabs(path):
            path = os.path.join(work_dir, path)
        path = os.path.expanduser(path)
        dname = os.path.dirname(os.path.abspath(path))
        os.makedirs(dname, exist_ok=True)
        with open(path, "w") as f:
            f.write(op["content"])
        return {"type": "file_write", "path": path, "bytes": len(op["content"])}

    elif op["op"] == "edit":
        path = op["path"]
        if not os.path.isabs(path):
            path = os.path.join(work_dir, path)
        path = os.path.expanduser(path)
        try:
            # If file doesn't exist, auto-convert EDIT to FILE (create with replace content)
            if not os.path.exists(path):
                dname = os.path.dirname(os.path.abspath(path))
                os.makedirs(dname, exist_ok=True)
                with open(path, "w") as f:
                    f.write(op["replace"])
                return {"type": "file_write", "path": path, "bytes": len(op["replace"]), "note": "auto-created from EDIT (file did not exist)"}
            with open(path, "r") as f:
                content = f.read()

            # Try exact match first
            if op["search"] in content:
                # Find line number of match
                before_match = content[:content.index(op["search"])]
                start_line = before_match.count("\n") + 1
                content = content.replace(op["search"], op["replace"], 1)
                with open(path, "w") as f:
                    f.write(content)
                return {"type": "file_edit", "path": path,
                        "old": op["search"], "new": op["replace"], "line": start_line}

            # Fallback: normalize whitespace and try again
            # Strip trailing whitespace from each line in both search and content
            def normalize(text):
                return "\n".join(line.rstrip() for line in text.split("\n"))

            norm_content = normalize(content)
            norm_search = normalize(op["search"])

            if norm_search in norm_content:
                # Find the exact position in the original content
                # by matching line-by-line
                search_lines = op["search"].split("\n")
                content_lines = content.split("\n")
                for i in range(len(content_lines) - len(search_lines) + 1):
                    chunk = content_lines[i:i + len(search_lines)]
                    if normalize("\n".join(chunk)) == norm_search:
                        # Replace these lines
                        new_lines = content_lines[:i] + op["replace"].split("\n") + content_lines[i + len(search_lines):]
                        with open(path, "w") as f:
                            f.write("\n".join(new_lines))
                        return {"type": "file_edit", "path": path, "note": "fuzzy whitespace match"}

            # Fallback 2: try matching stripped key lines (first and last non-empty)
            # Only for search blocks with 3+ lines to avoid false positives
            search_stripped = [l.strip() for l in op["search"].split("\n") if l.strip()]
            if len(search_stripped) >= 3:
                content_lines = content.split("\n")
                first_key = search_stripped[0]
                last_key = search_stripped[-1]
                for i in range(len(content_lines)):
                    if content_lines[i].strip() == first_key:
                        for j in range(i + 1, min(i + len(search_stripped) + 3, len(content_lines))):
                            if content_lines[j].strip() == last_key:
                                chunk_stripped = [l.strip() for l in content_lines[i:j+1] if l.strip()]
                                # Require 90% overlap (was 70% - too loose, caused corruption)
                                if len(set(search_stripped) & set(chunk_stripped)) >= len(search_stripped) * 0.9:
                                    new_lines = content_lines[:i] + op["replace"].split("\n") + content_lines[j+1:]
                                    with open(path, "w") as f:
                                        f.write("\n".join(new_lines))
                                    return {"type": "file_edit", "path": path, "note": "fuzzy line match"}
                                break

            # Show context from the file to help the model correct its search
            search_first = op["search"].split("\n")[0].strip()
            context_hint = ""
            if search_first:
                for idx, line in enumerate(content.split("\n")):
                    if search_first[:20] in line or any(w in line for w in search_first.split()[:3] if len(w) > 3):
                        start = max(0, idx - 1)
                        end = min(len(content.split("\n")), idx + 4)
                        snippet = "\n".join(content.split("\n")[start:end])
                        context_hint = f"\nNearest match around line {idx+1}:\n{snippet}"
                        break
            return {"type": "file_edit", "path": path, "error": f"Search text not found in {os.path.basename(path)}{context_hint}"}
        except FileNotFoundError:
            basename = os.path.basename(path)
            suggestions = [os.path.relpath(os.path.join(r,f), work_dir)
                          for r, _, fs in os.walk(work_dir) for f in fs
                          if basename in f and not r.startswith(os.path.join(work_dir, '.')) and 'node_modules' not in r][:5]
            hint = f". Did you mean: {', '.join(suggestions)}" if suggestions else ""
            return {"type": "file_edit", "path": path, "error": f"File not found: {path}{hint}"}

    elif op["op"] == "diff":
        path = op["path"]
        patch = op["patch"]
        success, old_content, new_content, error = apply_udiff(path, patch, work_dir)
        if success:
            # Calculate what changed for the diff display
            old_lines = [l for l in patch.split("\n") if l.startswith("-") and not l.startswith("---")]
            new_lines = [l for l in patch.split("\n") if l.startswith("+") and not l.startswith("+++")]
            # Find start line from @@ header
            start_line = 1
            for l in patch.split("\n"):
                m = re.match(r'^@@ -(\d+)', l)
                if m:
                    start_line = int(m.group(1))
                    break
            return {
                "type": "file_diff", "path": path if os.path.isabs(path) else os.path.join(work_dir, path),
                "old": "\n".join(l[1:] for l in old_lines),
                "new": "\n".join(l[1:] for l in new_lines),
                "line": start_line,
            }
        else:
            return {"type": "file_diff", "path": path, "error": error or "Patch failed"}

    elif op["op"] == "run":
        cmd = op["cmd"]
        # Auto-detect dev servers and long-running processes -> background them
        bg_patterns = ['npm run dev', 'npm start', 'npx vite --', 'npx vite\n', 'python3 -m http.server',
                       'python -m http.server', 'node server', 'live-server', 'npx serve',
                       'flask run', 'uvicorn', 'next dev', 'yarn dev', 'bun dev']
        # These look similar but are NOT servers - they're one-time commands
        not_server = ['npm create', 'npm init', 'npm install', 'npm ci', 'npm build',
                      'npm test', 'npm run build', 'npx create-', 'npm exec']
        is_server = any(p in cmd for p in bg_patterns) or cmd.strip().endswith('&')
        if any(p in cmd for p in not_server):
            is_server = False
        cmd_clean = cmd.rstrip('& ')

        if is_server:
            jid = bg_jobs.start(cmd_clean, cwd=work_dir)
            if jid:
                # Wait for server to start - check multiple times
                port = bg_jobs.jobs[jid].get("port") or 5173  # default vite port
                alive = False
                for _ in range(10):  # check for up to 10 seconds
                    time.sleep(1)
                    if bg_jobs.jobs[jid]["process"].poll() is not None:
                        break  # process died
                    alive = True
                    # Try to detect the port from output
                    output_lines = bg_jobs.get_output(jid, lines=50) or []
                    for line in output_lines:
                        import re as _re
                        port_match = _re.search(r'(?:localhost|127\.0\.0\.1):(\d{4,5})', line)
                        if port_match:
                            port = int(port_match.group(1))
                            bg_jobs.jobs[jid]["port"] = port
                            break
                    if bg_jobs.jobs[jid].get("port"):
                        break

                if alive:
                    url = f"http://localhost:{port}"
                    return {
                        "type": "run_bg", "cmd": cmd_clean, "job_id": jid,
                        "port": port, "alive": True,
                        "output": f"Server running at {url}\nBackground job [{jid}] (pid {bg_jobs.jobs[jid]['pid']})",
                        "error": "",
                    }
                else:
                    # Process died - capture output for error message
                    err_lines = bg_jobs.get_output(jid, lines=20) or []
                    bg_jobs.stop(jid)
                    return {
                        "type": "run", "cmd": cmd_clean,
                        "output": "\n".join(err_lines)[:2000] if err_lines else "",
                        "error": "Server process exited. Check if dependencies are installed (npm install).",
                    }
            return {"type": "run", "cmd": cmd, "output": "", "error": "Failed to start background job"}

        try:
            # Longer timeout for install/create commands
            is_install = any(k in cmd_clean for k in ['npm create', 'npm install', 'npm init', 'npx create', 'pip install'])
            timeout = 300 if is_install else 120

            # Show running status
            short_cmd = cmd_clean[:50] + ("..." if len(cmd_clean) > 50 else "")
            console.print(f"  [dim]\u2699 running: {short_cmd}[/]", end="\r")

            result = sp.run(cmd_clean, shell=True, capture_output=True, text=True,
                          timeout=timeout, cwd=work_dir)
            output = result.stdout[:3000] if result.stdout else ""
            error = result.stderr[:1000] if result.returncode != 0 and result.stderr else ""
            return {"type": "run", "cmd": cmd, "output": output, "error": error, "code": result.returncode}
        except sp.TimeoutExpired:
            # If it timed out, it might be a server - background it
            jid = bg_jobs.start(cmd_clean, cwd=work_dir)
            if jid:
                return {
                    "type": "run_bg", "cmd": cmd_clean, "job_id": jid,
                    "output": f"Command ran for 120s, moved to background job [{jid}]",
                    "error": "",
                }
            return {"type": "run", "cmd": cmd, "output": "", "error": "Timed out after 120s"}
        except Exception as e:
            return {"type": "run", "cmd": cmd, "output": "", "error": str(e)}

    elif op["op"] == "read":
        path = op["path"]
        if not os.path.isabs(path):
            path = os.path.join(work_dir, path)
        path = os.path.expanduser(path)
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read(10000)
            return {"type": "read", "path": path, "content": content, "lines": content.count("\n") + 1}
        except FileNotFoundError:
            # Suggest similar files nearby
            basename = os.path.basename(path)
            suggestions = []
            for root, dirs, files in os.walk(work_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
                for f in files:
                    if basename in f or f.endswith(os.path.splitext(basename)[1]):
                        rel = os.path.relpath(os.path.join(root, f), work_dir)
                        suggestions.append(rel)
                if len(suggestions) >= 10:
                    break
            hint = ""
            if suggestions:
                hint = f". Did you mean: {', '.join(suggestions[:5])}"
            return {"type": "read", "path": path, "content": "", "error": f"File not found: {path}{hint}", "lines": 0}

    elif op["op"] == "mcp":
        tool_name = op["tool"]
        args = op["args"]
        try:
            result_text = call_mcp_tool(tool_name, args)
            is_error = result_text.startswith("Error:")
            return {
                "type": "mcp_call", "tool": tool_name, "args": args,
                "result": result_text[:5000],
                "error": result_text if is_error else None,
            }
        except Exception as e:
            return {"type": "mcp_call", "tool": tool_name, "args": args, "result": "", "error": str(e)}

    return {"type": "unknown"}


def is_truncated(text):
    """Check if response was truncated mid-code-block."""
    fences = text.count("```")
    if fences % 2 != 0:
        return True
    if text.rstrip().upper().endswith("CONTINUE"):
        return True
    return False


# ── picoclaw agent call with LIVE log streaming ───
def picoclaw_call_live(message, session="mac-code"):
    """Run picoclaw with real-time log streaming into animated display."""
    cmd = [PICOCLAW, "agent", "-m", message, "-s", session]
    display = WorkingDisplay()
    all_lines = []

    # Launch with Popen — picoclaw writes everything to stdout
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    # Read stdout line-by-line in a thread for real-time updates
    def read_output():
        try:
            for line in proc.stdout:
                all_lines.append(line)
                display.add_log(line)
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    # Animate while process runs
    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
        while proc.poll() is None:
            live.update(display.render())
            time.sleep(0.12)
        # Give reader a moment to finish
        time.sleep(0.3)
        live.update(display.render())

    reader.join(timeout=2)

    # Parse: strip ANSI, find lobster emoji, take text after it
    raw = "".join(all_lines)
    clean = strip_ansi(raw)

    idx = clean.rfind("\U0001f99e")  # last lobster emoji
    if idx >= 0:
        response = clean[idx:].lstrip("\U0001f99e").strip()
        # If it starts with "Error:" it's a picoclaw error, not a model response
        if response.startswith("Error:"):
            # Extract the useful part of the error
            response = f"[agent error] {response[:200]}"
    else:
        # No lobster — take non-banner lines
        lines = clean.split("\n")
        resp = []
        past = False
        for line in lines:
            s = line.strip()
            if not past:
                if not s or any(c in s for c in ["██", "╔", "╚", "╝", "║"]):
                    continue
                past = True
            if past and s:
                resp.append(s)
        response = "\n".join(resp).strip()

    return response, display.events

# ── banner ─────────────────────────────────────────
def print_banner(model_name, model_detail):
    console.print()
    logo = Text()
    logo.append("  \U0001f34e ", style="default")
    logo.append("mac", style="bold bright_cyan")
    logo.append(" ", style="default")
    logo.append("code", style="bold bright_yellow")
    console.print(logo)

    sub = Text()
    sub.append("  claude code, but it runs on your Mac for free", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", model_name, model_detail),
        ("tools", "search · fetch · exec · files", ""),
        ("cost", "$0.00/hr", "Apple M4 Metal · localhost:8000"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:6s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print("  [dim]type [bold bright_cyan]/[/bold bright_cyan] to see all commands[/]\n")

# ── render helpers ─────────────────────────────────
def render_diff_panel(path, old_text, new_text, start_line=1):
    """Render a Claude Code-style diff panel with colored additions/removals."""
    fname = os.path.basename(path)
    old_lines = old_text.split("\n") if old_text else []
    new_lines = new_text.split("\n") if new_text else []

    diff = Text()
    # Show removed lines (red background)
    for i, line in enumerate(old_lines):
        ln = start_line + i
        diff.append(f" {ln:4d} - ", style="bold red")
        diff.append(f"{line}\n", style="on #3c1010")
    # Show added lines (green background)
    for i, line in enumerate(new_lines):
        ln = start_line + i
        diff.append(f" {ln:4d} + ", style="bold green")
        diff.append(f"{line}\n", style="on #0c3c10")

    removed = len(old_lines)
    added = len(new_lines)
    summary = f"Added {added}, removed {removed} lines"

    panel = Panel(
        diff if diff.plain.strip() else Text("(empty change)"),
        title=f"[bold bright_cyan]\u25cf Update({fname})[/]",
        subtitle=f"[dim]{summary}[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )
    console.print(panel)

def render_bash_panel(cmd, output="", error=""):
    """Render a Claude Code-style bash command panel."""
    short_cmd = cmd[:80] + ("..." if len(cmd) > 80 else "")
    content = Text()
    if output:
        for line in output.split("\n")[:20]:
            content.append(f"  {line}\n", style="bright_green")
    if error:
        for line in error.split("\n")[:10]:
            content.append(f"  {line}\n", style="bright_red")
    if not content.plain.strip():
        content.append("  (no output)", style="dim")

    panel = Panel(
        content,
        title=f"[bold bright_yellow]\u25cf Bash[/]([dim]{short_cmd}[/])",
        border_style="bright_yellow",
        padding=(0, 1),
    )
    console.print(panel)

def render_create_panel(path, size):
    """Render a file creation panel."""
    fname = os.path.basename(path)
    console.print(Panel(
        f"[bold bright_green]\u2713[/] Created {size:,} bytes",
        title=f"[bold bright_green]Create({fname})[/]",
        border_style="bright_green",
        padding=(0, 1),
    ))

def render_response(response):
    """Render a response — use Rich Markdown if it has formatting, plain text otherwise."""
    if any(c in response for c in ["##", "**", "```", "| ", "- ", "1. ", "* "]):
        console.print(Padding(Markdown(response), (0, 2)))
    else:
        for line in response.split("\n"):
            console.print(f"  {line}")

def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s = Text()
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)

def render_timeline(events):
    """Show a compact summary of what the agent did."""
    if not events:
        return
    summary = []
    last_phase = None
    for ts, phase, detail in events:
        if phase != last_phase:
            summary.append(phase)
            last_phase = phase

    if len(summary) <= 1:
        return

    t = Text()
    t.append("  ", style="dim")
    for i, phase in enumerate(summary):
        t.append(phase, style="dim italic")
        if i < len(summary) - 1:
            t.append(" → ", style="dim")
    console.print(t)

# ── commands ───────────────────────────────────────
COMMANDS = [
    ("/agent",       "Switch to agent mode (tools + web search)"),
    ("/raw",         "Switch to raw mode (direct streaming, no tools)"),
    ("/btw",         "Ask a side question without adding to conversation history"),
    ("/loop",        "Run a prompt on a recurring interval — /loop 5m <prompt>"),
    ("/branch",      "Save conversation checkpoint you can restore later"),
    ("/restore",     "Restore last saved conversation checkpoint"),
    ("/add-dir",     "Set working directory — /add-dir <path>"),
    ("/save",        "Save conversation to a file — /save <filename>"),
    ("/search",      "Quick web search — /search <query>"),
    ("/bench",       "Run a quick speed benchmark"),
    ("/clear",       "Clear conversation and start fresh"),
    ("/stats",       "Show session statistics"),
    ("/model",       "Show or switch model — /model 9b or /model 35b"),
    ("/auto",        "Toggle smart auto-routing between 9B and 35B"),
    ("/tools",       "List available agent tools"),
    ("/system",      "Set system prompt — /system <message>"),
    ("/compact",     "Toggle compact output (no markdown rendering)"),
    ("/ps",          "Show running background jobs (dev servers, builds)"),
    ("/stop",        "Stop a background job or /loop - /stop <id>"),
    ("/logs",        "Show recent output from a background job - /logs <id>"),
    ("/run",         "Start a background job - /run npm run dev"),
    ("/sessions",    "List recent conversation sessions"),
    ("/resume",      "Resume a previous session - /resume <session_id>"),
    ("/cost",        "Show estimated cost savings vs cloud APIs"),
    ("/good",        "Grade last response as good (for self-improvement)"),
    ("/bad",         "Grade last response as bad (for self-improvement)"),
    ("/improve",     "Show self-improvement stats from logged interactions"),
    ("/quit",        "Exit mac code (conversation auto-saved)"),
]

def _slash_completer(text, state):
    """Tab-complete slash commands with descriptions shown inline."""
    if text.startswith("/"):
        options = [cmd for cmd, _ in COMMANDS if cmd.startswith(text)]
        if state == 0 and len(options) > 1:
            # Show all matching commands with descriptions
            console.print()
            for cmd, desc in COMMANDS:
                if cmd.startswith(text):
                    console.print(f"  [bold bright_cyan]{cmd:16}[/] [dim]{desc}[/]")
            console.print()
            # Re-display the prompt
            bg_status = bg_jobs.render_status()
            if bg_status:
                console.print(f"  [dim]auto[/] [dim cyan]{bg_status}[/]")
            console.print(f"  [dim]auto[/] [bold bright_yellow]>[/] {text}", end="")
            return None  # Don't complete yet, let user keep typing
    else:
        options = []
    return (options[state] + " ") if state < len(options) else None

def show_slash_menu(filter_text=""):
    """Print slash commands inline - like Claude Code."""
    matches = COMMANDS
    if filter_text and filter_text != "/":
        matches = [(c, d) for c, d in COMMANDS if c.startswith(filter_text)]

    for cmd, desc in matches:
        line = Text()
        line.append(f"  {cmd}", style="bold bright_cyan")
        pad = " " * max(14 - len(cmd), 1)
        line.append(pad)
        line.append(desc, style="dim")
        console.print(line)

# ── main ───────────────────────────────────────────
def main():
    _setup_readline()
    readline.set_completer(_slash_completer)
    console.clear()
    ensure_server_running()
    model_name, model_detail = detect_model()
    print_banner(model_name, model_detail)

    # Connect to MCP servers
    if MCP_CONFIG_PATH.exists():
        console.print("  [dim]connecting to MCP servers...[/]")
        load_mcp_servers()
        if mcp_clients:
            total_tools = sum(len(c.tools) for c in mcp_clients.values())
            console.print(f"  [dim]{total_tools} MCP tools available[/]")
        console.print()

    messages = []
    session_tokens = 0
    session_time = 0.0
    session_turns = 0
    session_id = f"mc-{int(time.time())}"
    use_agent = True
    compact_mode = False
    auto_route = True  # smart routing between 9B and 35B
    work_dir = os.getcwd()
    branch_save = None
    loop_thread = None
    loop_running = False
    last_interaction = None  # for /good /bad grading

    while True:
        try:
            bg_status = bg_jobs.render_status()
            if bg_status:
                console.print(f"  [dim cyan]{bg_status}[/]")
            console.print(f"  [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            if messages:
                save_session(session_id, messages, {
                    "tokens": session_tokens, "time": session_time, "turns": session_turns
                })
                console.print(f"  [dim]session saved: {session_id}[/]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip()
        cmd_lower = cmd.lower()

        # ── slash command handling ─────────────
        if cmd == "/":
            show_slash_menu()
            continue
        elif cmd_lower.startswith("/") and not cmd_lower.startswith("/system "):
            # Check for partial match — typing "/st" shows "/stats" and "/system"
            exact = cmd_lower.split()[0]

            if exact in ("/quit", "/exit", "/q"):
                # Auto-save session on exit
                if messages:
                    save_session(session_id, messages, {
                        "tokens": session_tokens, "time": session_time, "turns": session_turns
                    })
                    console.print(f"  [dim]session saved: {session_id}[/]")
                break
            elif exact == "/sessions":
                sessions = list_sessions()
                if not sessions:
                    console.print("  [dim]no saved sessions[/]\n")
                else:
                    t = Table(show_header=True, box=None, padding=(0, 1))
                    t.add_column("ID", style="bold bright_cyan", width=20)
                    t.add_column("Date", width=18)
                    t.add_column("Turns", width=6)
                    t.add_column("Preview")
                    for s in sessions:
                        t.add_row(s["id"], s["date"], str(s["turns"]), s["preview"])
                    console.print(t)
                    console.print(f"\n  [dim]resume with: /resume <session_id>[/]\n")
                continue
            elif exact == "/resume":
                parts = cmd.split()
                if len(parts) < 2:
                    # Show sessions and let user pick
                    sessions = list_sessions(5)
                    if not sessions:
                        console.print("  [dim]no saved sessions[/]\n")
                    else:
                        for i, s in enumerate(sessions):
                            console.print(f"  [bold bright_cyan]{s['id']}[/]  [dim]{s['date']}  {s['turns']} turns[/]  {s['preview']}")
                        console.print(f"\n  [dim]/resume <session_id>[/]\n")
                else:
                    target_id = parts[1]
                    # Support partial match
                    matches = [p.stem for p in SESSIONS_DIR.glob(f"{target_id}*.json")]
                    if len(matches) == 1:
                        target_id = matches[0]
                    data = load_session(target_id)
                    if data:
                        messages = data.get("messages", [])
                        session_id = data["session_id"]
                        session_turns = data.get("metadata", {}).get("turns", 0)
                        prev_dir = data.get("work_dir")
                        console.print(f"  [bold bright_green]\u2713[/] resumed session {session_id}")
                        console.print(f"  [dim]{len(messages)} messages, {session_turns} turns[/]")
                        if prev_dir and os.path.isdir(prev_dir):
                            work_dir = prev_dir
                            os.chdir(work_dir)
                            console.print(f"  [dim]work dir: {work_dir}[/]")
                        # Show last few messages as context
                        recent = [m for m in messages if m.get("role") == "user"][-3:]
                        for m in recent:
                            console.print(f"  [dim]> {m['content'][:80]}[/]")
                        console.print()
                    else:
                        console.print(f"  [bold red]session not found: {target_id}[/]\n")
                continue
            elif exact == "/clear":
                messages.clear()
                session_id = f"mc-{int(time.time())}"
                console.clear()
                print_banner(model_name, model_detail)
                console.print("  [dim]cleared.[/]\n")
                continue
            elif exact == "/stats":
                avg = session_tokens / session_time if session_time > 0 else 0
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=12)
                t.add_column()
                t.add_row("turns", str(session_turns))
                t.add_row("tokens", f"{session_tokens:,}")
                t.add_row("time", f"{session_time:.1f}s")
                t.add_row("avg speed", f"{avg:.1f} tok/s")
                t.add_row("mode", tag)
                console.print(t)
                console.print()
                continue
            elif exact == "/ps":
                jobs = bg_jobs.list_jobs()
                if not jobs:
                    console.print("  [dim]no background jobs running[/]\n")
                else:
                    t = Table(show_header=True, box=None, padding=(0, 1))
                    t.add_column("ID", style="bold bright_cyan", width=4)
                    t.add_column("PID", width=8)
                    t.add_column("Command")
                    t.add_column("Port", width=6)
                    t.add_column("Uptime", width=10)
                    for jid, job in jobs.items():
                        elapsed = int(time.time() - job["started"])
                        mins, secs = divmod(elapsed, 60)
                        t.add_row(
                            str(jid), str(job["pid"]),
                            job["cmd"][:50],
                            str(job.get("port", "-")),
                            f"{mins}m{secs}s",
                        )
                    console.print(t)
                    console.print()
                continue
            elif exact == "/run":
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [dim]usage: /run <command>[/]\n")
                else:
                    jid = bg_jobs.start(parts[1], cwd=work_dir)
                    if jid:
                        time.sleep(2)
                        port = bg_jobs.jobs[jid].get("port")
                        console.print(f"  [bold bright_green]\u2713[/] started job [{jid}] pid={bg_jobs.jobs[jid]['pid']}" + (f" port={port}" if port else ""))
                    else:
                        console.print("  [bold red]failed to start[/]")
                    console.print()
                continue
            elif exact == "/logs":
                parts = cmd.split()
                if len(parts) < 2:
                    console.print("  [dim]usage: /logs <job_id>[/]\n")
                else:
                    try:
                        jid = int(parts[1])
                        lines = bg_jobs.get_output(jid)
                        if lines is None:
                            console.print(f"  [dim]job {jid} not found[/]")
                        elif not lines:
                            console.print(f"  [dim]no new output from job {jid}[/]")
                        else:
                            for line in lines:
                                console.print(f"  {line}")
                    except ValueError:
                        console.print("  [dim]usage: /logs <job_id>[/]")
                    console.print()
                continue
            elif exact == "/stop" and len(cmd.split()) >= 2:
                parts = cmd.split()
                try:
                    jid = int(parts[1])
                    if bg_jobs.stop(jid):
                        console.print(f"  [dim]stopped job {jid}[/]\n")
                    else:
                        console.print(f"  [dim]job {jid} not found[/]\n")
                except ValueError:
                    console.print("  [dim]usage: /stop <job_id>[/]\n")
                continue
            elif exact == "/model":
                # Check if user passed an argument like "/model 9b"
                parts = cmd.split()
                if len(parts) >= 2:
                    target = parts[1].lower().replace("b", "b")
                    if target in MODELS:
                        console.print(f"  [dim]swapping to {MODELS[target]['name']}...[/]")
                        display = WorkingDisplay()
                        display.phase = f"loading {MODELS[target]['name']}"
                        with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                            ok, msg = swap_model(target)
                            while not ok and display.frame < 100:
                                display.frame += 1
                                live.update(display.render())
                                time.sleep(0.2)
                        if ok:
                            model_name = MODELS[target]["name"]
                            model_detail = MODELS[target]["detail"]
                            console.print(f"  [bold bright_green]{msg}[/]\n")
                        else:
                            console.print(f"  [bold red]{msg}[/]\n")
                    else:
                        console.print(f"  [dim]available: 9b, 35b[/]\n")
                else:
                    cur = get_current_model()
                    model_name, model_detail = detect_model()
                    console.print(f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]")
                    console.print(f"  [dim]auto-routing: {'on' if auto_route else 'off'}[/]")
                    console.print(f"  [dim]switch: /model 9b  or  /model 35b[/]\n")
                continue

            elif exact == "/auto":
                auto_route = not auto_route
                state = "on" if auto_route else "off"
                console.print(f"  [dim]smart auto-routing {state}[/]")
                if auto_route:
                    console.print(f"  [dim]  tools/search → 9B (32K ctx, reliable)[/]")
                    console.print(f"  [dim]  reasoning     → 35B (faster, smarter)[/]")
                console.print()
                continue
            elif exact == "/tools":
                for name, desc in [
                    ("web_search", "DuckDuckGo"), ("web_fetch", "read URLs"),
                    ("exec", "shell commands"), ("read_file", "local files"),
                    ("write_file", "create files"), ("edit_file", "modify files"),
                    ("list_dir", "browse dirs"), ("subagent", "spawn tasks"),
                ]:
                    t = Text()
                    t.append("  ▸ ", style="bright_cyan")
                    t.append(name, style="bold bright_cyan")
                    t.append(f"  {desc}", style="dim")
                    console.print(t)
                console.print()
                continue
            elif exact == "/agent":
                use_agent = True
                console.print("  [dim]agent mode (tools enabled)[/]\n")
                continue
            elif exact == "/raw":
                use_agent = False
                console.print("  [dim]raw mode (streaming, no tools)[/]\n")
                continue
            elif exact == "/compact":
                compact_mode = not compact_mode
                state = "on" if compact_mode else "off"
                console.print(f"  [dim]compact mode {state}[/]\n")
                continue

            elif exact == "/branch":
                branch_save = [m.copy() for m in messages]
                console.print(f"  [dim]conversation saved ({len(messages)} messages). use /restore to go back.[/]\n")
                continue

            elif exact == "/restore":
                if branch_save is not None:
                    messages = [m.copy() for m in branch_save]
                    console.print(f"  [dim]restored to checkpoint ({len(messages)} messages)[/]\n")
                else:
                    console.print("  [dim]no checkpoint saved. use /branch first.[/]\n")
                continue

            elif exact == "/bench":
                console.print("  [dim]running speed benchmark...[/]")
                try:
                    payload = json.dumps({
                        "model": "local",
                        "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
                        "max_tokens": 300, "temperature": 0.1,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    bstart = time.time()
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        d = json.loads(resp.read())
                    belapsed = time.time() - bstart
                    t = d.get("timings", {})
                    u = d.get("usage", {})
                    gen_speed = t.get("predicted_per_second", 0)
                    prompt_speed = t.get("prompt_per_second", 0)
                    tokens = u.get("completion_tokens", 0)
                    console.print(f"  [bold bright_green]{gen_speed:.1f} tok/s[/] generation")
                    console.print(f"  [bold bright_green]{prompt_speed:.1f} tok/s[/] prompt processing")
                    console.print(f"  [dim]{tokens} tokens in {belapsed:.1f}s[/]\n")
                except Exception as e:
                    console.print(f"  [bold red]benchmark failed: {e}[/]\n")
                continue

            elif exact == "/cost":
                cloud_rate = 0.34  # $/hr RunPod equivalent
                hours = session_time / 3600 if session_time > 0 else 0
                saved = cloud_rate * max(hours, 1/60)
                console.print(f"  [bold bright_green]$0.00[/] spent locally")
                console.print(f"  [dim]~${saved:.4f} would have cost on cloud GPU (${cloud_rate}/hr)[/]")
                console.print(f"  [dim]session: {session_time:.0f}s · {session_tokens:,} tokens[/]\n")
                continue

            elif exact == "/good":
                # Grade last response as good
                if last_interaction:
                    last_interaction["grade"] = "good"
                    log_interaction(**last_interaction)
                    console.print("  [bright_green]marked good[/]\n")
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/bad":
                # Grade last response as bad
                if last_interaction:
                    last_interaction["grade"] = "bad"
                    log_interaction(**last_interaction)
                    console.print("  [bright_red]marked bad — logged for improvement[/]\n")
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/improve":
                stats = get_failure_stats()
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=14)
                t.add_column()
                t.add_row("total", str(stats["total"]))
                t.add_row("good", str(stats["graded"].get("good", 0)))
                t.add_row("bad", str(stats["graded"].get("bad", 0)))
                t.add_row("errors", str(stats["errors"]))
                t.add_row("searches", str(stats["intents"].get("search", 0)))
                t.add_row("shell", str(stats["intents"].get("shell", 0)))
                t.add_row("chat", str(stats["intents"].get("chat", 0)))
                t.add_row("logs", str(LOGS_DIR))
                console.print(t)
                console.print()
                continue

            elif exact in ("/help", "/?"):
                show_slash_menu()
                continue
            else:
                # Partial match — show filtered results
                show_slash_menu(exact)
                continue

        # ── commands with arguments ────────────
        elif cmd_lower.startswith("/system "):
            sys_msg = cmd[8:].strip()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]system: {sys_msg[:80]}[/]\n")
            continue

        elif cmd_lower.startswith("/btw "):
            # Side question — don't add to conversation history
            side_q = cmd[5:].strip()
            if not side_q:
                console.print("  [dim]/btw <question>[/]\n")
                continue
            console.print()
            if use_agent:
                start = time.time()
                # Use a separate session so it doesn't pollute main conversation
                response, events = picoclaw_call_live(side_q, session=f"btw-{int(time.time())}")
                elapsed = time.time() - start
                if response:
                    console.print(f"  [dim italic](side answer)[/]")
                    render_response(response)
                    console.print()
                    tokens_est = len(response.split())
                    render_speed(tokens_est, elapsed)
                    session_tokens += tokens_est
                    session_time += elapsed
            else:
                side_msgs = [{"role": "user", "content": side_q}]
                try:
                    payload = json.dumps({
                        "model": "local", "messages": side_msgs,
                        "max_tokens": 2000, "temperature": 0.7,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        d = json.loads(resp.read())
                    content = d["choices"][0]["message"]["content"]
                    console.print(f"  [dim italic](side answer)[/]")
                    for line in content.split("\n"):
                        console.print(f"  {line}")
                except Exception as e:
                    console.print(f"  [bold red]{e}[/]")
            console.print()
            continue

        elif cmd_lower.startswith("/add-dir "):
            new_dir = os.path.expanduser(cmd[9:].strip())
            if os.path.isdir(new_dir):
                work_dir = new_dir
                os.chdir(work_dir)
                console.print(f"  [dim]working directory: {work_dir}[/]\n")
            else:
                console.print(f"  [bold red]not a directory: {new_dir}[/]\n")
            continue

        elif cmd_lower.startswith("/save "):
            filename = cmd[6:].strip()
            if not filename:
                filename = f"conversation-{int(time.time())}.json"
            try:
                save_path = os.path.join(work_dir, filename)
                with open(save_path, "w") as f:
                    json.dump({
                        "messages": messages,
                        "session_id": session_id,
                        "tokens": session_tokens,
                        "time": session_time,
                        "turns": session_turns,
                    }, f, indent=2)
                console.print(f"  [dim]saved to {save_path}[/]\n")
            except Exception as e:
                console.print(f"  [bold red]{e}[/]\n")
            continue

        elif cmd_lower.startswith("/search "):
            query = cmd[8:].strip()
            if not query:
                console.print("  [dim]/search <query>[/]\n")
                continue
            console.print()
            start = time.time()
            response, events = picoclaw_call_live(
                f"Search the web for: {query}. Give a brief summary of the top results.",
                session=f"search-{int(time.time())}"
            )
            elapsed = time.time() - start
            if response:
                for line in response.split("\n"):
                    console.print(f"  {line}")
                console.print()
                s = Text()
                s.append(f"  ▸ agent", style="bold bright_cyan")
                s.append(f"  {elapsed:.1f}s total (search + inference)", style="dim")
                console.print(s)
                session_tokens += len(response.split())
                session_time += elapsed
            console.print()
            continue

        elif cmd_lower.startswith("/loop "):
            # Parse: /loop 5m <prompt>
            parts = cmd[6:].strip().split(None, 1)
            if len(parts) < 2:
                console.print("  [dim]/loop <interval> <prompt>  — e.g. /loop 5m check server status[/]\n")
                continue

            interval_str, loop_prompt = parts
            # Parse interval
            try:
                if interval_str.endswith("m"):
                    interval_sec = int(interval_str[:-1]) * 60
                elif interval_str.endswith("s"):
                    interval_sec = int(interval_str[:-1])
                elif interval_str.endswith("h"):
                    interval_sec = int(interval_str[:-1]) * 3600
                else:
                    interval_sec = int(interval_str) * 60  # default minutes
            except ValueError:
                console.print(f"  [bold red]invalid interval: {interval_str}[/]\n")
                continue

            if loop_running:
                loop_running = False
                console.print("  [dim]stopped previous loop[/]")
                time.sleep(1)

            loop_running = True
            console.print(f"  [dim]looping every {interval_sec}s: {loop_prompt}[/]")
            console.print(f"  [dim]type /stop to cancel[/]\n")

            def run_loop(prompt, interval, sid):
                nonlocal loop_running, session_tokens, session_time
                while loop_running:
                    time.sleep(interval)
                    if not loop_running:
                        break
                    console.print(f"\n  [dim italic]loop: running '{prompt[:40]}...'[/]")
                    resp, _ = picoclaw_call_live(prompt, session=sid)
                    if resp:
                        for line in resp.split("\n"):
                            console.print(f"  {line}")
                    console.print()

            loop_thread = threading.Thread(
                target=run_loop,
                args=(loop_prompt, interval_sec, f"loop-{session_id}"),
                daemon=True
            )
            loop_thread.start()
            continue

        elif cmd_lower == "/stop":
            if loop_running:
                loop_running = False
                console.print("  [dim]loop stopped[/]\n")
            else:
                console.print("  [dim]no loop running[/]\n")
            continue

        console.print()

        # ── agent mode ─────────────────────────────
        if use_agent:
            start = time.time()

            # LLM classifies intent: search, shell, or chat (~1s)
            display = WorkingDisplay()
            display.phase = "classifying"
            intent_result = [None]

            def do_classify():
                intent_result[0] = classify_intent(user_input)

            cls_thread = threading.Thread(target=do_classify, daemon=True)
            cls_thread.start()

            with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                while cls_thread.is_alive():
                    display.frame += 1
                    live.update(display.render())
                    time.sleep(0.12)

            cls_thread.join(timeout=1)
            intent = intent_result[0] or "chat"

            # Sticky intent: if last turn was code, short follow-ups stay in code mode
            # ("fix it", "but it's still small", "now add X", "continue", etc.)
            if last_interaction and last_interaction.get("intent") == "code" and intent != "search":
                intent = "code"

            # Route based on LLM classification
            if intent == "shell":
                # Phase 1: Generate command + execute (with spinner)
                display = WorkingDisplay()
                display.phase = "generating command"
                prep_result = [None]

                def do_prep_shell():
                    try:
                        prep_result[0] = prepare_shell(user_input, work_dir)
                    except Exception:
                        prep_result[0] = None

                prep_thread = threading.Thread(target=do_prep_shell, daemon=True)
                prep_thread.start()

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    while prep_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "generating command"
                        else:
                            display.phase = "executing"
                        live.update(display.render())
                        time.sleep(0.12)

                prep_thread.join(timeout=1)

                if prep_result[0]:
                    cmd, summary_msgs = prep_result[0]
                    console.print()
                    console.print(f"  [dim]$ {cmd}[/]")
                    console.print()

                    # Phase 2: Stream the summary
                    full = ""
                    tokens = 0
                    console.print("  ", end="")
                    try:
                        for chunk in stream_chat(summary_msgs, max_tokens=1000):
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1
                        elapsed = time.time() - start
                        console.print("\n")
                        render_speed(tokens, elapsed)
                        session_tokens += tokens
                        session_time += elapsed
                        session_turns += 1
                        speed = tokens / max(elapsed, 0.1)
                        last_interaction = {"query": user_input, "intent": "shell", "response": full, "speed": speed}
                        messages.append({"role": "user", "content": user_input})
                        messages.append({"role": "assistant", "content": full})
                    except Exception as e:
                        console.print(f"\n  [bold red]{e}[/]")
                else:
                    console.print(f"  [bold red]command failed[/]\n")

            elif intent == "search":
                # Phase 1: Rewrite query + DuckDuckGo search (with spinner)
                display = WorkingDisplay()
                display.phase = "searching the web"
                search_prep = [None]

                def do_prep_search():
                    try:
                        search_prep[0] = prepare_search(user_input)
                    except Exception:
                        search_prep[0] = None

                search_thread = threading.Thread(target=do_prep_search, daemon=True)
                search_thread.start()

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    while search_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "rewriting query"
                        else:
                            display.phase = "searching the web"
                        live.update(display.render())
                        time.sleep(0.12)

                search_thread.join(timeout=1)

                if search_prep[0]:
                    summary_msgs = search_prep[0]
                    console.print()

                    # Phase 2: Stream the answer
                    full = ""
                    tokens = 0
                    console.print("  ", end="")
                    try:
                        for chunk in stream_chat(summary_msgs, max_tokens=1000):
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1
                        elapsed = time.time() - start
                        console.print("\n")
                        render_speed(tokens, elapsed)
                        session_tokens += tokens
                        session_time += elapsed
                        session_turns += 1
                        messages.append({"role": "user", "content": user_input})
                        messages.append({"role": "assistant", "content": full})
                        speed = tokens / max(elapsed, 0.1)
                        last_interaction = {"query": user_input, "intent": "search", "response": full, "speed": speed}
                    except Exception as e:
                        console.print(f"\n  [bold red]{e}[/]")
                else:
                    # Search failed, fall back to direct LLM streaming
                    console.print("  [dim]search failed, asking model directly...[/]")
                    messages.append({"role": "user", "content": user_input})
                    full = ""
                    tokens = 0
                    first_token = True
                    display.phase = "thinking"
                    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                        gen = stream_llm(messages)
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})
            elif intent == "code":
                # Coding agent — write/edit/run files with auto-continue
                today = datetime.now().strftime("%A, %B %d, %Y")

                # Scan project files so the model knows what exists
                project_files = []
                try:
                    for root, dirs, files in os.walk(work_dir):
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build')]
                        depth = root.replace(work_dir, '').count(os.sep)
                        if depth > 3:
                            dirs.clear()
                            continue
                        for f in files:
                            if not f.startswith('.') and not f.endswith(('.pyc', '.o', '.so', '.dylib')):
                                rel = os.path.relpath(os.path.join(root, f), work_dir)
                                sz = os.path.getsize(os.path.join(root, f))
                                project_files.append(f"  {rel} ({sz}b)")
                            if len(project_files) >= 50:
                                break
                        if len(project_files) >= 50:
                            break
                except Exception:
                    pass
                file_listing = "\n".join(project_files[:50]) if project_files else "(empty)"

                # Load project-specific instructions (like CLAUDE.md / AGENTS.md)
                project_instructions = ""
                for name in ("AGENTS.md", ".mac-code.md", "CLAUDE.md"):
                    agent_file = os.path.join(work_dir, name)
                    if os.path.isfile(agent_file):
                        try:
                            with open(agent_file, "r") as f:
                                project_instructions = f"\n\nProject instructions ({name}):\n{f.read(3000)}"
                            break
                        except Exception:
                            pass

                code_msgs = [
                    {"role": "system", "content": f"Today is {today}. Working directory: {work_dir}\n\nProject files:\n{file_listing}{project_instructions}\n\n{CODE_SYSTEM}{get_mcp_tool_descriptions()}"},
                ]
                for msg in messages[-20:]:
                    code_msgs.append(msg)
                code_msgs.append({"role": "user", "content": user_input})

                total_tokens = 0
                full_response = ""
                consecutive_edit_fails = 0

                for iteration in range(MAX_ITERATIONS):
                    # Show status while waiting for LLM
                    if iteration == 0:
                        status_msg = "  [dim]\u2699 thinking...[/]"
                    else:
                        status_msg = f"  [dim]\u2699 working... (step {iteration + 1})[/]"

                    # Stream LLM response with status
                    chunk_response = ""
                    chunk_tokens = 0
                    first_chunk = True

                    try:
                        with Live(status_msg, console=console, refresh_per_second=4, transient=True) as status:
                            for chunk in stream_chat(code_msgs, max_tokens=16000, temperature=0.3):
                                if first_chunk:
                                    first_chunk = False
                                    status.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                chunk_response += chunk
                                chunk_tokens += 1
                            if first_chunk:
                                # Non-streaming response (MLX) - show it
                                status.stop()
                                if chunk_response:
                                    console.print(f"  {chunk_response}", end="")
                    except Exception as e:
                        console.print(f"\n  [bold red]{e}[/]")
                        break

                    console.print("\n")
                    full_response += chunk_response
                    total_tokens += chunk_tokens

                    # Parse and execute file operations
                    ops = parse_code_ops(chunk_response)
                    truncated = is_truncated(chunk_response)

                    if ops:
                        console.print()
                        results = []
                        for op in ops:
                            result = execute_code_op(op, work_dir)
                            results.append(result)

                            if result["type"] == "file_write":
                                render_create_panel(result["path"], result["bytes"])
                            elif result["type"] in ("file_edit", "file_diff"):
                                if result.get("error"):
                                    console.print(f"  [bold red]\u2717[/] edit failed: {result['error']}")
                                else:
                                    render_diff_panel(
                                        result["path"],
                                        result.get("old", ""),
                                        result.get("new", ""),
                                        result.get("line", 1),
                                    )
                            elif result["type"] in ("run", "run_bg"):
                                if result["type"] == "run_bg":
                                    jid = result.get("job_id", "?")
                                    port = result.get("port")
                                    if result.get("alive"):
                                        render_bash_panel(result["cmd"], result.get("output", ""))
                                        console.print(f"  [bold bright_green]\u2713[/] server at [bold bright_cyan]http://localhost:{port}[/]  (job [{jid}])")
                                    else:
                                        render_bash_panel(result["cmd"], error=result.get("error", "failed"))
                                else:
                                    render_bash_panel(result["cmd"], result.get("output", ""), result.get("error", ""))
                            elif result["type"] == "read":
                                if result.get("error"):
                                    console.print(f"  [bold red]\u2717[/] {result['error']}")
                                else:
                                    console.print(f"  [dim]read {result['path']} ({result['lines']} lines)[/]")
                            elif result["type"] == "mcp_call":
                                tool = result.get("tool", "?")
                                if result.get("error"):
                                    console.print(f"  [bold red]\u2717[/] mcp:{tool} - {result['error'][:200]}")
                                else:
                                    preview = result.get("result", "")[:200]
                                    console.print(f"  [bold bright_green]\u2713[/] mcp:{tool} ({len(result.get('result',''))} chars)")

                        console.print()

                        # Check if we need to iterate
                        errors = [r for r in results if r.get("error")]
                        reads = [r for r in results if r["type"] == "read"]
                        mcp_calls = [r for r in results if r["type"] == "mcp_call"]

                        # Auto-lint: ALWAYS check syntax after any file write/edit
                        lint_errors = []
                        code_files = [r for r in results if r["type"] in ("file_write", "file_edit", "file_diff") and r.get("path") and not r.get("error")]
                        SYNTAX_CHECKS = {
                            ".py": ["python3", "-m", "py_compile"],
                            ".js": ["node", "--check"],
                            ".ts": ["npx", "tsc", "--noEmit"],
                            ".rb": ["ruby", "-c"],
                            ".go": ["gofmt", "-e"],
                            ".rs": ["rustfmt", "--check"],
                            ".sh": ["bash", "-n"],
                            ".html": None,
                        }
                        import subprocess as _sp
                        for cf in code_files[:5]:
                            fpath = cf["path"]
                            ext = os.path.splitext(fpath)[1].lower()
                            checker = SYNTAX_CHECKS.get(ext)
                            if checker is None and ext == ".html":
                                try:
                                    # Syntax check
                                    check = _sp.run(
                                        ["node", "-e", f"const h=require('fs').readFileSync('{fpath}','utf8');"
                                         "const m=h.match(/<script[^>]*>([\\s\\S]*?)<\\/script>/g)||[];"
                                         "m.forEach(s=>{{const c=s.replace(/<\\/?script[^>]*>/g,'');new Function(c)}})"],
                                        capture_output=True, text=True, timeout=5)
                                    if check.returncode != 0 and check.stderr:
                                        err = check.stderr.strip()[:500]
                                        console.print(f"  [bold yellow]![/] lint: {err[:200]}")
                                        lint_errors.append(f"Syntax error in {fpath}: {err}")
                                except Exception:
                                    pass
                                # DOM + function reference checks
                                try:
                                    with open(fpath, "r") as _f:
                                        html = _f.read()
                                    import re as _re
                                    # Check getElementById targets exist
                                    js_ids = set(_re.findall(r'getElementById\([\'"]([a-zA-Z0-9_-]+)[\'"]\)', html))
                                    html_ids = set(_re.findall(r'id=[\'"]([a-zA-Z0-9_-]+)[\'"]', html))
                                    missing = js_ids - html_ids
                                    if missing:
                                        msg = f"DOM mismatch in {os.path.basename(fpath)}: JS calls getElementById for {missing} but these IDs don't exist in the HTML. Add the missing elements."
                                        console.print(f"  [bold yellow]![/] {msg[:200]}")
                                        lint_errors.append(msg)
                                    # Check function calls match definitions in inline scripts
                                    scripts = _re.findall(r'<script[^>]*>([\s\S]*?)</script>', html)
                                    if scripts:
                                        all_js = "\n".join(scripts)
                                        defined = set(_re.findall(r'(?:function\s+|(?:const|let|var)\s+)(\w+)', all_js))
                                        called = set(_re.findall(r'(?<!function\s)(?<!\.)\b(\w+)\s*\(', all_js))
                                        # Filter out built-ins and common globals
                                        builtins = {'fetch','console','document','window','setTimeout','setInterval','clearInterval','clearTimeout','parseInt','parseFloat','JSON','Array','Object','Math','Date','Error','Promise','alert','confirm','encodeURIComponent','decodeURIComponent','addEventListener','removeEventListener','querySelector','querySelectorAll','getElementById','createElement','appendChild','forEach','map','filter','reduce','find','push','splice','join','split','trim','replace','match','test','toString','toFixed','toLocaleString','includes','toLowerCase','toUpperCase','sort','reverse','keys','values','entries','from','assign','stringify','parse','log','error','warn','catch','then','finally','resolve','reject','require','Number','String','Boolean','Set','Map','RegExp','isNaN','Infinity','undefined','NaN','Symbol','BigInt','Proxy','Reflect'}
                                        missing_fns = called - defined - builtins
                                        # Only report if the function name looks user-defined (not a method)
                                        missing_fns = {f for f in missing_fns if len(f) > 2 and f[0].islower()}
                                        if missing_fns:
                                            msg = f"Possible undefined functions in {os.path.basename(fpath)}: {missing_fns}. Make sure all called functions are defined."
                                            console.print(f"  [bold yellow]![/] {msg[:200]}")
                                            lint_errors.append(msg)
                                        # Check for fetch/init functions defined but never called
                                        async_fns = set(_re.findall(r'async\s+function\s+(\w+)', all_js))
                                        uncalled = async_fns - called
                                        if uncalled:
                                            msg = f"Warning in {os.path.basename(fpath)}: async functions {uncalled} are defined but never called. Add initialization calls."
                                            console.print(f"  [bold yellow]![/] {msg[:200]}")
                                            lint_errors.append(msg)
                                except Exception:
                                    pass
                            elif checker:
                                try:
                                    check = _sp.run(checker + [fpath], capture_output=True, text=True, timeout=5)
                                    if check.returncode != 0:
                                        err = (check.stderr or check.stdout).strip()[:500]
                                        console.print(f"  [bold yellow]![/] lint: {err[:200]}")
                                        lint_errors.append(f"Syntax error in {fpath}: {err}")
                                except FileNotFoundError:
                                    pass
                                except Exception:
                                    pass
                        if lint_errors:
                            errors.extend([{"error": e} for e in lint_errors])

                        if not errors and not truncated and not reads and not mcp_calls:
                            break

                        # Build feedback for next iteration
                        feedback = []
                        # If there are lint errors, auto-read the broken files so the model can fix them
                        if lint_errors:
                            for lerr in lint_errors:
                                feedback.append(f"LINT ERROR (you must fix this): {lerr}")
                                # Extract file path from error and include its content
                                for cf in code_files:
                                    if cf.get("path") and cf["path"] in lerr:
                                        try:
                                            with open(cf["path"], "r") as f:
                                                content = f.read(10000)
                                            feedback.append(f"Current content of {cf['path']}:\n{content}")
                                        except Exception:
                                            pass
                                        break
                        edit_fails_this_round = sum(1 for r in results if r.get("type") == "file_edit" and r.get("error"))
                        if edit_fails_this_round > 0:
                            consecutive_edit_fails += edit_fails_this_round
                        else:
                            consecutive_edit_fails = 0

                        for r in results:
                            if r.get("error") and r not in [{"error": e} for e in lint_errors]:
                                if consecutive_edit_fails >= 3:
                                    # After 3+ failed EDITs, tell the model to use FILE: instead
                                    fpath = r.get("path", "the file")
                                    feedback.append(f"EDIT has failed {consecutive_edit_fails} times on {fpath}. STOP using EDIT. Use FILE: to rewrite the entire file instead. READ it first, then FILE: with the complete new content.")
                                    consecutive_edit_fails = 0  # reset after switching strategy
                                else:
                                    feedback.append(f"Error: {r['error']}\nUse the exact text shown in 'Nearest match' for your EDIT search block.")
                            if r["type"] == "run" and r.get("output"):
                                feedback.append(f"Output of `{r['cmd']}`:\n{r['output'][:2000]}")
                            if r["type"] == "read" and r.get("content"):
                                feedback.append(f"Contents of {r['path']}:\n{r['content'][:10000]}")
                            if r["type"] == "mcp_call" and r.get("result") and not r.get("error"):
                                feedback.append(f"Result of MCP tool '{r['tool']}':\n{r['result'][:4000]}")

                        if truncated:
                            feedback.append("Your response was cut off. Continue from where you left off.")

                        code_msgs.append({"role": "assistant", "content": chunk_response})
                        code_msgs.append({"role": "user", "content": "\n\n".join(feedback) if feedback else "Continue."})
                        console.print(f"  [dim]iterating... ({iteration + 2})[/]\n")

                    elif truncated:
                        # Response was cut off mid-code-block but no complete ops parsed
                        console.print(f"  [dim]response truncated, continuing... ({iteration + 2})[/]\n")
                        code_msgs.append({"role": "assistant", "content": chunk_response})
                        code_msgs.append({"role": "user", "content": "Your response was cut off mid-code. Please rewrite the complete file from the beginning using FILE: markers."})

                    else:
                        break

                elapsed = time.time() - start
                render_speed(total_tokens, elapsed)
                session_tokens += total_tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": full_response})
                speed = total_tokens / max(elapsed, 0.1)
                last_interaction = {"query": user_input, "intent": "code", "response": full_response, "speed": speed}

            else:
                # Direct LLM streaming (no tools needed)
                messages.append({"role": "user", "content": user_input})
                full = ""
                tokens = 0
                first_token = True
                display = WorkingDisplay()
                display.phase = "thinking"
                try:
                    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                        gen = stream_llm(messages)
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    console.print(f"\n  [bold red]{e}[/]")
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()

        # ── raw streaming mode ─────────────────────
        else:
            messages.append({"role": "user", "content": user_input})
            full = ""
            tokens = 0
            start = time.time()

            try:
                display = WorkingDisplay()
                display.phase = "thinking"
                first_token = True

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    gen = stream_llm(messages)
                    for chunk in gen:
                        if isinstance(chunk, str):
                            if first_token:
                                first_token = False
                                live.stop()
                                console.print("  ", end="")
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1

                elapsed = time.time() - start
                if not compact_mode and any(c in full for c in ["##", "**", "```", "- ", "1. "]):
                    console.print("\n")
                    console.print(Padding(Markdown(full), (0, 2)))
                else:
                    console.print("\n")
                render_speed(tokens, elapsed)
                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "assistant", "content": full})

            except Exception as e:
                console.print(f"  [bold red]{e}[/]")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

        console.print()

    # ── exit ───────────────────────────────────────
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  \U0001f34e [bold bright_cyan]mac[/] [bold bright_yellow]code[/]"
            f"  [dim]{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} tok/s[/]"
        )
    console.print()

if __name__ == "__main__":
    try:
        main()
    finally:
        save_readline_history()
        bg_jobs.stop_all()
        close_mcp_servers()
