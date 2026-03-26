#!/usr/bin/env python3
"""Codex <-> llama.cpp proxy with full debug logging."""
import json, sys, os, socket, signal, time, http.client
from http.server import HTTPServer, BaseHTTPRequestHandler

UPSTREAM_HOST = "localhost"
UPSTREAM_PORT = int(os.environ.get('UPSTREAM_PORT', '8080'))
PORT = int(os.environ.get('PROXY_PORT', '8001'))
LOG = "/tmp/codex_proxy_debug.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        log(f"POST {self.path} ({len(body)} bytes)")

        try:
            data = json.loads(body)
            # Fix 1: Remove non-function tools
            if 'tools' in data:
                before = len(data['tools'])
                data['tools'] = [t for t in data['tools'] if t.get('type') == 'function']
                if len(data['tools']) < before:
                    log(f"  removed {before - len(data['tools'])} non-function tools")
            # Fix 2: Remove instructions (Qwen template ordering issue)
            if 'instructions' in data:
                log(f"  removed instructions ({len(data['instructions'])} chars)")
                data.pop('instructions')
            # Fix 3: Convert developer role to user
            for item in data.get('input', []):
                if item.get('role') == 'developer':
                    item['role'] = 'user'
                    log("  converted developer -> user")
            # Only keep fields llama.cpp accepts
            clean = {
                "model": data.get("model", "local"),
                "input": data.get("input", []),
                "tools": data.get("tools", []),
                "stream": data.get("stream", True),
                "tool_choice": data.get("tool_choice", "auto"),
            }
            body = json.dumps(clean).encode('utf-8')
            log(f"  fixed body: {len(body)} bytes, {len(clean.get('tools',[]))} tools")
        except Exception as e:
            log(f"  JSON error: {e}")

        # Forward to llama-server
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=600)
            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ('host','transfer-encoding')}
            hdrs['Content-Length'] = str(len(body))
            log(f"  forwarding to {UPSTREAM_HOST}:{UPSTREAM_PORT}")
            conn.request('POST', self.path, body=body, headers=hdrs)
            resp = conn.getresponse()
            log(f"  upstream response: {resp.status} {resp.getheader('Content-Type','?')}")

            if resp.status >= 400:
                err = resp.read()
                log(f"  ERROR: {err[:300]}")
                self.send_response(resp.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err)
                conn.close()
                return

            # Forward response headers
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.flush()
            log("  headers sent, streaming...")

            # Stream response byte by byte
            total = 0
            start = time.time()
            while True:
                byte = resp.read(1)
                if not byte:
                    break
                self.wfile.write(byte)
                total += 1
                if byte == b"\n":
                    self.wfile.flush()
            self.wfile.flush()
            elapsed = time.time() - start
            log(f"  streamed {total} bytes in {elapsed:.1f}s")
            conn.close()

        except BrokenPipeError:
            log("  client disconnected (broken pipe)")
        except Exception as e:
            log(f"  EXCEPTION: {type(e).__name__}: {e}")
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error":{"message":str(e)}}).encode())
            except: pass

    def do_GET(self):
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=10)
            conn.request('GET', self.path)
            r = conn.getresponse(); b = r.read()
            self.send_response(r.status)
            for k,v in r.getheaders():
                if k.lower() not in ('transfer-encoding','connection'): self.send_header(k,v)
            self.end_headers(); self.wfile.write(b); conn.close()
        except: self.send_response(502); self.end_headers()

    def log_message(self, *a): pass

if __name__ == "__main__":
    # Clear log
    open(LOG, "w").close()
    # Kill existing
    try: s=socket.socket(); s.bind(("127.0.0.1",PORT)); s.close()
    except:
        import subprocess
        subprocess.run(f"lsof -ti :{PORT} | xargs kill 2>/dev/null", shell=True)
        time.sleep(1)
    log(f"Proxy starting: :{PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}")
    print(f"Codex proxy: :{PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}")
    print(f"Logs: {LOG}")
    try: HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt: log("Stopped"); print("\nStopped.")
