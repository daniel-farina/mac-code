#!/usr/bin/env python3
"""
Codex <-> llama.cpp proxy.
Fixes tool type mismatch and properly streams SSE responses.

Usage:
    python3 codex-proxy.py                     # :8001 -> :8080
    UPSTREAM_PORT=8000 python3 codex-proxy.py  # :8001 -> :8000
"""

import json, sys, os, socket, signal, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen

UPSTREAM = f"http://localhost:{os.environ.get('UPSTREAM_PORT', '8080')}"
PORT = int(os.environ.get('PROXY_PORT', '8001'))

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))

        try:
            data = json.loads(body)

            # Fix non-function tool types (Codex sends web_search_preview etc)
            if 'tools' in data:
                fixed = []
                for t in data['tools']:
                    if t.get('type') == 'function':
                        fixed.append(t)
                    else:
                        # Convert to function type
                        name = t.get('name', t.get('type', 'unknown'))
                        fixed.append({
                            "type": "function",
                            "name": name,
                            "description": t.get('description', f'{name} tool'),
                            "parameters": t.get('parameters', t.get('input_schema',
                                {"type": "object", "properties": {}})),
                        })
                        print(f"[proxy] fixed tool type: {t.get('type')} -> function ({name})", file=sys.stderr)
                data['tools'] = fixed

            body = json.dumps(data).encode()
        except Exception as e:
            print(f"[proxy] parse error: {e}", file=sys.stderr)

        # Forward request and stream response byte-by-byte
        try:
            req = Request(
                f"{UPSTREAM}{self.path}",
                data=body,
                headers={k: v for k, v in self.headers.items() if k.lower() != 'host'},
                method='POST',
            )

            with urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                # Forward ALL headers including content-type (critical for SSE)
                for k, v in resp.getheaders():
                    if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                        self.send_header(k, v)
                self.end_headers()

                # Stream byte-by-byte for SSE support
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except Exception as e:
            print(f"[proxy] error: {e}", file=sys.stderr)
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}).encode())
            except Exception:
                pass

    def do_GET(self):
        try:
            req = Request(f"{UPSTREAM}{self.path}")
            with urlopen(req, timeout=10) as resp:
                body = resp.read()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
        except Exception:
            self.send_response(502)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}", file=sys.stderr)


if __name__ == "__main__":
    # Kill existing process on our port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", PORT))
        s.close()
    except OSError:
        print(f"Port {PORT} in use, killing...")
        import subprocess
        pids = subprocess.run(f"lsof -ti :{PORT}", shell=True, capture_output=True, text=True).stdout.strip()
        for pid in pids.split("\n"):
            try: os.kill(int(pid), signal.SIGTERM)
            except: pass
        time.sleep(1)

    print(f"Codex proxy: :{PORT} -> {UPSTREAM}")
    print(f"Config: base_url = \"http://localhost:{PORT}/v1\"")
    try:
        HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
