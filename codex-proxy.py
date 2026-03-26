#!/usr/bin/env python3
"""
Proxy that converts Codex 'responses' API tool types to 'function' type
for llama.cpp compatibility. Supports streaming (SSE) passthrough.

Usage: python3 codex-proxy.py  (listens on :8001, forwards to :8000)
"""

import json, sys, http.client
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

UPSTREAM_HOST = "localhost"
UPSTREAM_PORT = 8000
PORT = 8001

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)

            # Convert non-function tools to function type
            if 'tools' in data:
                fixed_tools = []
                for tool in data['tools']:
                    if tool.get('type') != 'function':
                        fixed = {
                            "type": "function",
                            "name": tool.get('name', tool.get('type', 'unknown')),
                            "description": tool.get('description', ''),
                            "parameters": tool.get('parameters', tool.get('input_schema', {"type": "object", "properties": {}})),
                        }
                        if tool.get('type') in ('web_search', 'web_search_preview', 'code_interpreter'):
                            fixed['name'] = tool.get('type')
                            fixed['description'] = f"Built-in {tool.get('type')} tool"
                            fixed['parameters'] = {"type": "object", "properties": {"query": {"type": "string"}}}
                        fixed_tools.append(fixed)
                    else:
                        fixed_tools.append(tool)
                data['tools'] = fixed_tools

            body = json.dumps(data).encode()
        except Exception as e:
            print(f"[proxy] parse error: {e}", file=sys.stderr)

        # Forward with streaming support
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=300)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ('host', 'transfer-encoding')}
            headers['Content-Length'] = str(len(body))
            conn.request('POST', self.path, body=body, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(k, v)
            self.end_headers()

            # Stream the response in chunks
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

            conn.close()
        except Exception as e:
            print(f"[proxy] upstream error: {e}", file=sys.stderr)
            err = json.dumps({"error": {"message": str(e), "type": "proxy_error"}})
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err.encode())
            except Exception:
                pass

    def do_GET(self):
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=10)
            conn.request('GET', self.path)
            resp = conn.getresponse()
            body = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            conn.close()
        except Exception:
            self.send_response(502)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}", file=sys.stderr)

if __name__ == "__main__":
    import signal, socket
    # Kill any existing process on our port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", PORT))
        s.close()
    except OSError:
        print(f"Port {PORT} in use, killing existing process...")
        import subprocess
        pids = subprocess.run(f"lsof -ti :{PORT}", shell=True, capture_output=True, text=True).stdout.strip()
        if pids:
            for pid in pids.split("\n"):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
            import time; time.sleep(1)

    server = HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    print(f"Codex proxy: :{PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}")
    print(f"Set in ~/.codex/config.toml: base_url = \"http://localhost:{PORT}/v1\"")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy stopped.")
        server.server_close()
