#!/usr/bin/env node
/**
 * Codex <-> llama.cpp proxy (Node.js)
 * Fixes tool types + message roles for Qwen compatibility.
 * Streams SSE natively.
 *
 * Usage: node codex-proxy.js
 *   UPSTREAM_PORT=8080 PROXY_PORT=8001 node codex-proxy.js
 */

const http = require('http');

const UPSTREAM_PORT = parseInt(process.env.UPSTREAM_PORT || '8080');
const UPSTREAM_HOST = '127.0.0.1';
const PORT = parseInt(process.env.PROXY_PORT || '8001');

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  console.error(`[${ts}] ${msg}`);
}

function fixRequest(data) {
  // Fix 1: Remove non-function tools
  if (data.tools) {
    const before = data.tools.length;
    data.tools = data.tools.filter(t => t.type === 'function');
    const removed = before - data.tools.length;
    if (removed) log(`  removed ${removed} non-function tools`);
  }

  // Fix 2: Remove instructions (Qwen template ordering issue)
  if (data.instructions) {
    log(`  removed instructions (${data.instructions.length} chars)`);
    delete data.instructions;
  }

  // Fix 3: Convert developer role to user
  if (data.input && Array.isArray(data.input)) {
    for (const item of data.input) {
      if (item.role === 'developer') {
        item.role = 'user';
        log('  converted developer -> user');
      }
    }
  }

  // Only keep fields llama.cpp accepts
  return {
    model: data.model || 'local',
    input: data.input || [],
    tools: data.tools || [],
    stream: data.stream !== undefined ? data.stream : true,
    tool_choice: data.tool_choice || 'auto',
  };
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET') {
    // Passthrough GET requests
    const proxy = http.request({
      hostname: UPSTREAM_HOST, port: UPSTREAM_PORT,
      path: req.url, method: 'GET',
    }, (upstream) => {
      res.writeHead(upstream.statusCode, upstream.headers);
      upstream.pipe(res);
    });
    proxy.on('error', () => { res.writeHead(502); res.end(); });
    proxy.end();
    return;
  }

  // POST - collect body
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    log(`POST ${req.url} (${body.length} bytes)`);

    let fixed;
    try {
      const data = JSON.parse(body);
      fixed = JSON.stringify(fixRequest(data));
      log(`  fixed: ${fixed.length} bytes, ${JSON.parse(fixed).tools.length} tools`);
    } catch (e) {
      log(`  JSON error: ${e.message}`);
      fixed = body; // pass through on error
    }

    // Forward to llama-server
    const proxyReq = http.request({
      hostname: UPSTREAM_HOST,
      port: UPSTREAM_PORT,
      path: req.url,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(fixed),
      },
    }, (upstream) => {
      log(`  upstream: ${upstream.statusCode} ${upstream.headers['content-type'] || '?'}`);

      // Forward status + headers
      res.writeHead(upstream.statusCode, upstream.headers);

      // Pipe SSE stream directly - Node handles this natively
      upstream.pipe(res);

      upstream.on('end', () => {
        log('  stream ended');
      });
    });

    proxyReq.on('error', (e) => {
      log(`  upstream error: ${e.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: e.message } }));
    });

    proxyReq.write(fixed);
    proxyReq.end();
  });
});

server.listen(PORT, '127.0.0.1', () => {
  log(`Proxy: :${PORT} -> ${UPSTREAM_HOST}:${UPSTREAM_PORT}`);
  console.log(`Codex proxy: :${PORT} -> ${UPSTREAM_HOST}:${UPSTREAM_PORT}`);
  console.log(`Config: base_url = "http://127.0.0.1:${PORT}/v1"`);
});
