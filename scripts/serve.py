"""Jev-compatible API server: POST /v1/systemone, same wire format as
api.typesafe.ai, so the official Typesafe SDKs work against it via a base-URL
override (TYPESAFE_BASE_URL / baseURL).

    uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M \
        --adapter adapters/smollm2-135m --calibrate --port 8399
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from jev.engine import SystemOneEngine


def flatten_state(state) -> str:
    """The JS SDK allows object states ({document: "..."}); flatten to text."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


def make_handler(engine: SystemOneEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # undici (node fetch) rejects HTTP/1.0

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                request = json.loads(body)
                request["state"] = flatten_state(request.get("state", ""))
                response = engine.respond(request)
                self._send(200, response)
            except (KeyError, TypeError, ValueError) as e:
                self._send(400, {"error": {"message": str(e)}})

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *fmt_args):  # quiet request logging
            print(f"{self.command} {self.path} -> {fmt % fmt_args}")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    args = ap.parse_args()

    engine = SystemOneEngine(
        args.model, contextual_calibration=args.calibrate, adapter_path=args.adapter
    )
    # single-threaded: MLX ops must run on the thread that owns the stream
    server = HTTPServer((args.host, args.port), make_handler(engine))
    print(f"jev-compatible server on http://{args.host}:{args.port}/v1/systemone")
    server.serve_forever()


if __name__ == "__main__":
    main()
