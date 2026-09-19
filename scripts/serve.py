"""Jev-shaped development endpoint: POST /v1/systemone.

Use a base-URL override (TYPESAFE_BASE_URL / baseURL) for local SDK smoke tests.
This is a tested subset, not complete Jev API compatibility or a hardened service.

    uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M \
        --adapter adapters/smollm2-135m --calibrate --port 8399
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from jev.engine import SystemOneEngine
from jev.rendering import LEGACY_V0, RENDERER_VERSIONS, READOUT_VERSIONS
from jev.serialization import loads, validate_state


def flatten_state(state) -> str:
    """Historical v0 HTTP conversion only; structured-v1 must not use this."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


def make_handler(engine: SystemOneEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # Used by the SDK integration tests.

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                request = loads(body)
                if isinstance(request, dict) and "state" in request:
                    validate_state(request["state"])
                # Preserve the historical server serialization ONLY for old
                # adapters. structured-v1 receives the original JSON unchanged.
                if engine.renderer_version == LEGACY_V0 and isinstance(request, dict) and "state" in request:
                    request["state"] = flatten_state(request["state"])
                response = engine.respond(request)
                self._send(200, response)
            except (KeyError, TypeError, ValueError) as e:
                self._send(422, {"error": {"message": str(e)}})

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
    ap.add_argument("--renderer", choices=RENDERER_VERSIONS, default=None)
    ap.add_argument("--readout", choices=READOUT_VERSIONS)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--precision", choices=["native", "float16", "float32"], default="native")
    ap.add_argument("--execution-mode", choices=["independent", "shared"], default="independent")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    args = ap.parse_args()

    engine = SystemOneEngine(
        args.model, contextual_calibration=args.calibrate, adapter_path=args.adapter,
        renderer_version=args.renderer, precision=args.precision, execution_mode=args.execution_mode,
        readout_version=args.readout,
    )
    # Retain the tested single-thread path until a controlled worker is added.
    server = HTTPServer((args.host, args.port), make_handler(engine))
    print(f"jev-shaped development server on http://{args.host}:{args.port}/v1/systemone")
    server.serve_forever()


if __name__ == "__main__":
    main()
