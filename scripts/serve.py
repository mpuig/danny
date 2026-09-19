"""Bounded local MLX service: POST /v1/systemone, GET /v1/models, /health, /metrics.

A single dedicated thread loads/owns the model; bounded HTTP handlers enqueue work.
No authentication/TLS: stay on loopback or explicitly accept remote exposure risk.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
from dataclasses import fields
from pathlib import Path

from jev.confidence import SCHEMES
from jev.engine import SystemOneEngine
from jev.limits import EngineLimits
from jev.rendering import RENDERER_VERSIONS, READOUT_VERSIONS
from jev.serving import BoundedHTTPServer, InferenceWorker, make_handler, flatten_state  # noqa: F401 (compatibility exports)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--renderer", choices=RENDERER_VERSIONS)
    ap.add_argument("--readout", choices=READOUT_VERSIONS)
    ap.add_argument("--temperature")
    ap.add_argument("--confidence", choices=SCHEMES)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="content-free contextual correction, separate from temperature fitting",
    )
    ap.add_argument(
        "--precision", choices=["native", "float16", "float32"], default="native"
    )
    ap.add_argument(
        "--execution-mode", choices=["independent", "shared"], default="independent"
    )
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument(
        "--mlx-cache-limit-mb",
        type=int,
        default=512,
        help="MLX free-buffer reclamation threshold; not a total RAM limit",
    )
    ap.add_argument("--queue-capacity", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=30)
    ap.add_argument("--io-timeout", type=float, default=10)
    ap.add_argument("--max-body-bytes", type=int, default=262144)
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument(
        "--ready-file", type=Path, help="write startup metadata to a fresh file"
    )
    for field in fields(EngineLimits):
        ap.add_argument(
            "--" + field.name.replace("_", "-"), type=int, default=field.default
        )
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not args.allow_remote:
        ap.error(
            "remote binding requires --allow-remote; this server has no authentication or TLS"
        )
    for name in (
        "max_batch_size",
        "queue_capacity",
        "max_body_bytes",
        "max_connections",
    ):
        if getattr(args, name) < 1:
            ap.error(f"{name} must be positive")
    for name in ("request_timeout", "io_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            ap.error(f"{name} must be finite and positive")
    if args.mlx_cache_limit_mb < 0 or not 0 <= args.port <= 65535:
        ap.error("invalid allocator cache limit or port")
    # Background shells can pass SIGINT=SIG_IGN through subprocesses. Install
    # explicit handlers instead of silently ignoring benchmark/operator shutdown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    limits = EngineLimits(
        **{field.name: getattr(args, field.name) for field in fields(EngineLimits)}
    )

    def factory():
        return SystemOneEngine(
            args.model,
            adapter_path=args.adapter,
            contextual_calibration=args.calibrate,
            renderer_version=args.renderer,
            readout_version=args.readout,
            precision=args.precision,
            execution_mode=args.execution_mode,
            max_batch_size=args.max_batch_size,
            limits=limits,
            temperature_path=args.temperature,
            confidence_scheme=args.confidence,
            mlx_cache_limit_bytes=args.mlx_cache_limit_mb * 1024 * 1024,
        )

    worker = InferenceWorker(
        factory, capacity=args.queue_capacity, timeout=args.request_timeout
    )
    server = None
    try:
        server = BoundedHTTPServer(
            (args.host, args.port),
            make_handler(
                worker, max_body_bytes=args.max_body_bytes, io_timeout=args.io_timeout
            ),
            max_connections=args.max_connections,
        )
        ready = {
            "ready": True,
            "host": server.server_address[0],
            "port": server.server_address[1],
            "model": worker.describe(),
        }
        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            with args.ready_file.open("x") as handle:
                json.dump(ready, handle)
        print(json.dumps(ready), flush=True)
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
