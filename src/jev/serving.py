"""Bounded local HTTP service with one model-owning inference thread.

Deadlines are cooperative between model forwards; an in-flight Metal kernel cannot
be preempted safely. Queued expired jobs are discarded. This is not an authenticated
public Internet service. HTTP connections close after each response.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import socket
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .limits import InferenceDeadlineExceeded, RequestValidationError
from .rendering import LEGACY_V0
from .serialization import loads, validate_state


class ServiceBusy(RuntimeError):
    pass


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def flatten_state(state):
    """Historical HTTP conversion; v1 must preserve the original JSON."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{key}: {value}" for key, value in state.items())
    return str(state)


def respond_http(engine, request):
    if isinstance(request, dict) and "state" in request:
        try:
            validate_state(request["state"])
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from exc
        if engine.renderer_version == LEGACY_V0:
            request = {**request, "state": flatten_state(request["state"])}
    return engine.respond(request)


class _DeadlineReader:
    """Bound total header/body read time, not just idle time between bytes."""

    def __init__(self, reader, connection, deadline, clock=time.monotonic):
        self.reader = reader
        self.connection = connection
        self.deadline = deadline
        self.clock = clock
        self.buffer = bytearray()

    def _fill(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise socket.timeout("HTTP read deadline exceeded")
        self.connection.settimeout(remaining)
        # BufferedReader.read1 performs at most one underlying socket read. A
        # trickling peer cannot keep resetting a full timeout inside read(n).
        data = self.reader.read1(4096)
        self.buffer.extend(data)
        return bool(data)

    def readline(self, limit=65537):
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0 or len(self.buffer) >= limit:
                count = min(newline + 1 if newline >= 0 else limit, limit)
                break
            if not self._fill():
                count = min(len(self.buffer), limit)
                break
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def read(self, count):
        if count < 0:
            raise ValueError("unbounded HTTP reads are disabled")
        chunks = []
        while count:
            if not self.buffer and not self._fill():
                break
            take = min(count, len(self.buffer))
            chunks.append(bytes(self.buffer[:take]))
            del self.buffer[:take]
            count -= take
        return b"".join(chunks)

    def close(self):
        self.reader.close()


@dataclass
class Job:
    request: dict
    deadline: float
    future: Future


class InferenceWorker:
    def __init__(self, factory, *, capacity=8, timeout=30.0, startup_timeout=120.0):
        if (
            type(capacity) is not int
            or capacity < 1
            or not _positive_number(timeout)
            or not _positive_number(startup_timeout)
        ):
            raise ValueError("positive queue capacity and timeouts required")
        self.capacity = capacity
        self.timeout = timeout
        self._queue = queue.Queue(maxsize=capacity)
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error = None
        self._lock = threading.Lock()
        self._admission = threading.Lock()
        self._telemetry = {}
        self._counts = dict(
            accepted=0, rejected=0, completed=0, failed=0, expired=0, cancelled=0, active=0
        )
        self._thread = threading.Thread(
            target=self._run, args=(factory,), name="jev-inference", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            self._closed.set()
            raise TimeoutError("model startup timed out")
        if self._startup_error is not None:
            raise RuntimeError("model initialization failed") from self._startup_error

    def _count(self, **updates):
        with self._lock:
            for key, value in updates.items():
                self._counts[key] += value

    def _run(self, factory):
        try:
            # Loading, model arrays, and every inference execute on this thread.
            engine = factory()
            self._description = engine.describe()
            if hasattr(engine, "runtime_stats"):
                self._telemetry = engine.runtime_stats()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            if self._closed.is_set() and self._queue.empty():
                break
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                if not job.future.set_running_or_notify_cancel():
                    self._count(cancelled=1)  # client cancelled; not a deadline expiry
                    continue
                if time.monotonic() >= job.deadline:
                    self._count(expired=1)
                    job.future.set_exception(
                        InferenceDeadlineExceeded("request expired in queue")
                    )
                    continue
                self._count(active=1)
                engine.request_deadline = job.deadline
                try:
                    result = respond_http(engine, job.request)
                    if time.monotonic() >= job.deadline:
                        raise InferenceDeadlineExceeded("request deadline exceeded")
                    job.future.set_result(result)
                    self._count(completed=1)
                except InferenceDeadlineExceeded as exc:
                    job.future.set_exception(exc)
                    self._count(expired=1)
                except BaseException as exc:
                    job.future.set_exception(exc)
                    self._count(failed=1)
                finally:
                    engine.request_deadline = None
                    if hasattr(engine, "runtime_stats"):
                        try:
                            telemetry = engine.runtime_stats()
                        except Exception as exc:
                            telemetry = {"telemetry_error": type(exc).__name__}
                        with self._lock:
                            self._telemetry = telemetry
                    self._count(active=-1)
            finally:
                self._queue.task_done()

    def submit(self, request, *, timeout=None):
        if timeout is not None and not _positive_number(timeout):
            raise ValueError("timeout must be finite and positive")
        budget = self.timeout if timeout is None else min(timeout, self.timeout)
        future = Future()
        # Callers cannot mutate a queued request after admission.
        job = Job(copy.deepcopy(request), time.monotonic() + budget, future)
        with self._admission:
            if self._closed.is_set():
                raise ServiceBusy("service is stopping")
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                self._count(rejected=1)
                raise ServiceBusy("inference queue is full") from exc
            self._count(accepted=1)
        return future

    def infer(self, request):
        future = self.submit(request)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout as exc:
            future.cancel()  # Pending jobs are skipped; running jobs check deadline.
            raise InferenceDeadlineExceeded("request deadline exceeded") from exc

    def describe(self):
        return copy.deepcopy(self._description)

    def stats(self):
        with self._lock:
            result = {**self._counts, **self._telemetry}
        return {
            **result,
            "queue_depth": self._queue.qsize(),
            "queue_capacity": self.capacity,
            "ready": self._ready.is_set() and not self._closed.is_set(),
            "request_timeout_seconds": self.timeout,
        }

    def close(self, join_timeout=60):
        if not _positive_number(join_timeout):
            raise ValueError("join timeout must be finite and positive")
        with self._admission:
            self._closed.set()
            while True:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                if (
                    job is not None
                    and not job.future.done()
                    and job.future.set_running_or_notify_cancel()
                ):
                    job.future.set_exception(ServiceBusy("service is stopping"))
                self._queue.task_done()
            self._queue.put_nowait(None)
        self._thread.join(join_timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                "inference has not stopped; a running kernel cannot be force-cancelled safely"
            )
        # Remove a wakeup sentinel if the worker observed closure before blocking.
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, handler, *, max_connections=16):
        if type(max_connections) is not int or max_connections < 1:
            raise ValueError("max_connections must be positive")
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(0.2)
                body = b'{"error":{"message":"connection capacity reached"}}'
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 1\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()


def make_handler(backend, *, max_body_bytes=262144, io_timeout=10.0):
    """Direct engine backend remains available for synchronous tests only.

    The CLI always uses InferenceWorker; HTTP threads never execute model code.
    """
    if (
        type(max_body_bytes) is not int
        or max_body_bytes < 1
        or not _positive_number(io_timeout)
    ):
        raise ValueError("positive HTTP limits required")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            self.request.settimeout(io_timeout)
            super().setup()
            self.rfile = _DeadlineReader(
                self.rfile, self.request, time.monotonic() + io_timeout
            )

        def _send(self, code, payload):
            self.close_connection = True
            data = json.dumps(payload, allow_nan=False).encode()
            try:
                # _DeadlineReader may have shrunk the socket timeout to the read
                # deadline's remainder; the response gets its own full budget so
                # a completed inference is never dropped by a slow-reading client.
                self.request.settimeout(io_timeout)
            except OSError:
                pass
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                if code == 503:
                    self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass

        def _error(self, status, message):
            self._send(status, {"error": {"message": message}})

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/v1/models":
                self._send(200, {"object": "list", "data": [backend.describe()]})
            elif path in ("/health", "/metrics"):
                stats = (
                    backend.stats() if hasattr(backend, "stats") else {"ready": True}
                )
                self._send(200 if stats["ready"] else 503, stats)
            else:
                self._error(404, "not found")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._error(404, "not found")
            if self.headers.get("Transfer-Encoding"):
                return self._error(400, "chunked request bodies are not supported")
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                return self._error(411, "Content-Length is required")
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
                or len(lengths[0]) > 12
            ):
                return self._error(400, "invalid Content-Length")
            length = int(lengths[0])
            if length > max_body_bytes:
                return self._error(413, f"request body exceeds {max_body_bytes} bytes")
            if (
                self.headers.get_content_type() != "application/json"
                or self.headers.get("Content-Encoding")
            ):
                return self._error(415, "uncompressed application/json is required")
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    return self._error(400, "incomplete request body")
                try:
                    request = loads(body)
                except (ValueError, TypeError) as exc:
                    raise RequestValidationError(str(exc)) from exc
                result = (
                    backend.infer(request)
                    if hasattr(backend, "infer")
                    else respond_http(backend, request)
                )
                self._send(200, result)
            except InferenceDeadlineExceeded as exc:
                self._error(504, str(exc))
            except socket.timeout:
                self._error(408, "request body read timed out")
            except RequestValidationError as exc:
                self._error(422, str(exc))
            except ServiceBusy as exc:
                self._error(503, str(exc))
            except Exception:
                self._error(500, "inference failed")

        def log_message(self, fmt, *args):
            # Do not log request bodies, keys, or states.
            pass

    return Handler
