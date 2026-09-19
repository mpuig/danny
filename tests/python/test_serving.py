import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
import threading
import time
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from jev.limits import InferenceDeadlineExceeded, RequestValidationError
from jev.serving import (
    BoundedHTTPServer,
    InferenceWorker,
    ServiceBusy,
    make_handler,
    _DeadlineReader,
)


class FakeEngine:
    renderer_version = "structured-v1"

    def __init__(self, seen, entered=None, release=None):
        self.seen = seen
        self.entered = entered
        self.release = release
        self.owner = threading.get_ident()

    def describe(self):
        return {"id": "fixture-model", "object": "model"}

    def respond(self, request):
        if threading.get_ident() != self.owner:
            raise RuntimeError("model used from the wrong thread")
        state = request["state"]
        self.seen.append(state)
        if state == "hold":
            self.entered.set()
            if not self.release.wait(2):
                raise RuntimeError("test barrier timed out")
        if state == "fail":
            raise RuntimeError("do not expose this internal detail")
        if state == "invalid":
            raise RequestValidationError("invalid fixture request")
        if state == "slow":
            time.sleep(0.1)
        return {"model": "fixture-model", "echo": state}


class WorkerTests(unittest.TestCase):
    def test_thread_ownership_saturation_and_expired_queue(self):
        seen = []
        entered = threading.Event()
        release = threading.Event()
        worker = InferenceWorker(
            lambda: FakeEngine(seen, entered, release), capacity=1, timeout=1
        )
        try:
            first = worker.submit({"state": "hold"})
            self.assertTrue(entered.wait(1))
            second = worker.submit({"state": "expired"}, timeout=0.01)
            with self.assertRaises(ServiceBusy):
                worker.submit({"state": "too many"})
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result(1)["echo"], "hold")
            with self.assertRaises(InferenceDeadlineExceeded):
                second.result(1)
            self.assertEqual(seen, ["hold"])
            self.assertEqual(worker.stats()["rejected"], 1)
            self.assertEqual(worker.stats()["expired"], 1)
        finally:
            release.set()
            worker.close()
        with self.assertRaises(ServiceBusy):
            worker.submit({"state": "after close"})

    def test_requests_are_copied_and_worker_recovers_from_errors(self):
        seen = []
        entered = threading.Event()
        release = threading.Event()
        worker = InferenceWorker(
            lambda: FakeEngine(seen, entered, release), capacity=1, timeout=1
        )
        try:
            first = worker.submit({"state": "hold"})
            self.assertTrue(entered.wait(1))
            request = {"state": {"nested": ["original"]}}
            future = worker.submit(request)
            request["state"]["nested"][0] = "mutated"
            release.set()
            first.result(1)
            self.assertEqual(future.result(1)["echo"], {"nested": ["original"]})
            with self.assertRaises(RuntimeError):
                worker.infer({"state": "fail"})
            self.assertEqual(worker.infer({"state": "healthy"})["echo"], "healthy")
        finally:
            release.set()
            worker.close()

    def test_pending_cancellation_skips_model_work(self):
        seen = []
        entered = threading.Event()
        release = threading.Event()
        worker = InferenceWorker(
            lambda: FakeEngine(seen, entered, release), capacity=1, timeout=1
        )
        try:
            first = worker.submit({"state": "hold"})
            self.assertTrue(entered.wait(1))
            pending = worker.submit({"state": "cancelled"})
            self.assertTrue(pending.cancel())
            release.set()
            first.result(1)
            deadline = time.monotonic() + 1
            while worker.stats()["expired"] < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(seen, ["hold"])
        finally:
            release.set()
            worker.close()

    def test_invalid_budgets_do_not_create_unbounded_workers(self):
        for kwargs in [
            {"capacity": 0},
            {"capacity": -1},
            {"timeout": float("nan")},
            {"timeout": float("inf")},
        ]:
            with self.assertRaises(ValueError):
                InferenceWorker(lambda: FakeEngine([]), **kwargs)


class BoundedHTTPTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.worker = InferenceWorker(
            lambda: FakeEngine(self.seen, self.entered, self.release),
            capacity=2,
            timeout=0.05,
        )
        self.server = BoundedHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.worker, max_body_bytes=64, io_timeout=0.1),
            max_connections=4,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)
        self.worker.close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return (
                response.status,
                json.loads(response.read()),
                dict(response.getheaders()),
            )
        finally:
            connection.close()

    def post(self, body, headers=None):
        return self.request(
            "POST",
            "/v1/systemone",
            body,
            {"Content-Type": "application/json", **(headers or {})},
        )

    def test_discovery_and_success(self):
        status, data, _ = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["data"][0]["id"], "fixture-model")
        status, data, headers = self.post('{"state":"ok"}')
        self.assertEqual(status, 200)
        self.assertEqual(data["echo"], "ok")
        self.assertEqual(headers["Connection"], "close")

    def test_rejection_and_timeout_statuses(self):
        self.assertEqual(self.post('{"state":"' + "x" * 100 + '"}')[0], 413)
        self.assertEqual(self.post('{"state":NaN}')[0], 422)
        self.assertEqual(self.post(r'{"state":"\ud800"}')[0], 422)
        self.assertEqual(self.post(r'{"state":{"\ud800":"x"}}')[0], 422)
        self.assertEqual(self.post('{"state":"invalid"}')[0], 422)
        self.assertEqual(self.post('{"state":"slow"}')[0], 504)
        time.sleep(0.08)
        status, body, _ = self.post('{"state":"fail"}')
        self.assertEqual(status, 500)
        self.assertNotIn("internal detail", json.dumps(body))

    def test_queue_saturation_returns_retryable_503(self):
        self.worker.timeout = 1
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(self.post, '{"state":"hold"}')
            self.assertTrue(self.entered.wait(1))
            pending = [pool.submit(self.post, '{"state":"queued"}') for _ in range(2)]
            deadline = time.monotonic() + 1
            while (
                self.worker.stats()["queue_depth"] < 2 and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            status, _, headers = self.post('{"state":"overflow"}')
            self.assertEqual(status, 503)
            self.assertEqual(headers["Retry-After"], "1")
            self.release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertTrue(all(future.result()[0] == 200 for future in pending))

    def test_connection_limit_rejects_before_starting_another_handler(self):
        accepted = threading.Event()
        base = make_handler(self.worker, io_timeout=2)

        class Handler(base):
            def setup(self):
                super().setup()
                accepted.set()

        server = BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=1)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        first = socket.create_connection(server.server_address, timeout=2)
        try:
            self.assertTrue(accepted.wait(1))
            # The first connection holds its handler while awaiting headers.
            # Capacity rejection requires no second handler or parsed request.
            with socket.create_connection(server.server_address, timeout=2) as second:
                self.assertTrue(second.recv(4096).startswith(b"HTTP/1.1 503"))
            self.assertEqual(self.worker.stats()["accepted"], 0)
        finally:
            first.close()
            server.shutdown()
            server.server_close()
            thread.join(1)

    def test_total_read_deadline_cannot_be_reset_by_trickling_bytes(self):
        clock = iter([0.0, 0.04, 0.08, 0.12])
        reader = _DeadlineReader(
            SimpleNamespace(read1=lambda n: b"x"),
            SimpleNamespace(settimeout=lambda n: None),
            0.1,
            clock=lambda: next(clock),
        )
        with self.assertRaises(TimeoutError):
            reader.read(4)

    def test_body_read_deadline_and_bad_framing(self):
        self.assertEqual(self.post("", {"Content-Length": "3"})[0], 408)
        self.assertEqual(self.post("", {"Content-Length": "-1"})[0], 400)
        self.assertEqual(self.post("", {"Transfer-Encoding": "chunked"})[0], 400)
        self.assertEqual(self.post("{}", {"Content-Type": "text/plain"})[0], 415)


@unittest.skipUnless(
    os.environ.get("JEV_TEST_MODEL"),
    "set JEV_TEST_MODEL for real server lifecycle test",
)
class ModelServerLifecycleTests(unittest.TestCase):
    def test_background_inherited_sigint_is_replaced_and_shutdown_is_clean(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready.json"
            args = [
                str(root / "scripts/serve.py"),
                "--model",
                os.environ["JEV_TEST_MODEL"],
                "--port",
                "0",
                "--ready-file",
                str(ready),
            ]
            code = (
                "import signal,runpy,sys;signal.signal(signal.SIGINT,signal.SIG_IGN);"
                f'sys.argv={args!r};runpy.run_path(sys.argv[0],run_name="__main__")'
            )
            with (Path(tmp) / "server.log").open("w") as log:
                child = subprocess.Popen(
                    [sys.executable, "-c", code],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            try:
                deadline = time.monotonic() + 120
                info = None
                while time.monotonic() < deadline and child.poll() is None:
                    if ready.exists():
                        try:
                            info = json.loads(ready.read_text())
                            break
                        except json.JSONDecodeError:
                            pass
                    time.sleep(0.05)
                self.assertIsNotNone(info, (Path(tmp) / "server.log").read_text())
                connection = http.client.HTTPConnection(
                    "127.0.0.1", info["port"], timeout=30
                )
                connection.request(
                    "POST",
                    "/v1/systemone",
                    body=json.dumps(
                        {
                            "state": "Hello",
                            "questions": {
                                "q": {
                                    "type": "noul",
                                    "instructions": "Is this a greeting?",
                                }
                            },
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200, response.read())
                connection.close()
                child.send_signal(signal.SIGINT)
                self.assertEqual(
                    child.wait(timeout=15), 0, (Path(tmp) / "server.log").read_text()
                )
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    unittest.main()
