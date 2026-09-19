import http.client
import importlib.util
import json
import threading
import unittest
from types import SimpleNamespace
from http.server import HTTPServer

from jev.rendering import LEGACY_V0, STRUCTURED_V1

HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    from jev.engine import SystemOneEngine
    from scripts.serve import make_handler


@unittest.skipUnless(HAS_MLX, "MLX is not installed")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(SystemOneEngine)
        self.engine.renderer_version = STRUCTURED_V1
        self.engine.model_name = "local-test-model"
        self.engine.contextual_calibration = False
        self.engine._input_tokens = 0
        self.engine.tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))))
        self.prompts = []
        def score(items):
            self.prompts.extend(prompt for prompt, _ in items)
            return [[1 / len(labels)] * len(labels) for _, labels in items]
        self.engine._score_batch = score
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, body):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        try:
            connection.request("POST", "/v1/systemone", body=body,
                               headers={"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_structured_states_do_not_collapse(self):
        for state in [{"a": "x\nb: y"}, {"a": "x", "b": "y"}]:
            status, response = self.request(json.dumps({
                "model": "jev-latest", "state": state,
                "questions": {"q": {"type": "noul", "instructions": {"question": "Is it true?"}}},
            }))
            self.assertEqual(status, 200)
            self.assertEqual(response["model"], "local-test-model")
        self.assertNotEqual(self.prompts[0], self.prompts[1])

    def test_validation_uses_422(self):
        for body in ["{}", '[]', '{"state":NaN}',
                     '{"state":"a","state":"b","questions":{}}',
                     '{"state":null,"questions":{"q":{"type":"noul","instructions":"?"}}}']:
            with self.subTest(body=body):
                status, response = self.request(body)
                self.assertEqual(status, 422)
                self.assertIn("message", response["error"])

    def test_legacy_server_rendering_is_retained(self):
        self.engine.renderer_version = LEGACY_V0
        status, _ = self.request(json.dumps({"state": {"document": "hello"},
                                           "questions": {"q": {"type": "noul", "instructions": "?"}}}))
        self.assertEqual(status, 200)
        self.assertIn("State:\ndocument: hello\n\nQuestion:", self.prompts[0])


if __name__ == "__main__":
    unittest.main()
