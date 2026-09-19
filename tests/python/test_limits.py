import importlib.util
import time
import unittest
from types import SimpleNamespace

from jev.limits import EngineLimits, RequestLimitError, InferenceDeadlineExceeded
from jev.schema import Question

HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    from jev.engine import SystemOneEngine


@unittest.skipUnless(HAS_MLX, "MLX is not installed")
class EngineLimitTests(unittest.TestCase):
    def engine(self, **limits):
        engine = object.__new__(SystemOneEngine)
        engine.model_name = "fixture"
        engine.renderer_version = "structured-v1"
        engine.readout_version = "letters-v1"
        engine.limits = EngineLimits(**limits)
        engine.tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))))
        engine.contextual_calibration = False
        engine._prior_cache = {}
        engine._token_cache = {}
        engine._score_batch = lambda items: [[0.5, 0.5] for _ in items]
        return engine

    def test_limits_fail_before_forward(self):
        q = Question("noul", "?")
        for engine, state, questions in [
            (self.engine(max_state_tokens=3), "long state", {"q": q}),
            (self.engine(max_questions=1), "s", {"q": q, "r": q}),
            (self.engine(max_prompt_tokens=20), "s", {"q": q}),
            (self.engine(max_request_tokens=20), "s", {"q": q}),
            (self.engine(max_views=1), "s", {"q": q, "r": q}),
            (self.engine(max_options=1), "s", {"q": q}),
        ]:
            engine._score_batch = lambda items: self.fail(
                "model must not run for rejected input"
            )
            with self.assertRaises(RequestLimitError):
                engine.ask(state, questions)

    def test_caches_are_bounded_and_clearable_by_zero_capacity(self):
        engine = self.engine(max_token_cache_entries=2, max_prior_cache_entries=1)
        for i in range(10):
            engine._encode(str(i))
        self.assertEqual(len(engine._token_cache), 2)
        engine._raw_distributions = lambda state, questions: {"q": [0.4, 0.6]}
        for i in range(10):
            engine._prior(Question("noul", str(i)))
        self.assertEqual(len(engine._prior_cache), 1)
        disabled = self.engine(max_token_cache_entries=0, max_prior_cache_entries=0)
        disabled._encode("s")
        disabled._raw_distributions = lambda state, questions: {"q": [0.4, 0.6]}
        disabled._prior(Question("noul", "?"))
        self.assertEqual(disabled._token_cache, {})
        self.assertEqual(disabled._prior_cache, {})

    def test_cold_contextual_priors_share_the_request_budget(self):
        engine = self.engine(max_views=3)
        engine.contextual_calibration = True
        calls = []
        engine._score_batch = lambda items: (
            calls.extend(items) or [[0.5, 0.5] for _ in items]
        )
        with self.assertRaises(RequestLimitError):
            engine.ask("s", {"q": Question("noul", "?")})
        self.assertEqual(len(calls), 3)
        self.assertFalse(hasattr(engine, "_request_views"))
        engine.contextual_calibration = False
        self.assertIn("q", engine.ask("s", {"q": Question("noul", "?")}))

    def test_cooperative_deadline(self):
        engine = self.engine()
        engine.request_deadline = time.monotonic() - 1
        with self.assertRaises(InferenceDeadlineExceeded):
            engine.ask("s", {"q": Question("noul", "?")})

    def test_invalid_configuration(self):
        for kwargs in [
            {"max_views": 0},
            {"max_options": 256},
            {"max_questions": True},
            {"max_prior_cache_entries": -1},
        ]:
            with self.assertRaises(ValueError):
                EngineLimits(**kwargs)


if __name__ == "__main__":
    unittest.main()
