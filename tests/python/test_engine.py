import importlib.util
import math
import os
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jev.rendering import STRUCTURED_V1
from jev.schema import Question

HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    import mlx.core as mx
    from mlx.utils import tree_map
    from jev.engine import SystemOneEngine


@unittest.skipUnless(HAS_MLX, "MLX is not installed")
class EngineContractTests(unittest.TestCase):
    def engine(self):
        engine = object.__new__(SystemOneEngine)
        engine.renderer_version = STRUCTURED_V1
        engine.model_name = "actual-backbone"
        engine.contextual_calibration = False
        engine._input_tokens = 0
        engine._label_cache = {}
        engine._prior_cache = {}
        engine.tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))))
        return engine

    def test_question_ids_do_not_enter_prompts_or_model_identity(self):
        engine = self.engine()
        captured = []

        def score(items):
            captured.append(items)
            return [[0.2, 0.8]] * len(items)

        engine._score_batch = score
        q = {"type": "noul", "instructions": {"question": "Is it true?"}}
        response = engine.respond(
            {"model": "jev-latest", "state": {"a": ["b"]}, "questions": {"first": q}}
        )
        other = engine.respond({"state": {"a": ["b"]}, "questions": {"renamed": q}})
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(response["model"], "actual-backbone")
        self.assertEqual(response["answers"]["first"], other["answers"]["renamed"])
        for request in [
            [],
            {},
            {"state": "s", "questions": []},
            {"state": "s", "questions": {}},
            {"state": "s", "questions": {"a": None}},
        ]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                engine.respond(request)

    def test_invalid_model_probabilities_rejected(self):
        engine = self.engine()
        engine.tokenizer = SimpleNamespace(
            encode=lambda label, **kwargs: [0 if label == " A" else 1]
        )
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            engine._probs_from_logits(mx.array([float("nan"), 1]), [" A", " B"])

    def test_temperature_artifact_is_configuration_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temperature.json"
            artifact = {
                "format_version": 1,
                "method": "per-primitive-temperature-v1",
                "prediction_config": {
                    "renderer_version": STRUCTURED_V1,
                    "readout_version": "letters-v1",
                    "precision": "native",
                    "execution_mode": "independent",
                    "contextual_correction": False,
                    "backbone_files": {},
                    "adapter_sha256": None,
                },
                "fits": {"noul": {"temperature": 2.0}},
            }
            path.write_text(json.dumps(artifact))
            with (
                patch(
                    "jev.engine.load",
                    return_value=(
                        SimpleNamespace(eval=lambda: None),
                        SimpleNamespace(encode=lambda text: list(range(len(text)))),
                        {},
                    ),
                ),
                patch("jev.engine.model_identity", return_value={"files": {}}),
            ):
                engine = SystemOneEngine("fixture", temperature_path=str(path))
                engine._score_batch = lambda items: [[0.1, 0.9] for _ in items]
                self.assertAlmostEqual(
                    engine.ask("s", {"q": Question("noul", "?")})["q"].noul, 0.75
                )
                with self.assertRaisesRegex(ValueError, "does not match"):
                    SystemOneEngine(
                        "fixture", temperature_path=str(path), max_batch_size=8
                    )
                artifact["prediction_config"]["backbone_files"] = {"changed": "weights"}
                path.write_text(json.dumps(artifact))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    SystemOneEngine("fixture", temperature_path=str(path))

    def test_unknown_families_keep_wrapper_transforms_and_skip_shared_cache(self):
        engine = self.engine()
        engine.execution_mode = "shared"
        engine.model_config = {"model_type": "unverified"}
        engine.tokenizer = SimpleNamespace(
            encode=lambda text, **kw: (
                [0 if text == " A" else 1]
                if kw
                else list(range(12)) + [1 if text == "a" else 2]
            )
        )

        class Model:
            calls = 0

            def __init__(self):
                self.model = lambda *a, **kw: self.fail_if_bypassed()
                self.lm_head = lambda x: x

            def fail_if_bypassed(self):
                raise AssertionError("wrapper-specific logit transform was bypassed")

            def __call__(self, tokens, cache=None):
                self.calls += 1
                return mx.broadcast_to(mx.array([0.25, -0.25]), (*tokens.shape, 2))

        engine.model = Model()
        with patch(
            "jev.engine.make_prompt_cache",
            side_effect=AssertionError("unsafe cache path"),
        ):
            rows = engine._score_batch([("a", [" A", " B"]), ("b", [" A", " B"])])
        self.assertEqual(engine.model.calls, 2)
        self.assertAlmostEqual(rows[0][0], 1 / (1 + math.exp(-0.5)), places=6)
        self.assertEqual(rows[0], rows[1])

    def test_mutated_cache_is_not_retried(self):
        engine = self.engine()
        engine.execution_mode = "shared"
        engine.model_config = {"model_type": "llama"}
        engine.tokenizer = SimpleNamespace(
            encode=lambda prompt: list(range(12)) + [1 if prompt == "a" else 2],
            eos_token_id=0,
        )

        class Model:
            def __init__(self):
                self.calls = 0
                self.model = self.body
                self.lm_head = lambda x: x

            def __call__(self, tokens, cache):
                self.calls += 1

            def body(self, tokens, cache):
                raise AttributeError("failure AFTER suffix cache advancement")

        engine.model = Model()
        cache = [
            SimpleNamespace(
                keys=mx.zeros((1, 1, 12, 1)), values=mx.zeros((1, 1, 12, 1))
            )
        ]
        with patch("jev.engine.make_prompt_cache", return_value=cache):
            with self.assertRaisesRegex(AttributeError, "AFTER"):
                engine._score_batch([("a", [" A", " B"]), ("b", [" A", " B"])])
        self.assertEqual(
            engine.model.calls, 1, "only prefix call; must not retry the suffix"
        )


@unittest.skipUnless(
    HAS_MLX and os.environ.get("JEV_TEST_MODEL"),
    "set JEV_TEST_MODEL for local model tests",
)
class ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SystemOneEngine(
            os.environ["JEV_TEST_MODEL"], renderer_version=STRUCTURED_V1
        )

    def test_mixed_batch_padding_and_question_invariance(self):
        state = {
            "ticket": {"messages": ["The package arrived damaged. Please refund it."]}
        }
        questions = {
            "refund": Question("noul", "Does `ticket.messages[0]` request a refund?"),
            "team": Question(
                "choice",
                "Which department fits?",
                {
                    "returns": {"covers": ["refunds", "damaged products"]},
                    "sales": "New purchases",
                    "account": None,
                },
            ),
            "tone": Question(
                "score",
                "How frustrated is the customer?",
                ["Calm", "Frustrated", "Very angry"],
            ),
        }
        engine = self.engine
        items = [engine._render(state, q) for q in questions.values()]
        independent = engine._score_batch(items)
        for item, row in zip(items, independent):
            self.assertEqual(row, engine._score_batch([item])[0])
        engine.execution_mode = "shared"
        native = engine._score_batch(items)
        native_reversed = list(reversed(engine._score_batch(list(reversed(items)))))
        for row, reversed_row in zip(native, native_reversed):
            for a, b in zip(row, reversed_row):
                self.assertAlmostEqual(a, b, delta=2e-3)

        # Cache-math oracle: BF16 shape-dependent rounding in this backbone can
        # move probabilities by ~0.03 across full-prefill versus cached paths.
        # Do not hide that behind a loose tolerance or claim native bit parity.
        # FP32 isolates structural/cache errors; native serving drift is a known
        # limitation to benchmark separately before using tight decision gates.
        engine.model.update(
            tree_map(
                lambda x: (
                    x.astype(mx.float32) if mx.issubdtype(x.dtype, mx.floating) else x
                ),
                engine.model.parameters(),
            )
        )
        sequential = [engine._score_batch([item])[0] for item in items]
        batched = engine._score_batch(items)
        reversed_batch = list(reversed(engine._score_batch(list(reversed(items)))))
        for actual, expected, reversed_actual in zip(
            batched, sequential, reversed_batch
        ):
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in actual))
            self.assertAlmostEqual(sum(actual), 1, places=5)
            for a, b, c in zip(actual, expected, reversed_actual):
                self.assertAlmostEqual(a, b, delta=2e-5)
                self.assertAlmostEqual(a, c, delta=2e-5)
        alone = engine.ask(state, {"refund": questions["refund"]})["refund"].noul
        mixed = engine.ask(state, questions)["refund"].noul
        renamed = engine.ask(state, {"unrelated_id": questions["refund"]})[
            "unrelated_id"
        ].noul
        self.assertAlmostEqual(alone, mixed, delta=2e-5)
        self.assertEqual(alone, renamed)


if __name__ == "__main__":
    unittest.main()
