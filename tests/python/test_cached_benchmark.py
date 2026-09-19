import math
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from jev.data import Example
from jev.schema import Question
from scripts.benchmark_cached_teacher import (
    NoRedirect,
    document_text,
    fidelity,
    local_probabilities,
    metrics,
    overlap_keys,
    select_examples,
)


def example(identity, source="cached:imdb", primitive="noul", state=None):
    criteria = None if primitive == "noul" else (["low", "high"] if primitive == "score" else {"a": "A", "b": "B"})
    return Example(
        id=identity, group_id=identity, source=source,
        state=state if state is not None else f"Document {identity}",
        question=Question(primitive, "Judge this", criteria), target=[0.2, 0.8],
        target_origin="teacher-soft", provenance={"gold_label": 1},
    )


class CachedBenchmarkTests(unittest.TestCase):
    def test_selection_is_deterministic_and_not_prediction_based(self):
        population = [example(f"{source}:{i}", source, primitive)
                      for source, primitive in [("cached:ag_news", "choice"), ("cached:dbpedia", "choice"),
                                                ("cached:imdb", "noul"), ("cached:yelp_stars", "score")]
                      for i in range(4)]
        first, count = select_examples(population, set(), 2)
        changed = [replace(e, target=[0.9, 0.1], original_target=[0.9, 0.1],
                           provenance={"gold_label": 0}) for e in reversed(population)]
        second, _ = select_examples(changed, set(), 2)
        self.assertEqual([e.id for e in first], [e.id for e in second])
        self.assertEqual(count, 0)
        self.assertEqual([e.question.type for e in first], ["choice", "choice", "noul", "noul", "score", "score"])
        filtered, count = select_examples(population, overlap_keys(first[0]), 2)
        self.assertEqual(count, 1)
        self.assertNotIn(first[0].id, [e.id for e in filtered])
        for invalid in (0, 1, 3, -2):
            with self.assertRaises(ValueError):
                select_examples(population, set(), invalid)
        with self.assertRaises(ValueError):
            select_examples(population, set(), 20)

    def test_known_wrappers_and_truncated_overlap(self):
        text = "x" * 1500 + " omitted tail"
        states = [text, {"document": text}, {"ticket": {"channel": "web", "body": text}},
                  [{"role": "user", "content": text}]]
        teacher = example("teacher", state=text[:1500])
        for state in states:
            self.assertEqual(document_text(state), text)
            self.assertTrue(overlap_keys(teacher) & overlap_keys(example("train", state=state)))
        self.assertIsNone(document_text({"document": text, "other": "different context"}))
        self.assertIsNone(document_text([{"role": "user", "content": text}, {"role": "user", "content": "other"}]))

    def test_local_probability_and_answer_validation(self):
        noul = Question("noul", "Positive?")
        self.assertEqual(local_probabilities(noul, {"type": "noul", "noul": 0.25}), [0.75, 0.25])
        for value in (True, float("nan"), 1.1):
            with self.assertRaises(ValueError):
                local_probabilities(noul, {"type": "noul", "noul": value})
        choice = Question("choice", "Pick", {"a": "A", "b": "B"})
        with self.assertRaises(ValueError):
            local_probabilities(choice, {"type": "choice", "choice": "b", "probabilities": {"a": 0.8, "b": 0.2}})
        with self.assertRaises(ValueError):
            local_probabilities(choice, {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.19}})
        score = Question("score", "Rate", ["low", "high"])
        answer = {"type": "score", "score": 0.3, "legend": {"0": "low", "1": "high"}, "probabilities": {"0": 0.7, "1": 0.3}}
        self.assertEqual(local_probabilities(score, answer), [0.7, 0.3])
        with self.assertRaises(ValueError):
            local_probabilities(score, {**answer, "score": 0.7})

    def test_metrics_keep_outcomes_and_teacher_agreement_separate(self):
        rows = [{"primitive": "noul", "gold_label": 0,
                 "local_probabilities": [1.0, 0.0], "teacher_probabilities": [0.0, 1.0]}]
        self.assertEqual(metrics(rows, "local_probabilities")["accuracy"], 1)
        self.assertEqual(metrics(rows, "teacher_probabilities")["zero_gold_probability_count"], 1)
        comparison = fidelity(rows)
        self.assertEqual(comparison["argmax_agreement"], 0)
        self.assertEqual(comparison["mean_total_variation"], 1)
        self.assertAlmostEqual(comparison["mean_js_nats"], math.log(2))
        self.assertEqual(comparison["noul_mean_absolute_probability_difference"], 1)

    def test_redirects_and_external_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://api.typesafe.ai/")
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(root / "scripts/benchmark_cached_teacher.py"),
             "--base-url", "https://api.typesafe.ai", "--out-dir", "unused-benchmark-output"],
            text=True, capture_output=True, cwd=root,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("only a loopback", result.stderr)


if __name__ == "__main__":
    unittest.main()
