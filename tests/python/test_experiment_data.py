import unittest
from pathlib import Path

from jev.recast import AG_NEWS
from jev.rendering import LEGACY_V0, render
from scripts.build_rubric_holdout import build
from scripts.prepare_teacher_experiment import recover, with_target


class ExperimentDataTests(unittest.TestCase):
    def test_teacher_recovery_is_exact_and_does_not_invent_version(self):
        q = AG_NEWS.question(1)
        prompt, labels = render("A new space telescope was launched.", q, LEGACY_V0)
        target = [0.0, 0.1, 0.1, 0.8]
        row = {
            "prompt": prompt,
            "labels": labels,
            "target": target,
            "task": "ag_news",
            "gold_label": 3,
            "jev": {"probabilities": dict(zip(q.answer_keys, target))},
        }
        e = recover([row], "fixture-sha")[0]
        self.assertEqual(e.question, q)
        self.assertIsNone(e.provenance["teacher_resolved_model"])
        self.assertEqual(with_target(e, "gold").target, [0, 0, 0, 1])
        self.assertEqual(with_target(e, "hard").target, [0, 0, 0, 1])
        self.assertEqual(with_target(e, "soft").target, target)
        self.assertAlmostEqual(with_target(e, "mixed").target[3], 0.9)
        row["prompt"] = prompt.replace("Options:", "CHANGED:")
        with self.assertRaises(ValueError):
            recover([row], "fixture-sha")

    def test_rubric_holdout_covers_all_primitives_and_is_marked_for_review(self):
        rows = build(
            Path(__file__).resolve().parents[1] / "fixtures/heldout_rubrics.json"
        )
        self.assertEqual(len(rows), 42)
        self.assertEqual({e.question.type for e in rows}, {"noul", "choice", "score"})
        self.assertTrue(
            all(
                e.provenance["review_status"] == "requires independent review"
                for e in rows
            )
        )
        self.assertEqual(len({e.id for e in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
