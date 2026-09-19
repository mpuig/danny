import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from jev.data import Example, sha256_file
from jev.schema import Question
from scripts import refresh_benchmark_jev as live


def frozen_run(root):
    root.mkdir()
    cases, rows = [], []
    for i in range(2):
        example = Example(
            id=f"e{i}", group_id=f"g{i}", source="test", state=f"Good review {i}",
            question=Question("noul", "Positive?"), target=[0.1, 0.9],
            target_origin="teacher-soft", provenance={"gold_label": 1},
        )
        cases.append({"case": f"B{i}", "example": example.to_dict(), "request": {
            "model": "jev-latest", "state": example.state,
            "questions": {"q": asdict(example.question)},
        }})
        rows.append({"case": f"B{i}", "id": example.id, "source": "test", "primitive": "noul",
                     "answer_keys": ["no", "yes"], "gold_label": 1,
                     "local_probabilities": [0.2, 0.8], "teacher_probabilities": [0.1, 0.9],
                     "response": {"model": "local-test"}})
    (root / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (root / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "manifest.json").write_text(json.dumps({
        "cases_sha256": sha256_file(root / "cases.jsonl"), "server": {"id": "local-test"},
    }))
    (root / "report.json").write_text("{}")


class LiveBenchmarkTests(unittest.TestCase):
    def test_version_resolution_rejects_aliases_and_changes(self):
        self.assertEqual(live.resolved_version("jev-1.13.0"), "jev-1.13.0")
        self.assertEqual(live.resolved_version("jev-1.13.0", "jev-1.13.0"), "jev-1.13.0")
        for value in (None, "jev-latest", "jev-preview", "other", "jev-1.14.0"):
            with self.assertRaises(ValueError):
                live.resolved_version(value, "jev-1.13.0")

    def test_rounded_teacher_probabilities_and_label_order(self):
        question = Question("choice", "Which?", {"a": "A", "b": "B"})
        probabilities = live.teacher_probabilities(question, {
            "type": "choice", "probabilities": {"b": 0.33, "a": 0.66},
        })
        self.assertAlmostEqual(probabilities[0], 2 / 3)
        self.assertAlmostEqual(probabilities[1], 1 / 3)
        with self.assertRaises(ValueError):
            live.teacher_probabilities(question, {"type": "choice", "probabilities": {"a": 1}})
        for value in (True, float("nan"), -0.1):
            with self.assertRaises(ValueError):
                live.teacher_probabilities(Question("noul", "True?"), {"type": "noul", "noul": value})

    def test_budget_and_frozen_input_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            frozen_run(root)
            _, cases, rows = live.load_frozen_run(root, 2)
            self.assertEqual(len(cases), len(rows))
            with self.assertRaises(ValueError):
                live.load_frozen_run(root, 1)
            (root / "cases.jsonl").write_text((root / "cases.jsonl").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                live.load_frozen_run(root, 2)

    def test_confirmation_required_before_loading_credentials(self):
        with patch.object(sys, "argv", ["refresh", "--out-dir", "unused"]), \
                patch.object(live, "api_key") as key, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                live.main()
            self.assertEqual(error.exception.code, 2)
            key.assert_not_called()

    def test_network_failure_is_not_retried_and_no_key_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            frozen_run(source)
            opener = Mock()
            opener.open.side_effect = urllib.error.URLError("offline test")
            argv = ["refresh", "--source-run", str(source), "--out-dir", str(output), "--confirm-live"]
            with patch.object(sys, "argv", argv), patch.object(live, "api_key", return_value="test-secret"), \
                    patch.object(live.urllib.request, "build_opener", return_value=opener):
                with self.assertRaises(urllib.error.URLError):
                    live.main()
            self.assertEqual(opener.open.call_count, 1)
            self.assertEqual(len((output / "requests.jsonl").read_text().splitlines()), 1)
            self.assertFalse((output / "report.json").exists())
            for path in output.iterdir():
                self.assertNotIn("test-secret", path.read_text())


if __name__ == "__main__":
    unittest.main()
