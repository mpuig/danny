import json
import tempfile
import unittest
from pathlib import Path

from jev.calibration import fit_temperature, temperature_scale
from jev.confidence import ADAPTER, ENTROPY, confidence
from jev.data import Example, sha256_file
from jev.schema import Question
from jev.serialization import dumps
from scripts.fit_calibration import fit


class ConfidenceTests(unittest.TestCase):
    def test_pinned_adapter_fixtures(self):
        self.assertEqual(confidence([0.5, 0.5], "choice"), 0)
        self.assertAlmostEqual(confidence([0.8, 0.2], "choice"), 0.6)
        self.assertAlmostEqual(confidence([0.1, 0.8, 0.1], "score"), 0.7)
        self.assertAlmostEqual(confidence([0.5, 0.5, 0], "score"), 0.25)
        self.assertEqual(confidence([0.5, 0, 0.5], "score"), 0)
        self.assertEqual(confidence([0, 0, 0], "score"), 0)
        self.assertEqual(confidence([1], "choice"), 1)
        self.assertNotEqual(
            confidence([0.8, 0.2], "choice", ADAPTER),
            confidence([0.8, 0.2], "choice", ENTROPY),
        )


class TemperatureTests(unittest.TestCase):
    def test_fit_reduces_nll_without_changing_argmax(self):
        rows = [
            {"probabilities": [0.99, 0.01], "target": [1.0, 0.0]},
            {"probabilities": [0.99, 0.01], "target": [0.0, 1.0]},
        ] * 10
        result = fit_temperature(rows)
        self.assertGreater(result["temperature"], 1)
        self.assertLess(result["nll_after"], result["nll_before"])
        self.assertGreater(
            temperature_scale([0.99, 0.01], result["temperature"])[0], 0.5
        )
        self.assertAlmostEqual(temperature_scale([0.1, 0.9], 2)[1], 0.75)
        for t in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                temperature_scale([0.2, 0.8], t)
        with self.assertRaises(ValueError):
            fit_temperature([{"probabilities": [0.2, 0.8], "target": [0.2, 0.8]}])

    def test_only_declared_complete_calibration_predictions_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"files": {}}
            partitions = {}
            for split in ["train", "development", "calibration"]:
                examples = [
                    Example(
                        id=f"{split}/{i}",
                        group_id=f"{split}/{i}",
                        source="fixture",
                        state=f"{split} state {i}",
                        question=Question("noul", "?"),
                        target=[float(i == 0), float(i == 1)],
                        target_origin="gold",
                        provenance={},
                    )
                    for i in range(2)
                ]
                partitions[split] = examples
                p = root / f"{split}.jsonl"
                p.write_text("".join(dumps(e.to_dict()) + "\n" for e in examples))
                manifest["files"][p.name] = {"sha256": sha256_file(p)}
            (root / "manifest.json").write_text(json.dumps(manifest))
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            report = {
                "arguments": {},
                "data_sha256": sha256_file(root / "calibration.jsonl"),
                "renderer_version": "structured-v1",
                "backbone": {"files": {}},
                "adapter_sha256": None,
            }
            report_path = prediction_dir / "report.json"
            report_path.write_text(json.dumps(report))
            rows = [
                {
                    "id": e.id,
                    "target": e.target,
                    "primitive": "noul",
                    "answer_keys": e.question.answer_keys,
                    "probabilities": [0.99, 0.01],
                }
                for e in partitions["calibration"]
            ]
            (prediction_dir / "predictions.jsonl").write_text(
                "".join(dumps(r) + "\n" for r in rows)
            )
            result = fit(
                prediction_dir, root / "manifest.json", root / "temperature.json"
            )
            self.assertIn("noul", result["fits"])
            self.assertEqual(
                result["prediction_config"]["readout_version"], "letters-v1"
            )
            with self.assertRaises(ValueError):
                fit(prediction_dir, root / "manifest.json", root / "temperature.json")
            report["data_sha256"] = sha256_file(root / "development.jsonl")
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "declared calibration"):
                fit(prediction_dir, root / "manifest.json", root / "wrong.json")


if __name__ == "__main__":
    unittest.main()
