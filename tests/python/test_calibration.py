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


class WorkloadFitTests(unittest.TestCase):
    def test_label_mapping_is_strict(self):
        from jev.calibration import label_to_target_index
        from jev.schema import Question

        choice = Question(type="choice", instructions="pick", criteria={"a": None, "b": None})
        noul = Question(type="noul", instructions="is it?")
        score = Question(type="score", instructions="rate", criteria=["low", "mid", "high"])
        self.assertEqual(label_to_target_index(choice, "b"), 1)
        self.assertEqual(label_to_target_index(noul, True), 1)
        self.assertEqual(label_to_target_index(noul, False), 0)
        self.assertEqual(label_to_target_index(score, 2), 2)
        for question, label in (
            (choice, "c"),
            (choice, 0),
            (noul, 1),
            (noul, "yes"),
            (score, True),
            (score, 3),
            (score, -1),
            (score, "2"),
        ):
            with self.assertRaises(ValueError):
                label_to_target_index(question, label)

    def test_build_workload_fits_verdicts_and_exclusions(self):
        from jev.calibration import build_workload_fits

        # 70% correct at 0.97 confidence: optimal temperature ~ 4, finite
        overconfident = [
            {"probabilities": [0.97, 0.03], "target": [1.0, 0.0]} if i < 21
            else {"probabilities": [0.97, 0.03], "target": [0.0, 1.0]}
            for i in range(30)
        ]
        fits, diagnostics = build_workload_fits({"noul": overconfident}, 25)
        self.assertIn("noul", fits)
        self.assertGreater(fits["noul"]["temperature"], 1.25)
        self.assertEqual(diagnostics["noul"]["verdict"], "apply")
        self.assertLess(diagnostics["noul"]["ece_after"], diagnostics["noul"]["ece_before"])

        # a sample below the floor is diagnosed but never fitted
        with self.assertRaises(ValueError):
            build_workload_fits({"noul": overconfident[:10]}, 25)
        fits, diagnostics = build_workload_fits(
            {"noul": overconfident, "score": overconfident[:5]}, 25
        )
        self.assertNotIn("score", fits)
        self.assertEqual(diagnostics["score"]["verdict"], "insufficient_examples")

        # a degenerate all-correct sample drives the fit to a bound: excluded
        degenerate = [
            {"probabilities": [0.9, 0.1], "target": [1.0, 0.0]} for _ in range(30)
        ]
        with self.assertRaises(ValueError):
            build_workload_fits({"noul": degenerate}, 25)

    def test_top1_ece_orders_calibration_quality(self):
        from jev.calibration import top1_ece

        honest = [
            {"probabilities": [0.8, 0.2], "target": [1.0, 0.0]} if i < 8
            else {"probabilities": [0.8, 0.2], "target": [0.0, 1.0]}
            for i in range(10)
        ]
        overconfident = [
            {"probabilities": [0.99, 0.01], "target": [0.0, 1.0]} for _ in range(10)
        ]
        self.assertLess(top1_ece(honest, 1.0), 0.05)
        self.assertGreater(top1_ece(overconfident, 1.0), 0.9)
