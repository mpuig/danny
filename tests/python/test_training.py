import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jev.data import Example
from jev.rendering import STRUCTURED_V1
from jev.schema import Question
from jev.serialization import dumps

HAS_MLX = importlib.util.find_spec("mlx") is not None
ROOT = Path(__file__).resolve().parents[2]
if HAS_MLX:
    from scripts.train_lora import tokenize_rows
    from scripts.eval_dataset import summarize_rows


@unittest.skipUnless(HAS_MLX, "MLX is not installed")
class TrainingTests(unittest.TestCase):
    def test_tokenization_rejects_invalid_labels_before_skipping(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                if kwargs:
                    return [1, 2]
                return list(range(100))
        row = {"id": "a", "prompt": "long", "labels": [" A"], "target": [1]}
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            tokenize_rows([row], Tokenizer(), max_seq=10)

    def test_empty_after_filtering_fails(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1] if kwargs else list(range(100))
        row = {"id": "a", "prompt": "long", "labels": [" A"], "target": [1]}
        with self.assertRaisesRegex(ValueError, "no examples"):
            tokenize_rows([row], Tokenizer(), max_seq=10)

    def test_external_metric_group_count(self):
        rows = [{"group_id": "a", "primitive": "noul", "correct": True,
                 "top1_probability": 0.8, "nll": 0.22, "brier": 0.08}] * 2
        summary = summarize_rows(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["groups"], 1)
        self.assertAlmostEqual(summary["ece"], 0.2)


@unittest.skipUnless(HAS_MLX and os.environ.get("JEV_TEST_TRAINING"), "set JEV_TEST_TRAINING for a two-step LoRA smoke test")
class AdapterRoundTripTests(unittest.TestCase):
    def test_train_save_load_and_evaluate(self):
        from jev.engine import SystemOneEngine
        model = os.environ.get("JEV_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, messages in {
                "train": ["Please refund my order.", "When will my package arrive?", "I want my money back.", "Thanks for your help."],
                "development": ["Return the duplicate payment please.", "How do I reset my password?"],
            }.items():
                rows = []
                for i, message in enumerate(messages):
                    rows.append(Example(
                        id=f"{split}/{i}", group_id=f"{split}/{i}", source="fixture",
                        state={"message": message}, question=Question("noul", "Does `message` request a refund?"),
                        target=[float(i % 2 == 1), float(i % 2 == 0)], target_origin="gold",
                        provenance={"fixture": True},
                    ).to_dict())
                (root / f"{split}.jsonl").write_text("".join(dumps(row) + "\n" for row in rows))
            adapter = root / "adapter"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/train_lora.py"), "--model", model,
                "--train", str(root / "train.jsonl"), "--val", str(root / "development.jsonl"),
                "--out", str(adapter), "--batch-size", "2", "--max-steps", "2", "--lr", "1e-4",
            ], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((adapter / "adapter_config.json").read_text())
            self.assertEqual(config["renderer_version"], STRUCTURED_V1)
            manifest = json.loads((adapter / "training_manifest.json").read_text())
            self.assertEqual(manifest["steps"], 2)
            self.assertEqual(manifest["examples_seen"], 4)
            engine = SystemOneEngine(model, adapter_path=str(adapter))
            self.assertEqual(engine.renderer_version, STRUCTURED_V1)
            state = {"message": "Please refund my order."}
            q = Question("noul", "Does `message` request a refund?")
            first = engine.ask(state, {"q": q})["q"].noul
            del engine
            engine = SystemOneEngine(model, adapter_path=str(adapter))
            self.assertEqual(first, engine.ask(state, {"q": q})["q"].noul)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/eval_dataset.py"), "--model", model,
                "--adapter", str(adapter), "--data", str(root / "development.jsonl"),
                "--out-dir", str(root / "evaluation"),
            ], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "evaluation/report.json").read_text())
            self.assertEqual(report["overall_micro"]["n"], 2)
            self.assertEqual(report["renderer_version"], STRUCTURED_V1)
            predictions = (root / "evaluation/predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
