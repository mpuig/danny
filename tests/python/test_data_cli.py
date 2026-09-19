import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jev.data import assert_disjoint, load_examples, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def run_script(script, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def make_inputs(root):
    files = {}
    for split, count in [("train", 30), ("test", 5)]:
        path = root / f"{split}.jsonl"
        rows = []
        for i in range(count):
            identity = f"{split}/{i}"
            rows.append({
                "state": {"document": f"Document {identity}"},
                "questions": {"q": {"type": "noul", "instructions": "Is it true?", "src": "boolq", "label": True}},
                "_meta": {"id": identity, "group_id": identity, "source": "boolq",
                          "repo": "google/boolq", "revision": "test-revision", "split": split, "variant": "clean"},
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[path.name] = {"sha256": sha256_file(path), "records": count, "questions": count}
    (root / "manifest.json").write_text(json.dumps({"files": files}))


class DataCLITests(unittest.TestCase):
    def test_model_free_import(self):
        result = subprocess.run(
            [sys.executable, "-c", "import jev.data, sys; assert 'mlx.core' not in sys.modules"],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_audit_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_inputs(root)
            args = ["--train", root / "train.jsonl", "--test", root / "test.jsonl",
                    "--manifest", root / "manifest.json"]
            output = root / "prepared"
            result = run_script("prepare_data.py", *args, "--out-dir", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            partitions = {split: load_examples(output / f"{split}.jsonl")
                          for split in ["train", "development", "calibration", "test"]}
            assert_disjoint(partitions)
            self.assertEqual(len(partitions["train"]), 24)
            self.assertEqual(len(partitions["development"]), 3)
            self.assertEqual(len(partitions["calibration"]), 3)
            self.assertEqual(len(partitions["test"]), 5)
            manifest = json.loads((output / "manifest.json").read_text())
            for filename, entry in manifest["files"].items():
                self.assertEqual(sha256_file(output / filename), entry["sha256"])
            result = run_script("audit_data.py", *sorted(output.glob("*.jsonl")))
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script("prepare_data.py", *args, "--out-dir", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            second = root / "prepared-again"
            result = run_script("prepare_data.py", *args, "--out-dir", second)
            self.assertEqual(result.returncode, 0, result.stderr)
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_failed_download_and_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_inputs(root)
            (root / "test.jsonl").write_text("404: Not Found")
            result = run_script("audit_data.py", root / "train.jsonl", root / "test.jsonl")
            self.assertEqual(result.returncode, 1)
            reports = json.loads(result.stdout)
            self.assertTrue(reports[0]["valid"])
            self.assertFalse(reports[1]["valid"])
            output = root / "prepared"
            result = run_script("prepare_data.py", "--train", root / "train.jsonl", "--test", root / "test.jsonl",
                                "--manifest", root / "manifest.json", "--out-dir", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
