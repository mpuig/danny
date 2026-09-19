import json
import tempfile
import unittest
from pathlib import Path

from jev.serialization import dumps
from scripts.compare_runs import paired


class ComparisonTests(unittest.TestCase):
    def test_grouped_pairing_and_mismatched_data_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, better in [("first", False), ("second", True)]:
                path = root / name
                path.mkdir()
                rows = []
                for i in range(2):
                    target = [float(i == 0), float(i == 1)]
                    probabilities = [
                        0.8 if (j == i) == better else 0.2 for j in range(2)
                    ]
                    rows.append(
                        {
                            "id": str(i),
                            "group_id": "same-group",
                            "primitive": "noul",
                            "answer_keys": ["no", "yes"],
                            "target": target,
                            "probabilities": probabilities,
                        }
                    )
                (path / "predictions.jsonl").write_text(
                    "".join(dumps(r) + "\n" for r in rows)
                )
                (path / "report.json").write_text(
                    json.dumps({"data_sha256": "fixture", "overall_micro": {}})
                )
            result = paired(root / "first", root / "second", samples=20)
            self.assertEqual(result["groups"], 1)
            self.assertEqual(result["examples"], 2)
            self.assertEqual(
                result["delta_second_minus_first"]["accuracy"]["estimate"], 1
            )
            self.assertEqual(
                result["delta_second_minus_first"]["accuracy"][
                    "group_bootstrap_95_interval"
                ],
                [1, 1],
            )
            (root / "second/report.json").write_text(
                json.dumps({"data_sha256": "changed", "overall_micro": {}})
            )
            with self.assertRaises(ValueError):
                paired(root / "first", root / "second")


if __name__ == "__main__":
    unittest.main()
