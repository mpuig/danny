import tempfile
import unittest
from pathlib import Path

from jev.data import (
    Example, assert_disjoint, assert_training_disjoint, content_key, kev_examples,
    load_examples, load_training_rows, normalize_target, read_jsonl, sha256_file,
    split_examples, verify_kev_file,
)
from jev.rendering import LEGACY_V0, STRUCTURED_V1, render
from jev.schema import Question
from jev.serialization import dumps


def example(name, state=None, group=None):
    return Example(
        id=name, group_id=group or f"group:{name}", source="test",
        state=state if state is not None else f"Document {name}",
        question=Question("noul", "Does it fit?"), target=[0.0, 1.0],
        target_origin="gold", provenance={"revision": "test-revision"},
    )


def kev_row(name, state=None, variant="clean", label=True):
    return {
        "state": state if state is not None else f"Document {name}",
        "questions": {"q": {"type": "noul", "instructions": "Is it true?", "label": label, "src": "boolq"}},
        "_meta": {
            "id": name, "group_id": name, "source": "boolq", "repo": "google/boolq",
            "revision": "pinned", "split": "train", "variant": variant,
        },
    }


class TargetTests(unittest.TestCase):
    def test_normalization_retains_original(self):
        row = example("a").to_dict()
        row["target"] = [0.49, 0.5]
        row.pop("original_target")
        parsed = Example.from_dict(row)
        self.assertEqual(parsed.original_target, [0.49, 0.5])
        self.assertAlmostEqual(sum(parsed.target), 1)
        self.assertAlmostEqual(parsed.target[0], 0.49 / 0.99)
        self.assertEqual(Example.from_dict(parsed.to_dict()), parsed)
        row["original_target"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "disagree"):
            Example.from_dict(row)

    def test_invalid_targets(self):
        for target in [[0, 0], [0.8, 0.8], [-0.1, 1.1], [True, False], [float("nan"), 0], [1]]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                normalize_target(target, 2)


class FileTests(unittest.TestCase):
    def test_bad_downloads_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            for text in ["404: Not Found", "<html>oops</html>", "[]", "", "{\"a\":1,\"a\":2}"]:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, "data.jsonl"):
                    list(read_jsonl(path))
            path.write_text((dumps(example("a").to_dict()) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate example"):
                load_examples(path)

    def test_hash_and_count_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(dumps(kev_row("a")) + "\n")
            manifest = {"files": {"train.jsonl": {"sha256": sha256_file(path), "records": 1, "questions": 1}}}
            self.assertEqual(len(verify_kev_file(path, manifest, "train")[0]), 1)
            manifest["files"]["train.jsonl"]["questions"] = 2
            with self.assertRaisesRegex(ValueError, "question count"):
                verify_kev_file(path, manifest, "train")
            path.write_text("404: Not Found")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_kev_file(path, manifest, "train")

    def test_training_renders_canonical_and_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            item = example("a", {"ticket": {"message": "hello"}})
            path.write_text(dumps(item.to_dict()) + "\n")
            row = load_training_rows(path)[0]
            self.assertEqual(row["renderer_version"], STRUCTURED_V1)
            self.assertEqual((row["prompt"], row["labels"]), render(item.state, item.question))
            with self.assertRaisesRegex(ValueError, "leakage"):
                assert_training_disjoint([row], [row])
            prompt, labels = render("hello", item.question, LEGACY_V0)
            path.write_text(dumps({"prompt": prompt, "labels": labels, "target": [0.49, 0.5], "task": "old"}) + "\n")
            row = load_training_rows(path)[0]
            self.assertEqual(row["renderer_version"], LEGACY_V0)
            self.assertEqual(row["prompt"], prompt)
            self.assertAlmostEqual(sum(row["target"]), 1)
            with self.assertRaisesRegex(ValueError, "requires legacy"):
                load_training_rows(path, STRUCTURED_V1)
            path.write_text(path.read_text() + dumps(item.to_dict()) + "\n")
            with self.assertRaisesRegex(ValueError, "mixed"):
                load_training_rows(path)


class KevTests(unittest.TestCase):
    def test_metadata_and_structures_preserved(self):
        row = kev_row("a", {"ticket": {"channel": "email", "body": "Question"}})
        row["questions"]["choice"] = {
            "type": "choice", "instructions": {"question": "Which?"},
            "criteria": {"b": None, "a": {"meaning": "the answer"}}, "label": "a", "src": "example",
        }
        row["questions"]["score"] = {
            "type": "score", "instructions": "Rate", "criteria": ["low", "high"], "label": 0, "src": "example",
        }
        items, skipped = kev_examples([row], "hash")
        self.assertEqual(skipped, {})
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].state, row["state"])
        self.assertEqual(items[0].provenance["upstream"], row["_meta"])
        self.assertEqual(items[1].question.instructions, {"question": "Which?"})
        self.assertEqual(items[1].target, [0, 1])
        self.assertEqual(items[2].target, [1, 0])
        self.assertEqual(len({item.group_id for item in items}), 1)
        self.assertEqual(len({item.id for item in items}), 3)

    def test_invalid_labels_are_not_silently_coerced(self):
        for label in ["yes", 1, "false", None]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "boolean"):
                kev_examples([kev_row("a", label=label)], "hash")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            kev_examples([kev_row("a"), kev_row("a")], "hash")
        row = kev_row("a")
        del row["_meta"]["revision"]
        with self.assertRaisesRegex(ValueError, "revision"):
            kev_examples([row], "hash")

    def test_filters_are_reported(self):
        wide = kev_row("wide")
        wide["questions"] = {"wide": {
            "type": "choice", "instructions": "Which?", "criteria": {str(i): None for i in range(77)},
            "label": "2", "src": "banking77",
        }}
        sst = kev_row("sst")
        sst["questions"]["q"]["src"] = "sst5"
        items, skipped = kev_examples([wide, sst, kev_row("normal")], "hash")
        self.assertEqual(len(items), 1)
        self.assertEqual(skipped, {"wide_choice": 1, "sst5": 1})
        items, _ = kev_examples([wide], "hash", max_options=255)
        self.assertEqual(len(items[0].target), 77)


class SplitTests(unittest.TestCase):
    def test_normalized_known_wrappers(self):
        states = [" Hello\nWORLD ", {"document": "hello world"},
                  {"ticket": {"channel": "email", "body": "Hello world"}},
                  [{"role": "customer", "content": "HELLO WORLD"}]]
        self.assertEqual(len({content_key(state) for state in states}), 1)
        # Multi-field contexts are not collapsed to whichever text field matches.
        self.assertNotEqual(content_key({"document": "hello world", "policy": "a"}), content_key("hello world"))

    def test_transitive_holdout_priority_and_variant_grouping(self):
        train = [example(str(i)) for i in range(20)]
        train += [example("leak-a", "same", "related"),
                  example("leak-b", "bridge", "related"),
                  example("leak-c", "bridge", "another")]
        test = [example("heldout", {"document": "SAME"}, "test-group")]
        partitions, removed = split_examples(train, test)
        self.assertEqual(set(removed), {"leak-a", "leak-b", "leak-c"})
        self.assertEqual(partitions["test"], test)
        self.assertEqual(sum(map(len, partitions.values())), 21)
        assert_disjoint(partitions)
        # Stable across source ordering; split RNG operates on sorted components.
        other, _ = split_examples(list(reversed(train)), list(reversed(test)))
        self.assertEqual(partitions, other)

    def test_related_variants_stay_together(self):
        train = [example(str(i)) for i in range(20)]
        train += [example("variant-a", "different a", "same-group"),
                  example("variant-b", "different b", "same-group")]
        partitions, _ = split_examples(train, [example("test")])
        places = [split for split, rows in partitions.items() for row in rows if row.group_id == "same-group"]
        self.assertEqual(len(places), 2)
        self.assertEqual(len(set(places)), 1)
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint({"train": [example("a", "X")], "test": [example("b", {"document": "x"})]})

    def test_invalid_fractions_and_tiny_corpora(self):
        for fraction in [0, 1, float("nan")]:
            with self.assertRaises(ValueError):
                split_examples([example("a")], [example("b")], development_fraction=fraction)
        with self.assertRaisesRegex(ValueError, "not enough"):
            split_examples([example("a")], [example("b")])


if __name__ == "__main__":
    unittest.main()
