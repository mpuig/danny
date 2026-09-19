"""Validate JSONL datasets without loading a model. Invalid files return exit code 1.

    uv run python scripts/audit_data.py data/external/*.jsonl
    uv run python scripts/audit_data.py data/kev-v1/*.jsonl

This is a file/schema audit. prepare_data performs grouped cross-split checks;
training also rejects train/validation overlaps. No files are changed.
"""

import argparse
import json

from jev.data import kev_examples, load_examples, load_training_rows, read_jsonl, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    reports = []
    for path in args.files:
        report = {"path": path}
        try:
            report["sha256"] = sha256_file(path)
            rows = list(read_jsonl(path))
            report["records"] = len(rows)
            if "format_version" in rows[0]:
                examples = load_examples(path)
                report.update(format="canonical-v1", examples=len(examples))
            elif "questions" in rows[0]:
                kev_examples(rows, report["sha256"], max_options=255)
                report.update(format="kev", questions=sum(len(r["questions"]) for r in rows))
            else:
                examples = load_training_rows(path)
                report.update(format="legacy-v0", examples=len(examples))
            report["valid"] = True
        except (OSError, KeyError, TypeError, ValueError) as exc:
            report.update(valid=False, error=str(exc))
        reports.append(report)
    print(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(r["valid"] for r in reports) else 1)


if __name__ == "__main__":
    main()
