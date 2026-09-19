# Data inventory and integrity

Local audit: **2026-09-19**. These are observations about files on the development
machine, not guarantees about future downloads. `data/` is gitignored. No data was
modified, no datasets were downloaded, and no teacher API calls were made during
this audit.

## External files

| File under `data/external/` | Records | Questions | Status |
|---|---:|---:|---|
| `kev_pp4_train.jsonl` | 10,000 | 13,000 | Valid JSONL; SHA-256 matches local manifest |
| `kev_pp4_test.jsonl` | 980 | 1,244 | Valid JSONL; SHA-256 matches local manifest |
| `kev_pp4_manifest.json` | — | — | Manifest version 2; includes dataset revisions and split hashes |
| `nimble_train.jsonl` | 0 valid | — | **Invalid: 14 bytes containing `404: Not Found`** |
| `nimble_eval.jsonl` | 0 valid | — | **Invalid: 14 bytes containing `404: Not Found`** |

The Nimble paths are failed-download bodies, not datasets. Exclude them from all
training/evaluation until valid files and provenance are available. Their presence
does not establish whether upstream data is published or unavailable.

### Kev provenance and splits

The local files match these entries in `kev_pp4_manifest.json`:

```text
train.jsonl  21807df00aa0ab67b5c499f902316e76f8b65b609c31cc3f12f07e9fcbdcb525
test.jsonl   34b527ea42bfcf23c321fac642118e50434c54fa5756a208dc7dccfa98586613
```

This verifies correspondence to the supplied manifest, not independent authenticity
or license clearance. Source rows carry `_meta` fields including source, revision,
original split, ID, group ID, and variant.

The manifest also lists **calibration** (600 records / 780 questions) and
**development** (980 / 1,244) partitions. Neither file is present locally. Do not
describe those partitions as available, or substitute the test set for them.

There are no held-out sources in this manifest (`holdout_sources: []`). The
`policy.eval_only` list names possible sources; it does not mean those sources
are contained in the downloaded test file. This is an in-family test benchmark,
not evidence of unseen-task generalization after training on its training split.

The raw test file has **800 unique group IDs**, not 980 independent documents:
800 clean rows plus 60 `none_present`, 60 `none_absent`, and 60 `permuted` variants.
Keep all variants of a group in one partition. Bootstrap by group when estimating
uncertainty rather than treating every question or variant as independent.

State shapes also vary:

| Raw split | String states | Object states | Array states |
|---|---:|---:|---:|
| Train | 6,784 | 2,531 | 685 |
| Test | 681 | 232 | 67 |

This makes the files useful for structured-input testing. The current converter
renders non-string states using Python representations and discards `_meta`.
Keep the originals; the converted prompts are not sufficient provenance.

### What the current Kev converter retains

`scripts/convert_kev.py` excludes `sst5` by default because it shares the Stanford
Sentiment Treebank with SST-2. It also skips Choice questions above 26 options,
which removes `banking77`. This is a v0 limitation, not a reason to discard the
original wide-option data.

| Converted artifact | Question rows | Choice | Noul | Score |
|---|---:|---:|---:|---:|
| `data/kev_train.jsonl` | 11,000 | 4,000 | 5,000 | 2,000 |
| `data/kev_test.jsonl` | 1,048 | 464 | 424 | 160 |

Retained task labels: `boolq`, `agnews`, `agnews_yn`, `mnli`, `yelp`, `yelp_yn`,
`trec`, `dbpedia14`, `amazon`, `imdb`. These are ten labels from eight underlying
sources; derived yes/no tasks are not independent datasets. The converted test
contains 640 distinct rendered states shared across questions and variants.

## Existing local training artifacts

| File | Rows | Contents |
|---|---:|---|
| `data/train.jsonl` | 8,000 | 2,000 each: ag_news, dbpedia, imdb, yelp_stars |
| `data/val.jsonl` | 800 | 200 each from the same four source training splits |
| `data/distill_train.jsonl` | 2,000 | 500 Jev-target rows per source |
| `data/jev_answers_sst2_100.jsonl` | 100 | Cached teacher answers for agreement evaluation |
| `data/jev_answers_tweet_emotion_100.jsonl` | 100 | Cached teacher answers for agreement evaluation |

The current recast train/val files include shuffled options. Historical adapter
results were recorded as pre-shuffling runs. File names do not prove which data
version trained an adapter; no complete adapter-to-dataset manifest exists yet.

### Overlap found

Counts below are intersections of distinct **rendered state strings**, extracted
from the saved prompts, not whole-prompt matches. Different questions or option
orders do not make a shared state independent.

| Pair | Exact shared states |
|---|---:|
| Recast train / recast validation | 0 |
| Distillation train / recast validation | **6** (1 ag_news, 5 imdb) |
| Kev converted train / recast validation | **4** |
| Recast train / distillation train | 49 |
| Recast train / Kev converted train | 28 |
| Distillation train / Kev converted train | 7 |
| Kev converted train / Kev converted test | 0 |

Lowercasing and collapsing whitespace increases recast/Kev training overlap to
33 and distillation/Kev training overlap to 8; both validation overlap counts stay
unchanged. Training/training overlap mainly changes sampling weights. Training/
validation overlap invalidates strict separation and must be removed before use.

These checks are not fuzzy decontamination. Wrapping the same source text in
objects/arrays, truncation, paraphrases, and near-duplicates can conceal overlap.
Prefer source IDs and group IDs, supplemented by normalized content matching
across sources. Nothing here establishes absence of backbone pretraining exposure.

### Teacher target checks

Four distillation rows have target sums of **0.99**: lines 49, 835, 1516, and 1859.
All other rows sum to one within `1e-6`. Validate finiteness, bounds, and nonzero
mass, then normalize rounded distributions before using them in losses or metrics.
The current loader does not do this.

Recomputed from the stored targets:

- Gold-label argmax agreement: **86.3%**, using first-index tie breaking.
- **52.15%** have `max(target) < 1`; this is a defined softness statistic, not proof
  of calibrated ambiguity. One tied row changes agreement if ties count as correct.

These replace the earlier undocumented 88.4% / 45% summaries. Teacher agreement
with labels is not a measurement of the student's performance or calibration.

## Before the next training run

1. Reject malformed downloads and verify hashes before conversion.
2. Preserve source revision, source/group ID, primitive, original structured state,
   question, target origin, and rendering version in a canonical record format.
3. Reserve test groups first; separate training, development, and calibration.
   Exclude training overlaps across **all** sources, not just within each file.
4. Keep SST-5 out while SST-2 is a reserved evaluation dataset. Verify other
   cross-source overlaps rather than relying on different random seeds.
5. Obtain the missing Kev partitions or define a documented group-disjoint split
   from its training partition. Never fit calibration on its test partition.
6. Do not concatenate existing rendered files as the definitive training mix:
   they lose provenance and use inconsistent augmentation and serialization.
7. Keep teacher evaluation caches out of training. Pin teacher versions and record
   timestamps for future pulls. Check dataset licenses and distillation terms.

A minimal download-integrity check (no model load or network calls):

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('data/external')
manifest = json.loads((root / 'kev_pp4_manifest.json').read_text())
for split in ('train', 'test'):
    path = root / f'kev_pp4_{split}.jsonl'
    expected = manifest['files'][f'{split}.jsonl']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == expected['records']
    assert sum(len(row['questions']) for row in rows) == expected['questions']
    print(path, 'manifest matches')
for path in sorted(root.glob('nimble_*.jsonl')):
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows and all(isinstance(row, dict) for row in rows)
        print(path, 'JSON objects parsed; schema/provenance still need validation')
    except (ValueError, AssertionError) as error:
        print(path, 'INVALID:', error)
PY
```
