# Example benchmark: local Qwen vs cached Jev

Measured **2026-09-19**, using the existing Qwen HTTP server on port **8399**.

**Result on 24 examples:** Qwen matched the recorded dataset labels on **19/24
(79.2%)**; cached Jev matched **21/24 (87.5%)**. Their most likely answers agreed
on **16/24 (66.7%)**. Neither model's answer was treated as ground truth.

**This is not a live comparison with today's Jev.** The reference consists of
previously collected, unversioned Jev responses. No external API calls were made.
These few examples illustrate behavior; they do not establish model equivalence,
calibration, or a reliable ranking for production workloads.

## Configuration and selection

| Item | Configuration |
|---|---|
| Local URL | `http://127.0.0.1:8399` |
| Served identity | `Qwen/Qwen3-0.6B@237c66883c32d5e6` |
| Adapter | `adapters/qwen3-0.6b-structured-v1-lr1e-5` |
| Rendering / readout | `structured-v1` / `letters-v1` |
| Precision / execution | Native / independent; configured microbatch size 4 |
| Fitted temperature | None, verified in model discovery |
| Contextual correction | Off in the supplied server launch command |
| Jev reference | Existing cache originally requested `jev-latest`; resolved version unknown |
| Sources | AG News, DBpedia, IMDB, Yelp ratings |
| Questions | 8 Choice, 8 Noul, 8 Score; one question per HTTP request |

Selection was fixed **before looking at Qwen predictions**:

1. Start with the existing 400-example cached-teacher test partition.
2. Exclude matches against local training, development, and calibration data.
   Check canonical IDs/groups, normalized content, recognized document wrappers,
   and 1,500-character prefixes to catch the historical teacher truncation.
   **Two reference candidates were excluded**, leaving 398 eligible examples.
3. Sort candidates by `SHA256("42:" + example_id)`. Select four AG News, four
   DBpedia, eight IMDB, and eight Yelp cases, without repeated content/groups.
4. Send the recovered original state and question to the local server. Preserve
   option order, wording, and the historical truncation; do not rewrite inputs.

The exclusion files match the adapter's recorded training/development hashes.
The reserved main Kev test was **not opened or evaluated**. Matching is not fuzzy
or pretraining decontamination; unknown source IDs prevent stronger guarantees.
These are familiar task families, not unseen workflow rubrics.

## Results

Accuracy means the highest-probability label matches the original dataset label.
For Score, this tests the **modal level**, not the rounded weighted-mean score.

| Primitive | Cases | Qwen correct | Cached Jev correct | Same most likely answer |
|---|---:|---:|---:|---:|
| Choice | 8 | 7/8 | 6/8 | 5/8 |
| Noul | 8 | 7/8 | 8/8 | 7/8 |
| Score | 8 | 5/8 | 7/8 | 4/8 |
| **Total** | **24** | **19/24** | **21/24** | **16/24** |

### Probability quality and agreement

| Metric | Qwen | Cached Jev |
|---|---:|---:|
| Multiclass Brier error, lower is better | 0.288 | 0.243 |
| Negative log-likelihood (NLL), lower is better | 0.463 | 1.404* |
| Score mean absolute error, 0–4 level scale | 0.4869 | 0.2775 |

\* **Do not interpret the NLL column as proof that Qwen is better calibrated.**
The cached Jev distribution assigns zero probability to the recorded label in B03.
With the declared `1e-12` probability floor, that one case contributes about **1.151**
to its aggregate NLL. Cached probabilities are rounded; their internal precision
is unavailable. Dataset labels can also disagree with a reasonable rubric interpretation.

Agreement metrics compare distributions, not correctness:

- Mean Jensen–Shannon divergence: **0.128 nats**; zero means identical distributions.
- Mean total variation distance: **0.300** on a 0–1 scale.
- Noul mean absolute difference in `P(yes)`: **0.081**.

No ECE or statistical-significance claim is made from this small, source-balanced
but not label-balanced sample. The metrics do not use the API's derived `confidence`.

## Six examples

These are the first two selected cases of each primitive, not cases selected to
favor either model. Quotes below are excerpts; saved requests contain the full
teacher-visible state. Display probabilities are rounded to three decimals.

### B01 — Choice: servers seized from a media organization

> UK servers returned to media group by US feds Independent Media Center (Indymedia),
> the U.K.-based group responsible for running about 20 independent news Web sites…

Question: **“What subject area does the article in the state belong to?”**
Recorded label: `science_tech`.

| Option | Cached Jev | Qwen |
|---|---:|---:|
| world | 0.790 | 0.024 |
| sports | 0.000 | 0.000 |
| business | 0.010 | 0.059 |
| science_tech | 0.200 | 0.917 |

Qwen matches the dataset category; Jev selects world news. Government seizure of
servers plausibly crosses category boundaries, so this is not an independently
adjudicated Jev error.

### B02 — Choice: a retail executive retires

> Wal-Mart Vice Chairman Coughlin to Retire (Reuters)…

Question: **“What is the main topic of the news article in the state?”**
Recorded label: `business`.

| Option | Cached Jev | Qwen |
|---|---:|---:|
| world | 0.000 | 0.031 |
| sports | 0.000 | 0.002 |
| business | 1.000 | 0.956 |
| science_tech | 0.000 | 0.011 |

Both select business. Jev's cached `1.000` does not establish perfect certainty;
it is the stored, rounded value.

### B09 — Noul: praise mixed with a movie-plot description

> A real classic. … A cult film waiting to happen!

Question: **“Is the reviewer's overall opinion of the movie favorable?”**
Recorded label: **yes**. The full input also contains the film's plot description.

| Value | Cached Jev | Qwen |
|---|---:|---:|
| P(yes) | 0.670 | 0.307 |
| Decision at a 0.5 threshold | yes | no |

Qwen disagrees with both the recorded label and Jev. The example shows a local
sentiment failure, not its cause; no controlled wording ablation was performed.

### B10 — Noul: enthusiastic television review

> The Sopranos is probably the most widely acclaimed TV series ever, so naturally
> my expectations were through the roof, and yet the show surpassed them.

Question: **“Is the reviewer's overall opinion of the movie favorable?”**
Recorded label: **yes**. The original question says “movie” even though this review
concerns television; that wording was preserved for both models.

| Value | Cached Jev | Qwen |
|---|---:|---:|
| P(yes) | 0.960 | 0.996 |
| Decision at a 0.5 threshold | yes | yes |

Both agree. Qwen's more extreme probability is not evidence of better calibration.

### B17 — Score: a disappointing restaurant

> a lot of people seem to like this place, and i don't know really know why.
> i mean, the food just tastes okay. i think it's kind of pricey, too, for what you get.

Question: **“How satisfied does the customer in the state sound?”**
Recorded level: **1**. This benchmark uses five levels, **0–4**, not the earlier
three-level frustration example.

The exact rubric is:

0. Terrible experience; strong complaints and would not return.
1. Poor; mostly negative with significant issues.
2. Mixed; some good points and some clear problems.
3. Good; mostly positive with minor complaints.
4. Excellent; enthusiastic praise with no real complaints.

| Level | Cached Jev | Qwen |
|---|---:|---:|
| 0 — Terrible | 0.090 | 0.045 |
| 1 — Poor | 0.890 | 0.663 |
| 2 — Mixed | 0.020 | 0.276 |
| 3 — Good | 0.000 | 0.013 |
| 4 — Excellent | 0.000 | 0.002 |
| **Weighted-mean score** | **0.930** | **1.264** |

Both select level 1 as most likely, but Qwen assigns more weight to “Mixed.”

### B18 — Score: satisfactory clothing alterations

> The workers at the front desk are pleasant and helpful. … They did a great job
> on my suits, not the cheapest, $25 for both but I was willing to pay to have it done right.

Question: **“Rate the overall sentiment of the review in the state.”**
Recorded level: **3**. The same five-level rubric applies.

| Level | Cached Jev | Qwen |
|---|---:|---:|
| 0 — Terrible | 0.000 | 0.012 |
| 1 — Poor | 0.000 | 0.056 |
| 2 — Mixed | 0.000 | 0.503 |
| 3 — Good | 0.530 | 0.392 |
| 4 — Excellent | 0.470 | 0.036 |
| **Weighted-mean score** | **3.470** | **2.385** |

Jev's modal level matches the recorded rating. Qwen places its largest probability
on “Mixed” and produces a lower mean. This is one reason the Score aggregate is
weaker for Qwen in this sample.

## Every selected case

Labels below are argmax labels, not `confidence` values or weighted-mean scores.
Full distributions, inputs, original teacher answers, and raw local responses are
preserved in the local artifacts.

| Case | Primitive | Recorded label | Cached Jev | Qwen |
|---|---|---|---|---|
| B01 | Choice | science_tech | world | science_tech |
| B02 | Choice | business | business | business |
| B03 | Choice | science_tech | business | science_tech |
| B04 | Choice | business | business | business |
| B05 | Choice | transportation | transportation | transportation |
| B06 | Choice | office_holder | office_holder | artist |
| B07 | Choice | natural_place | natural_place | natural_place |
| B08 | Choice | album | album | album |
| B09 | Noul | yes | yes | no |
| B10 | Noul | yes | yes | yes |
| B11 | Noul | yes | yes | yes |
| B12 | Noul | yes | yes | yes |
| B13 | Noul | no | no | no |
| B14 | Noul | no | no | no |
| B15 | Noul | yes | yes | yes |
| B16 | Noul | yes | yes | yes |
| B17 | Score | 1 | 1 | 1 |
| B18 | Score | 3 | 3 | 2 |
| B19 | Score | 3 | 3 | 2 |
| B20 | Score | 1 | 0 | 1 |
| B21 | Score | 2 | 2 | 1 |
| B22 | Score | 2 | 2 | 2 |
| B23 | Score | 0 | 0 | 0 |
| B24 | Score | 0 | 0 | 0 |

## Local runtime observations

All 24 cases completed successfully in each of two passes. The second pass used
the same requests after additional runner validation was added. **Maximum probability
drift between passes was 0.0.** This is a repeatability check on one loaded model,
not a guarantee across versions or devices.

- First-pass median HTTP latency: **49.4 ms**, range **26.7–67.4 ms**.
- Repeat-pass median: **56.0 ms**, range **33.7–70.6 ms**.
- One question per request, serial requests, already-loaded server on the development
  Mac. No explicit warmup; the repeated requests could reuse token-cache entries.
- These are heterogeneous examples and small samples, not production p95 figures.
- **Jev latency and cost were not measured**, because its answers came from a cache.

## Reproduce

Keep the supplied server running:

```fish
uv run python scripts/serve.py \
  --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-lr1e-5 \
  --port 8399
```

If it is already running, do not start a second process on the same port.
In another terminal, from the repository root:

```fish
uv run python scripts/benchmark_cached_teacher.py \
  --base-url http://127.0.0.1:8399 \
  --out-dir data/runs/benchmark-examples-new
```

The runner requires the local teacher cache, canonical exclusion files, adapter
weights, and its training manifest. These are gitignored artifacts, not files
bundled with the repository. It does not download models or collect new Jev answers.
It verifies adapter identity, freezes selection before inference, checks response
identity/shape/distributions, and refuses an existing output directory. Redirects
and environment HTTP proxies are disabled; only loopback server URLs are accepted.

Results used here:

- `data/runs/benchmark-examples-v1/`: first pass and a frozen copy of its runner.
- `data/runs/benchmark-examples-v2/`: validated repeat, `manifest.json`,
  `cases.jsonl`, `predictions.jsonl`, `report.json`, and `repeat_check.json`.
- Case-file SHA-256: `4fbba1963b642401eb948547ae525a83080cf73ac60e4e8c70a5260d6031a675`.
- Reference SHA-256: `aca77fc3ad91835637ccb89cd9b92cf0f25911382fa950adde90a85585e7014d`.
- Metrics use normalized distributions, first-index argmax ties, and NLL clipped
  at `1e-12`. Brier is summed across labels, not divided by the number of labels.

Replay B01 from the saved request, with fish-compatible syntax:

```fish
head -n 1 data/runs/benchmark-examples-v2/cases.jsonl \
  | jq '.request' \
  | curl -sS http://127.0.0.1:8399/v1/systemone \
      -H 'Content-Type: application/json' --data-binary @- \
  | jq '.answers'
```

Five tests cover selection, overlap handling, probability/answer validation, metrics,
and rejection of remote URLs/redirects:

```fish
uv run python -m unittest discover -s tests/python -p test_cached_benchmark.py -v
```

## What this does not establish

- Agreement with the **current** Jev service or a pinned Jev version.
- Generalization to unfamiliar support workflows, including the refund example.
- Safe automation thresholds, reliable calibration, or a statistically established ranking.
- Absence of pretraining/fuzzy overlap. Original source IDs are unknown.
- Perfect gold labels for truncated states: historical collection kept only the
  first 1,500 characters while retaining the original dataset labels.
- Jev-equivalent Score internals. This Qwen adapter scores levels jointly; Jev's
  public documentation describes independent level evaluation.

A live follow-up requires a selected Jev version, an approved request/cost budget,
and fresh preserved responses. See [Experiments](docs/EXPERIMENTS.md) for the larger
controlled local evaluations and [Serving](docs/SERVING.md) for server options.
