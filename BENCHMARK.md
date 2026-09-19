# Example benchmark: local Qwen vs Jev

Measured **2026-09-19**, using Qwen HTTP predictions from the existing server on
port **8399**, then refreshing Jev on the same 24 frozen examples.

**Fresh result:** Qwen matched the recorded dataset labels on **19/24 (79.2%)**;
**Jev 1.13.0 matched 21/24 (87.5%)**. Their most likely answers agreed on
**16/24 (66.7%)**. Neither model's answer was treated as ground truth.

The initial comparison used an unversioned cache. The live follow-up made exactly
**24 external API requests**, with no retries. The first requested `jev-latest`,
which reported `jev-1.13.0`; the remaining 23 explicitly requested that version.
Every response reported `jev-1.13.0`. Original Qwen predictions were reused.

**No Jev argmax decisions changed** between cached and fresh responses. The largest
probability change was **0.03**. Cached results remain alongside the live results
below; their original teacher version is still unknown.

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
| Fresh Jev reference | `jev-1.13.0`, reported on all 24 live responses |
| Historical reference | Existing cache originally requested `jev-latest`; resolved version unknown |
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

| Primitive | Cases | Qwen correct | Fresh Jev correct | Same most likely answer |
|---|---:|---:|---:|---:|
| Choice | 8 | 7/8 | 6/8 | 5/8 |
| Noul | 8 | 7/8 | 8/8 | 7/8 |
| Score | 8 | 5/8 | 7/8 | 4/8 |
| **Total** | **24** | **19/24** | **21/24** | **16/24** |

The cached and fresh Jev responses have identical argmax labels, so the original
accuracy and agreement counts are unchanged.

### Probability quality and agreement

| Metric | Qwen | Fresh Jev 1.13.0 | Cached Jev |
|---|---:|---:|---:|
| Multiclass Brier error, lower is better | 0.2879 | 0.2422 | 0.2432 |
| Negative log-likelihood (NLL), lower is better | 0.4631 | 1.4038* | 1.4041* |
| Score mean absolute error, 0–4 level scale | 0.4869 | 0.2775 | 0.2775 |

\* **Do not interpret the NLL column as proof that Qwen is better calibrated.**
Both cached and fresh Jev distributions assign zero probability to the recorded
label in B03. With the declared `1e-12` floor, that one case contributes about
**1.151** to aggregate NLL. Returned Jev probabilities have two-decimal precision
in these responses; underlying unrounded probabilities are unavailable. Dataset
labels can also disagree with a reasonable rubric interpretation.

Qwen-versus-fresh-Jev agreement metrics compare distributions, not correctness:

- Mean Jensen–Shannon divergence: **0.129 nats**; zero means identical distributions
  (cached comparison: 0.128).
- Mean total variation distance: **0.302** on a 0–1 scale (cached: 0.300).
- Noul mean absolute difference in `P(yes)`: **0.083** (cached: 0.081).

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

| Option | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| world | 0.790 | 0.800 | 0.024 |
| sports | 0.000 | 0.000 | 0.000 |
| business | 0.010 | 0.010 | 0.059 |
| science_tech | 0.200 | 0.190 | 0.917 |

Qwen matches the dataset category; Jev selects world news. Government seizure of
servers plausibly crosses category boundaries, so this is not an independently
adjudicated Jev error.

### B02 — Choice: a retail executive retires

> Wal-Mart Vice Chairman Coughlin to Retire (Reuters)…

Question: **“What is the main topic of the news article in the state?”**
Recorded label: `business`.

| Option | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| world | 0.000 | 0.000 | 0.031 |
| sports | 0.000 | 0.000 | 0.002 |
| business | 1.000 | 1.000 | 0.956 |
| science_tech | 0.000 | 0.000 | 0.011 |

Both select business. Jev's returned `1.000` does not establish perfect certainty;
this is the API-visible value, not an unrounded internal probability.

### B09 — Noul: praise mixed with a movie-plot description

> A real classic. … A cult film waiting to happen!

Question: **“Is the reviewer's overall opinion of the movie favorable?”**
Recorded label: **yes**. The full input also contains the film's plot description.

| Value | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| P(yes) | 0.670 | 0.690 | 0.307 |
| Decision at a 0.5 threshold | yes | yes | no |

Qwen disagrees with both the recorded label and Jev. The example shows a local
sentiment failure, not its cause; no controlled wording ablation was performed.

### B10 — Noul: enthusiastic television review

> The Sopranos is probably the most widely acclaimed TV series ever, so naturally
> my expectations were through the roof, and yet the show surpassed them.

Question: **“Is the reviewer's overall opinion of the movie favorable?”**
Recorded label: **yes**. The original question says “movie” even though this review
concerns television; that wording was preserved for both models.

| Value | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| P(yes) | 0.960 | 0.960 | 0.996 |
| Decision at a 0.5 threshold | yes | yes | yes |

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

| Level | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| 0 — Terrible | 0.090 | 0.100 | 0.045 |
| 1 — Poor | 0.890 | 0.890 | 0.663 |
| 2 — Mixed | 0.020 | 0.010 | 0.276 |
| 3 — Good | 0.000 | 0.000 | 0.013 |
| 4 — Excellent | 0.000 | 0.000 | 0.002 |
| **Weighted-mean score** | **0.930** | **0.910** | **1.264** |

Both select level 1 as most likely, but Qwen assigns more weight to “Mixed.”

### B18 — Score: satisfactory clothing alterations

> The workers at the front desk are pleasant and helpful. … They did a great job
> on my suits, not the cheapest, $25 for both but I was willing to pay to have it done right.

Question: **“Rate the overall sentiment of the review in the state.”**
Recorded level: **3**. The same five-level rubric applies.

| Level | Cached Jev | Fresh Jev | Qwen |
|---|---:|---:|---:|
| 0 — Terrible | 0.000 | 0.000 | 0.012 |
| 1 — Poor | 0.000 | 0.000 | 0.056 |
| 2 — Mixed | 0.000 | 0.000 | 0.503 |
| 3 — Good | 0.530 | 0.560 | 0.392 |
| 4 — Excellent | 0.470 | 0.440 | 0.036 |
| **Weighted-mean score** | **3.470** | **3.440** | **2.385** |

Jev's modal level matches the recorded rating. Qwen places its largest probability
on “Mixed” and produces a lower mean. This is one reason the Score aggregate is
weaker for Qwen in this sample.

## Every selected case

Labels below are argmax labels, not `confidence` values or weighted-mean scores.
Full distributions, inputs, original teacher answers, and raw local responses are
preserved in the local artifacts.

| Case | Primitive | Recorded label | Jev (cached and fresh) | Qwen |
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

The live follow-up completed all 24 Jev calls without retries or errors:

- Median external HTTP latency: **804 ms**, range **720–995 ms**.
- These are serial HTTPS requests with new connections, including WAN/TLS overhead.
  They were not measured simultaneously with Qwen; this is not a model-compute
  speed comparison or a production latency guarantee.
- Reported usage: **12,915 input tokens / 996 output tokens**. Local and Jev usage
  counters have different meanings; nonzero Jev output accounting does not establish
  autoregressive answer generation. Actual billed cost was not retrieved.

## Reproduce the local/cached comparison

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

## Reproduce the live Jev refresh

This command makes **paid external API requests** using `TYPESAFE_API_KEY` from the
environment or local `.env`. Run only when you intend to spend that budget:

```fish
uv run python scripts/refresh_benchmark_jev.py \
  --source-run data/runs/benchmark-examples-v2 \
  --teacher jev-1.13.0 --max-requests 24 --confirm-live \
  --out-dir data/runs/benchmark-live-jev-new
```

It reuses the frozen Qwen predictions and verifies the saved case hash and
request/label alignment before calling Jev. There are no automatic retries or
resumes. It stops on an error or model-version mismatch; inspect partial artifacts
before authorizing another run. `--teacher jev-latest` instead resolves the first
response, then pins that reported version for the remaining requests.

Fresh artifacts are in `data/runs/benchmark-live-jev-v1/`:

- `manifest.json`: source hashes, collection policy, and local model identity.
- `requests.jsonl`: exact submitted state/question/model and timestamps; no credentials.
- `responses.jsonl`: unmodified full responses, model IDs, usage, and elapsed times.
- `predictions.jsonl`: aligned Qwen, cached Jev, and fresh Jev distributions.
- `report.json`: metrics, version consistency, token totals, and timings.

Five additional tests cover explicit live consent, request budgets, frozen input
hashes, reported-version pinning, rounding, and no retries or persisted credentials.
Run both benchmark test files without any external API calls:

```fish
uv run python -m unittest discover -s tests/python -p '*benchmark.py' -v
```

## What this does not establish

- General equivalence with Jev, or behavior of versions other than the observed
  `jev-1.13.0` responses on these 24 cases.
- Generalization to unfamiliar support workflows, including the refund example.
- Safe automation thresholds, reliable calibration, or a statistically established ranking.
- Absence of pretraining/fuzzy overlap. Original source IDs are unknown.
- Perfect gold labels for truncated states: historical collection kept only the
  first 1,500 characters while retaining the original dataset labels.
- Jev-equivalent Score internals. This Qwen adapter scores levels jointly; Jev's
  public documentation describes independent level evaluation.

The live refresh resolves the missing teacher-version evidence for these cases;
it does not remove the sampling, label, or task-family limitations. See
[Experiments](docs/EXPERIMENTS.md) for the larger controlled local evaluations and
[Serving](docs/SERVING.md) for server options.
