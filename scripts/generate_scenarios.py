"""Generate synthetic decision scenarios with a cheap generator model (Fireworks).

Division of labor: the generator produces diverse, plausible STATES and RUBRICS —
it is never trusted for labels. Targets come later from a pinned Jev teacher
(scripts/score_scenarios.py). The generator's own expected answer is retained in
provenance for disagreement analysis only.

Diversity plan (seeded, sampled per scenario — not free-form generation):
  - domain            22 realistic domains
  - length tier       short (15-40 words) / medium (60-200) / long (300-700, with
                      2-3 irrelevant sections and decisive evidence planted at a
                      sampled early/middle/late position)
  - state format      plain text / ticket object / message thread / record object
  - primitive         choice (width tier 3-5 / 6-10 / 11-26) / score (3-10 levels)
                      / noul
  - question style    direct / inferential (combine two facts) / negated phrasing /
                      quantitative (thresholds, dates, amounts) / evidence-attribution
                      ("does the state explicitly state...") / speech-act
  - ambiguity         clear / ambiguous / missing-evidence
  - register          clean prose / informal with typos / terse operational notes /
                      OCR-ish artifacts / non-native phrasing
  - questions/state   1-3 for singles (same state, one row per question, shared group)
  - answer balance    for "clear" cells the correct answer slot is rotated by the
                      sampler so the generator cannot bias toward the first option
  - contrastive pairs ~20% of scenarios: <= 8-word edit flips the answer; pair
                      members share a group_id so grouped splits keep them together

Every scenario is validated through the project's own Question/state validators and
deduplicated across groups by normalized content key.

    uv run python scripts/generate_scenarios.py --n 24 --out data/experiments/synthetic-v1/scenarios.jsonl
    uv run python scripts/generate_scenarios.py --list-models deepseek   # pick a model id

Requires FIREWORKS_API_KEY (or --key-env NAME) in the environment or .env.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from jev.data import content_key
from jev.schema import Question
from jev.serialization import dumps, validate_state

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
RETRYABLE = (408, 429, 500, 502, 503, 529)
SCENARIO_FORMAT = "scenario-1"

DOMAINS = [
    "airline customer support", "SaaS billing support", "insurance claims intake",
    "e-commerce order issues", "bank fraud operations", "healthcare appointment admin",
    "logistics and shipping exceptions", "devops incident reports", "code review comments",
    "hiring and recruiting screens", "content moderation queues", "legal client intake",
    "property management requests", "restaurant delivery complaints",
    "telecom service tickets", "government benefits applications",
    "university admissions and registrar", "energy utility outages and billing",
    "travel booking changes", "IT helpdesk requests", "manufacturing quality reports",
    "nonprofit grant applications",
]

LENGTH_TIERS = {
    "short": ("a very short state of 15-40 words (one or two sentences or 3-5 record fields)", 500),
    "medium": ("a state of 60-200 words", 2500),
    "long": ("a LONG state of 300-700 words with 2-3 sections that are realistic but "
             "irrelevant to the question; place the decisive evidence {position} in the document",
             6500),
}
EVIDENCE_POSITIONS = ["early", "in the middle", "late"]

STATE_FORMATS = {
    "text": "a single plain-text string",
    "ticket": 'an object {"ticket": {"channel": "...", "subject": "...", "body": "..."}}',
    "thread": 'a list of message objects [{"role": "customer"|"agent", "content": "..."}] '
              "(2-4 messages for short/medium, 5-9 for long)",
    "record": "a flat object of realistic fields (strings, numbers, booleans; more fields for longer tiers)",
}

CHOICE_WIDTHS = {"narrow": (3, 5), "medium": (6, 10), "wide": (11, 26)}
CHOICE_WIDTH_WEIGHTS = [0.5, 0.3, 0.2]
SCORE_LEVELS = (3, 10)

QUESTION_STYLES = {
    "direct": "a direct question about an explicit fact or classification",
    "inferential": "a question whose answer requires combining two separate facts in the state",
    "negated": "a question phrased with a negation (e.g. 'did the customer NOT receive...')",
    "quantitative": "a question hinging on a threshold, count, date, or amount in the state",
    "evidence": "a question about what the state explicitly states versus implies",
    "speech_act": "a question about what the writer is requesting, promising, or threatening",
}

AMBIGUITY = {
    "clear": "the evidence should point clearly to one answer",
    "ambiguous": "the evidence should be genuinely ambiguous between two answers, so a "
                 "well-calibrated judge would split probability rather than commit",
    "missing_evidence": "the state should NOT contain enough evidence to answer confidently; "
                        "the honest response is high uncertainty",
}

REGISTERS = {
    "clean": "clean professional prose",
    "informal": "informal writing with a few typos and abbreviations",
    "terse": "terse operational notes or log-like fragments",
    "ocr": "text with mild OCR artifacts (odd spacing, a few character substitutions)",
    "nonnative": "fluent but slightly non-native phrasing",
}

EXPECTED_SPECS = {
    "choice": "the exact option name from criteria",
    "score": "the integer level index (0-based)",
    "noul": "true or false (JSON booleans)",
}


def env_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} not found in environment or .env")


def fireworks_request(key: str, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        FIREWORKS_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            raise


def chat(key: str, model: str, prompt: str, temperature: float) -> str:
    response = fireworks_request(key, "/chat/completions", {
        "model": model,
        "temperature": temperature,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system",
             "content": "You write evaluation scenarios for a typed decision-making AI. "
                        "Respond with STRICT JSON only: no markdown fences, no commentary."},
            {"role": "user", "content": prompt},
        ],
    })
    return response["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in generator output")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in generator output")


def sample_cell(rng: random.Random) -> dict:
    primitive = rng.choice(["choice", "score", "noul"])
    cell = {
        "domain": rng.choice(DOMAINS),
        "length": rng.choice(list(LENGTH_TIERS)),
        "state_format": rng.choice(list(STATE_FORMATS)),
        "primitive": primitive,
        "style": rng.choice(list(QUESTION_STYLES)),
        "ambiguity": rng.choice(list(AMBIGUITY)),
        "register": rng.choice(list(REGISTERS)),
        "evidence_position": rng.choice(EVIDENCE_POSITIONS),
        "spice": rng.randrange(10**9),
    }
    if primitive == "choice":
        width = rng.choices(list(CHOICE_WIDTHS), weights=CHOICE_WIDTH_WEIGHTS)[0]
        lo, hi = CHOICE_WIDTHS[width]
        cell["n_options"] = rng.randint(lo, hi)
        cell["answer_slot"] = rng.randrange(cell["n_options"])
    elif primitive == "score":
        cell["n_levels"] = rng.randint(*SCORE_LEVELS)
        cell["answer_slot"] = rng.randrange(cell["n_levels"])
    else:
        cell["answer_slot"] = rng.choice([0, 1])  # 1 = yes
    return cell


def question_spec(cell: dict) -> str:
    if cell["primitive"] == "choice":
        return ('{"type": "choice", "instructions": "<the question>", "criteria": '
                f'{{"<option_name>": "<concrete description>", ...}}}} with EXACTLY '
                f'{cell["n_options"]} options, snake_case names, mutually distinguishable descriptions')
    if cell["primitive"] == "score":
        return ('{"type": "score", "instructions": "<the question>", "criteria": '
                f'["<level 0>", ...]}} with EXACTLY {cell["n_levels"]} ordered levels low to high; '
                "each level describes a concrete situation, never intensity words like "
                "mildly/moderately/very")
    return ('{"type": "noul", "instructions": "<one yes/no question>", '
            '"criteria": {"true": "<what yes means>", "false": "<what no means>"}}')


def answer_constraint(cell: dict) -> str:
    if cell["ambiguity"] != "clear":
        return ""
    if cell["primitive"] == "choice":
        return (f"Design the evidence so the correct answer is option number "
                f"{cell['answer_slot'] + 1} in your criteria listing.\n")
    if cell["primitive"] == "score":
        return f"Design the evidence so the correct level is index {cell['answer_slot']}.\n"
    return f"Design the evidence so the correct answer is {'yes' if cell['answer_slot'] else 'no'}.\n"


def scenario_prompt(cell: dict, pair: bool, n_questions: int) -> str:
    length_desc, _ = LENGTH_TIERS[cell["length"]]
    base = (
        f"Domain: {cell['domain']}.\n"
        f"State: {length_desc.format(position=cell['evidence_position'])}, formatted as "
        f"{STATE_FORMATS[cell['state_format']]}, written as {REGISTERS[cell['register']]}.\n"
        f"Question style: {QUESTION_STYLES[cell['style']]}.\n"
        f"Evidence design: {AMBIGUITY[cell['ambiguity']]}.\n"
        + answer_constraint(cell) +
        f"\nWrite a realistic scenario about a concrete, specific situation (plausible names, "
        f"amounts, dates; never placeholders like <NAME> or lorem). Each question is shaped "
        f"exactly as: {question_spec(cell)}\n"
        f"For each question also give `expected` ({EXPECTED_SPECS[cell['primitive']]}) and a "
        f"one-sentence `rationale`.\n"
        f"Variety seed: {cell['spice']}. Let it influence names, numbers, and the sub-topic, "
        f"so repeated calls differ.\n"
    )
    if pair:
        return base + (
            "\nCreate a CONTRASTIVE PAIR: `base` with one question, and `counterfactual` "
            "identical except for one changed fact (edit at most 8 words of the state) that "
            "flips the expected answer. Same question in both. Return: "
            '{"base": {"state": ..., "question": ..., "expected": ..., "rationale": ...}, '
            '"counterfactual": {same shape}, "changed_fact": "<what changed>"}'
        )
    if n_questions == 1:
        return base + '\nReturn: {"state": ..., "question": ..., "expected": ..., "rationale": ...}'
    return base + (
        f"\nWrite ONE state and {n_questions} DIFFERENT questions about it (same primitive, "
        f"distinct aspects; the answer-design constraint applies to the first question only). "
        'Return: {"state": ..., "questions": [{"question": ..., "expected": ..., '
        '"rationale": ...}, ...]}'
    )


def validate_question(raw_q: dict, expected, cell: dict) -> dict:
    question = Question(**raw_q)
    if question.type != cell["primitive"]:
        raise ValueError("primitive mismatch")
    if question.type == "choice":
        if expected not in question.criteria:
            raise ValueError("expected not among options")
        lo, hi = 3, 26
        if not lo <= len(question.criteria) <= hi:
            raise ValueError("choice option count out of range")
    elif question.type == "score":
        if not isinstance(expected, int) or isinstance(expected, bool) \
                or not 0 <= expected < len(question.criteria):
            raise ValueError("expected level out of range")
    elif not isinstance(expected, bool):
        raise ValueError("noul expected must be a boolean")
    return {"type": question.type, "instructions": question.instructions,
            "criteria": question.criteria}


def validate_state_for(cell: dict, state) -> None:
    validate_state(state)
    if len(dumps(state)) > LENGTH_TIERS[cell["length"]][1]:
        raise ValueError("state too long for its tier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="target number of scenario rows")
    ap.add_argument("--out", default="data/experiments/synthetic-v1/scenarios.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--pair-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--resume", action="store_true", help="skip ids already in --out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list-models", metavar="SUBSTRING",
                    help="print matching Fireworks model ids and exit")
    args = ap.parse_args()

    key = env_key(args.key_env)
    if args.list_models:
        listing = fireworks_request(key, "/models")
        for row in listing.get("data", []):
            if args.list_models.lower() in row.get("id", "").lower():
                print(row["id"])
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    seen_content: set[str] = set()
    if args.resume and out.exists():
        for line in out.open():
            row = json.loads(line)
            existing_ids.add(row["id"])
            seen_content.add(content_key(row["state"]))
    elif out.exists():
        raise SystemExit(f"{out} exists; use --resume or a new path")

    rng = random.Random(args.seed)
    kept, rejected = 0, {"json": 0, "schema": 0, "duplicate": 0, "api": 0}
    written = len(existing_ids)
    progress = tqdm(total=args.n, initial=written, desc="scenarios")

    def produce(job: dict) -> tuple[dict, dict | None, str | None]:
        """Worker: one generator call + validation. Returns (job, payload, reject_bucket)."""
        try:
            raw = extract_json(chat(key, args.model,
                                    scenario_prompt(job["cell"], job["pair"], job["n_questions"]),
                                    args.temperature))
            if job["pair"]:
                units = [(raw["base"]["state"], raw["base"]),
                         (raw["counterfactual"]["state"], raw["counterfactual"])]
            elif job["n_questions"] == 1:
                units = [(raw["state"], raw)]
            else:
                units = [(raw["state"], q) for q in raw.get("questions", [])[: job["n_questions"]]]
                if len(units) != job["n_questions"]:
                    raise ValueError("wrong question count")
            validated = []
            for state, item in units:
                validate_state_for(job["cell"], state)
                question = validate_question(item["question"], item["expected"], job["cell"])
                validated.append((state, question, item))
            return job, {"raw": raw, "validated": validated}, None
        except urllib.error.HTTPError:
            return job, None, "api"
        except (KeyError, TypeError, ValueError) as exc:
            bucket = "json" if isinstance(exc, ValueError) and "JSON" in str(exc) else "schema"
            return job, None, bucket

    index = 0

    def next_job() -> dict:
        nonlocal index
        while True:
            index += 1
            cell = sample_cell(rng)
            pair = rng.random() < args.pair_fraction
            n_questions = 1 if pair or cell["length"] == "short" else rng.choice([1, 1, 2, 3])
            suffixes = ["a", "b"] if pair else [f"q{i + 1}" for i in range(n_questions)]
            ids = [f"syn-{args.seed}-{index:05d}-{s}" for s in suffixes]
            if any(i in existing_ids for i in ids):
                continue  # resumed: this index was already produced
            return {"cell": cell, "pair": pair, "n_questions": n_questions,
                    "ids": ids, "group": f"syng-{args.seed}-{index:05d}"}

    with out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(produce, next_job()) for _ in range(args.workers)}
        while pending:
            finished = next(as_completed(pending))
            pending.discard(finished)
            job, payload, bucket = finished.result()
            if bucket is not None:
                rejected[bucket] += 1
                if rejected["api"] > 20:
                    raise SystemExit("too many API failures; stopping")
            else:
                validated = payload["validated"]
                new_states = {content_key(state) for state, _, _ in validated}
                if new_states & seen_content or (job["pair"] and len(new_states) == 1):
                    rejected["duplicate"] += 1
                else:
                    seen_content.update(new_states)
                    for scenario_id, (state, question, item) in zip(job["ids"], validated):
                        handle.write(json.dumps({
                            "format_version": SCENARIO_FORMAT,
                            "id": scenario_id,
                            "group_id": job["group"],
                            "cell": {k: v for k, v in job["cell"].items() if k != "spice"},
                            "state": state,
                            "question": question,
                            "generator": {
                                "model": args.model,
                                "temperature": args.temperature,
                                "expected": item["expected"],
                                "rationale": str(item.get("rationale", "")),
                                **({"changed_fact": str(payload["raw"].get("changed_fact", ""))}
                                   if job["pair"] else {}),
                            },
                        }) + "\n")
                        handle.flush()
                        written += 1
                        kept += 1
                        progress.update(1)
            if written < args.n:
                pending.add(pool.submit(produce, next_job()))
    progress.close()
    print(f"kept {kept} new scenario rows ({written} total) -> {out}; rejected: {rejected}")


if __name__ == "__main__":
    main()
