"""v0 System One engine: single-forward-pass logit readout on an MLX model.

No text is ever generated. Each question is rendered as a completion-style
prompt; the answer distribution is a softmax restricted to the lettered label
tokens at the final position. Multi-question requests share one KV-cache pass
over the common token prefix (state + template header), Nimble-style.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .schema import Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer

_LETTERS = [chr(ord("A") + i) for i in range(26)]

# Content-free states for contextual calibration (Zhao et al. 2021,
# "Calibrate Before Use"): the model's answer distribution on these
# estimates its label prior, which is then divided out.
_CONTENT_FREE_STATES = ["N/A", "", "none"]

# minimum shared tokens for the KV-cache path to be worth it
_MIN_SHARED_PREFIX = 8


def _confidence(probs: list[float]) -> float:
    """1 - normalized entropy: 1.0 for a single peak, 0.0 for uniform."""
    k = len(probs)
    if k < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - entropy / math.log(k))


class SystemOneEngine:
    def __init__(
        self,
        model_name: str,
        contextual_calibration: bool = False,
        adapter_path: str | None = None,
    ):
        self.model_name = model_name
        self.contextual_calibration = contextual_calibration
        self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        self._label_cache: dict[str, int] = {}
        self._prior_cache: dict[str, list[float]] = {}
        self._input_tokens = 0

    # -- token-level readout --------------------------------------------------

    def _label_token_id(self, label: str) -> int:
        if label not in self._label_cache:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) > 1:
                warnings.warn(f"label {label!r} is {len(ids)} tokens; using the first")
            self._label_cache[label] = ids[0]
        return self._label_cache[label]

    def _probs_from_logits(self, logits: mx.array, labels: list[str]) -> list[float]:
        ids = [self._label_token_id(l) for l in labels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"label tokens collide for {labels}")
        restricted = mx.take(logits, mx.array(ids)).astype(mx.float32)
        return mx.softmax(restricted).tolist()

    def _score_batch(self, items: list[tuple[str, list[str]]]) -> list[list[float]]:
        """Restricted answer distributions for several prompts. Prompts that
        share a token prefix (same state, same template) are scored with one
        pass over that prefix plus ONE batched pass over all suffixes (prefix
        KV tiled across the batch). Questions stay strictly independent: each
        row's context is exactly prefix + its own suffix."""
        token_lists = [self.tokenizer.encode(p) for p, _ in items]

        common = 0
        if len(items) > 1:
            limit = min(len(t) for t in token_lists) - 1  # keep suffixes non-empty
            while common < limit and len({t[common] for t in token_lists}) == 1:
                common += 1

        if common < _MIN_SHARED_PREFIX:
            out = []
            for tokens, (_, labels) in zip(token_lists, items):
                self._input_tokens += len(tokens)
                logits = self.model(mx.array(tokens)[None])[0, -1, :]
                out.append(self._probs_from_logits(logits, labels))
            return out

        # one pass over the shared prefix, then ALL suffixes in one batched
        # forward: tile the prefix KV cache across the batch dimension and
        # read each row's logits at its own last real token. Near-flat
        # latency in the number of questions.
        n = len(items)
        cache = make_prompt_cache(self.model)
        self.model(mx.array(token_lists[0][:common])[None], cache=cache)
        self._input_tokens += common
        if n > 1:
            for layer_cache in cache:
                layer_cache.keys = mx.repeat(layer_cache.keys, n, axis=0)
                layer_cache.values = mx.repeat(layer_cache.values, n, axis=0)

        suffixes = [tokens[common:] for tokens in token_lists]
        self._input_tokens += sum(len(s) for s in suffixes)
        max_len = max(len(s) for s in suffixes)
        pad_id = self.tokenizer.eos_token_id or 0
        # right padding is safe under causal attention: each row's readout
        # position (its last real token) never attends to later pad tokens
        batch = mx.array([s + [pad_id] * (max_len - len(s)) for s in suffixes])
        last_idx = mx.array([len(s) - 1 for s in suffixes])
        try:
            # LM head only at each row's readout position: avoids materializing
            # the (n, max_len, vocab) logits tensor, which dominates latency
            hidden = self.model.model(batch, cache=cache)  # (n, max_len, H)
            sel = hidden[mx.arange(n), last_idx]  # (n, H)
            head = getattr(self.model, "lm_head", None)
            logits = (
                head(sel)
                if head is not None
                else self.model.model.embed_tokens.as_linear(sel)
            )
        except AttributeError:  # unfamiliar architecture: full logits fallback
            logits = self.model(batch, cache=cache)[mx.arange(n), last_idx]
        return [
            self._probs_from_logits(logits[i], labels)
            for i, (_, labels) in enumerate(items)
        ]

    # -- prompt builders ------------------------------------------------------

    @staticmethod
    def _choice_prompt(state: str, q: Question) -> tuple[str, list[str]]:
        options = list(q.criteria.keys())
        lines = []
        for letter, name in zip(_LETTERS, options):
            desc = q.criteria[name]
            lines.append(f"{letter}. {name}" + (f": {desc}" if desc else ""))
        prompt = (
            "Read the state, then answer the question by choosing the single best option.\n\n"
            f"State:\n{state}\n\n"
            f"Question: {q.instructions}\n\n"
            "Options:\n" + "\n".join(lines) + "\n\n"
            "The best option is"
        )
        return prompt, options

    @staticmethod
    def _score_prompt(state: str, q: Question) -> str:
        # Levels are lettered for the readout (single-token labels in BPE
        # vocabs, unlike " 0"), then mapped back to level numbers.
        lines = [f"{_LETTERS[i]}. {desc}" for i, desc in enumerate(q.criteria)]
        return (
            "Read the state, then rate it on the scale below. "
            "Pick the level whose description fits best.\n\n"
            f"State:\n{state}\n\n"
            f"Question: {q.instructions}\n\n"
            "Levels:\n" + "\n".join(lines) + "\n\n"
            "The best-fitting level is"
        )

    @staticmethod
    def _as_yes_no_choice(q: Question) -> Question:
        """Noul is answered through the choice template: on base models the
        lettered readout separates classes far better than a bare yes/no
        completion, which shows a strong acquiescence bias (measured on
        SST-2: 0.535 acc / 0.297 ECE bare vs 0.695 / 0.088 lettered)."""
        crit = q.criteria if isinstance(q.criteria, dict) else {}
        return Question(
            type="choice",
            instructions=q.instructions,
            criteria={"no": crit.get("false"), "yes": crit.get("true")},
        )

    def _render(self, state: str, q: Question) -> tuple[str, list[str]]:
        """Prompt plus readout labels. Noul renders as A=no, B=yes, so its
        raw distribution is [p_no, p_yes]."""
        if q.type == "score":
            prompt = self._score_prompt(state, q)
            n = len(q.criteria)
        elif q.type == "noul":
            prompt, _ = self._choice_prompt(state, self._as_yes_no_choice(q))
            n = 2
        else:
            prompt, _ = self._choice_prompt(state, q)
            n = len(q.criteria)
        return prompt, [f" {_LETTERS[i]}" for i in range(n)]

    # -- calibration -----------------------------------------------------------

    def _prior(self, q: Question) -> list[float]:
        """The model's label prior for this question, estimated as the mean
        answer distribution over content-free states. Cached per question."""
        key = f"{q.type}|{q.instructions}|{q.criteria!r}"
        if key not in self._prior_cache:
            dists = self._score_batch([self._render(cf, q) for cf in _CONTENT_FREE_STATES])
            self._prior_cache[key] = [sum(col) / len(dists) for col in zip(*dists)]
        return self._prior_cache[key]

    def _apply_calibration(self, q: Question, probs: list[float]) -> list[float]:
        if not self.contextual_calibration:
            return probs
        prior = self._prior(q)
        adjusted = [p / max(pr, 1e-9) for p, pr in zip(probs, prior)]
        total = sum(adjusted)
        return [a / total for a in adjusted]

    # -- public API ------------------------------------------------------------

    def ask(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        qids = list(questions)
        items = [self._render(state, questions[qid]) for qid in qids]
        raw = self._score_batch(items)
        answers: dict[str, Answer] = {}
        for qid, probs in zip(qids, raw):
            q = questions[qid]
            probs = self._apply_calibration(q, probs)
            answers[qid] = self._to_answer(q, probs)
        return answers

    @staticmethod
    def _to_answer(q: Question, probs: list[float]) -> Answer:
        if q.type == "choice":
            dist = dict(zip(q.criteria.keys(), probs))
            return ChoiceAnswer(
                choice=max(dist, key=dist.get),
                probabilities=dist,
                confidence=_confidence(probs),
            )
        if q.type == "score":
            return ScoreAnswer(
                score=sum(i * p for i, p in enumerate(probs)),
                probabilities={str(i): p for i, p in enumerate(probs)},
                legend={str(i): desc for i, desc in enumerate(q.criteria)},
                confidence=_confidence(probs),
            )
        return NoulAnswer(noul=probs[1])  # render order is [no, yes]

    def respond(self, request: dict) -> dict:
        """Serve a Jev-shaped request and return a Jev-shaped response.

        Request:  {"state": str, "questions": {id: {type, instructions, criteria?}}}
        Response: {"model", "answers": {id: {...}}, "usage": {...}}

        Every question ID in the request appears in `answers` — the readout
        cannot skip or fail to parse an answer by construction.
        """
        questions = {
            qid: Question(**spec) for qid, spec in request["questions"].items()
        }
        start_tokens = self._input_tokens
        answers = self.ask(request["state"], questions)
        return {
            "model": request.get("model", self.model_name),
            "answers": {qid: asdict(a) for qid, a in answers.items()},
            "usage": {
                "input_tokens": self._input_tokens - start_tokens,
                "output_tokens": 0,
            },
        }
