"""Versioned System One engine: restricted-logit readout on an MLX model.

No text is ever generated. Each question is rendered as a completion-style
prompt; the answer distribution is a softmax restricted to the lettered label
tokens at the final position. Multi-question requests share one KV-cache pass
over the common token prefix (state + template header), Nimble-style.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .schema import Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .serialization import State, dumps, validate_state
from .rendering import (
    LETTERS, as_yes_no_choice, label_token_ids, legacy_choice_prompt,
    legacy_score_prompt, render, resolve_renderer,
)

# Backward-compatible export for historical data scripts.
_LETTERS = LETTERS

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
        renderer_version: str | None = None,
    ):
        self.model_name = model_name
        self.contextual_calibration = contextual_calibration
        self.renderer_version = resolve_renderer(renderer_version, adapter_path)
        self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        self._label_cache: dict[str, int] = {}
        self._prior_cache: dict[str, list[float]] = {}
        self._input_tokens = 0

    # -- token-level readout --------------------------------------------------

    def _label_token_id(self, label: str) -> int:
        if label not in self._label_cache:
            self._label_cache[label] = label_token_ids(self.tokenizer, [label])[0]
        return self._label_cache[label]

    def _probs_from_logits(self, logits: mx.array, labels: list[str]) -> list[float]:
        ids = [self._label_token_id(label) for label in labels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"label tokens collide for {labels}")
        restricted = mx.take(logits, mx.array(ids)).astype(mx.float32)
        probs = mx.softmax(restricted).tolist()
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
            raise ValueError("model produced invalid probabilities")
        return probs

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
        # read each row's logits at its own last real token. Prefix memory is
        # physically copied; latency and memory scaling still need measurement.
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
        # Select the execution path BEFORE mutating suffix caches. Never retry a
        # partially executed forward against a cache it may already have advanced.
        body = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        if head is None:
            embedding = getattr(body, "embed_tokens", None)
            head = getattr(embedding, "as_linear", None)
        if callable(body) and callable(head):
            hidden = body(batch, cache=cache)
            sel = hidden[mx.arange(n), last_idx]
            logits = head(sel)
        else:
            logits = self.model(batch, cache=cache)[mx.arange(n), last_idx]
        return [
            self._probs_from_logits(logits[i], labels)
            for i, (_, labels) in enumerate(items)
        ]

    # -- prompt builders ------------------------------------------------------

    # Compatibility aliases for historical data-building scripts. New pipelines
    # import rendering directly and never need MLX just to prepare examples.
    _choice_prompt = staticmethod(legacy_choice_prompt)
    _score_prompt = staticmethod(legacy_score_prompt)
    _as_yes_no_choice = staticmethod(as_yes_no_choice)

    def _render(self, state: State, q: Question) -> tuple[str, list[str]]:
        return render(state, q, self.renderer_version)

    # -- calibration -----------------------------------------------------------

    def _prior(self, q: Question) -> list[float]:
        """The model's label prior for this question, estimated as the mean
        answer distribution over content-free states. Cached per question."""
        key = dumps([self.renderer_version, asdict(q)])
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

    def ask(self, state: State, questions: dict[str, Question]) -> dict[str, Answer]:
        validate_state(state)
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        if any(not isinstance(key, str) or not isinstance(q, Question) for key, q in questions.items()):
            raise ValueError("questions must map string IDs to Question objects")
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

        Request:  {"state": JSON, "questions": {id: {type, instructions, criteria?}}}
        Response: {"model", "answers": {id: {...}}, "usage": {...}}

        Successful calls return one typed answer per question ID, without parsing
        generated text. Invalid requests or model execution errors still fail.
        """
        if not isinstance(request, dict) or "state" not in request:
            raise ValueError("request must be an object containing state")
        specs = request.get("questions")
        if not isinstance(specs, dict) or not specs:
            raise ValueError("questions must be a nonempty object")
        if any(not isinstance(spec, dict) for spec in specs.values()):
            raise ValueError("each question must be an object")
        questions = {qid: Question(**spec) for qid, spec in specs.items()}
        start_tokens = self._input_tokens
        answers = self.ask(request["state"], questions)
        return {
            "model": self.model_name,
            "answers": {qid: asdict(a) for qid, a in answers.items()},
            "usage": {
                "input_tokens": self._input_tokens - start_tokens,
                "output_tokens": 0,
            },
        }
