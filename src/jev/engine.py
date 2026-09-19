"""Versioned System One engine: restricted-logit readout on an MLX model.

No text is ever generated. Each question is rendered as a completion-style
prompt; the answer distribution is a softmax restricted to the lettered label
tokens at the final position. Independent execution is the stable default.
Shared-prefix execution is opt-in: native BF16 rounding can depend on batch shape.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .schema import Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .calibration import temperature_scale
from .confidence import confidence, SCHEMES, ADAPTER, ENTROPY
from .provenance import model_identity
from .data import sha256_file
from pathlib import Path
from .serialization import State, dumps, loads, validate_state
from .rendering import (
    LETTERS, as_yes_no_choice, label_token_ids, legacy_choice_prompt,
    legacy_score_prompt, render, resolve_renderer, resolve_readout, render_views,
    combine_views, LETTER_READOUT,
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
        precision: str = "native",
        execution_mode: str = "independent",
        readout_version: str | None = None,
        max_batch_size: int = 4,
        temperature_path: str | None = None,
        confidence_scheme: str | None = None,
    ):
        if precision not in ("native", "float16", "float32"):
            raise ValueError("precision must be native, float16, or float32")
        if execution_mode not in ("independent", "shared"):
            raise ValueError("execution_mode must be independent or shared")
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.readout_version = resolve_readout(readout_version, adapter_path)
        self.precision = precision
        self.execution_mode = execution_mode
        self.model_name = model_name
        self.contextual_calibration = contextual_calibration
        self.renderer_version = resolve_renderer(renderer_version, adapter_path)
        if self.readout_version != LETTER_READOUT and self.renderer_version != "structured-v1":
            raise ValueError("candidate-v1 requires structured-v1")
        self.confidence_scheme = confidence_scheme or (ENTROPY if self.renderer_version == "legacy-v0" else ADAPTER)
        if self.confidence_scheme not in SCHEMES:
            raise ValueError("unknown confidence scheme")
        self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        if precision != "native":
            dtype = getattr(mx, precision)
            self.model.update(tree_map(
                lambda x: x.astype(dtype) if mx.issubdtype(x.dtype, mx.floating) else x,
                self.model.parameters(),
            ))
        self.model.eval()
        self._label_cache: dict[str, int] = {}
        self._prior_cache: dict[str, list[float]] = {}
        self._input_tokens = 0
        self._temperatures = {}
        self.temperature_path = temperature_path
        if temperature_path:
            artifact = loads(Path(temperature_path).read_text())
            expected = {
                "renderer_version": self.renderer_version, "readout_version": self.readout_version,
                "precision": precision, "execution_mode": execution_mode,
                "contextual_correction": contextual_calibration,
                "backbone_files": model_identity(model_name)["files"],
                "adapter_sha256": sha256_file(Path(adapter_path)/"adapters.safetensors") if adapter_path else None,
            }
            if artifact.get("format_version") != 1 or artifact.get("method") != "per-primitive-temperature-v1" or artifact.get("prediction_config") != expected:
                raise ValueError("temperature artifact does not match model/tokenizer/readout/runtime configuration")
            for primitive, fit in artifact["fits"].items():
                if primitive not in ("choice", "noul", "score"):
                    raise ValueError("invalid calibration primitive")
                temperature_scale([.5,.5], fit["temperature"])
                self._temperatures[primitive] = fit["temperature"]

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
        """Restricted distributions with independent or opt-in shared execution.

        Shared execution tiles prefix KV; each row sees only its own suffix.
        This isolates information, not low-precision rounding. Independent mode
        gives a prompt the same computation regardless of other questions.
        """
        limit = getattr(self, "max_batch_size", 4)
        if len(items) > limit:
            return [row for start in range(0, len(items), limit)
                    for row in self._score_batch(items[start:start+limit])]
        token_lists = [self.tokenizer.encode(p) for p, _ in items]

        common = 0
        if len(items) > 1 and getattr(self, "execution_mode", "independent") == "shared":
            limit = min(len(t) for t in token_lists) - 1  # keep suffixes non-empty
            while common < limit and len({t[common] for t in token_lists}) == 1:
                common += 1

        if common < _MIN_SHARED_PREFIX:
            out = []
            for tokens, (_, labels) in zip(token_lists, items):
                self._input_tokens += len(tokens)
                logits = self._last_logits(mx.array(tokens)[None], mx.array([len(tokens) - 1]))[0]
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
        logits = self._last_logits(batch, last_idx, cache=cache)
        return [
            self._probs_from_logits(logits[i], labels)
            for i, (_, labels) in enumerate(items)
        ]

    def _last_logits(self, batch, last_idx, cache=None):
        """Project only final positions on supported dense Llama/Qwen backbones.

        Select the path before execution; never retry a mutated cache. Both
        independent and shared execution use the same final-position projection.
        """
        n = batch.shape[0]
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
        return logits

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
            dists = [self._raw_distributions(cf, {"q": q})["q"] for cf in _CONTENT_FREE_STATES]
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
        raw = self._raw_distributions(state, questions)
        answers: dict[str, Answer] = {}
        for qid in qids:
            probs = raw[qid]
            q = questions[qid]
            probs = self._apply_calibration(q, probs)
            temperature = getattr(self, "_temperatures", {}).get(q.type)
            if temperature is not None:
                probs = temperature_scale(probs, temperature)
            answers[qid] = self._to_answer(q, probs, getattr(self, "confidence_scheme", ENTROPY))
        return answers

    def _raw_distributions(self, state, questions):
        readout = getattr(self, "readout_version", LETTER_READOUT)
        views = {qid: render_views(state, q, self.renderer_version, readout) for qid, q in questions.items()}
        raw = iter(self._score_batch([item for items in views.values() for item in items]))
        return {qid: combine_views(questions[qid], [next(raw) for _ in items], readout)
                for qid, items in views.items()}

    @staticmethod
    def _to_answer(q: Question, probs: list[float], confidence_scheme: str = ENTROPY) -> Answer:
        if q.type == "choice":
            dist = dict(zip(q.criteria.keys(), probs))
            return ChoiceAnswer(
                choice=max(dist, key=dist.get),
                probabilities=dist,
                confidence=confidence(probs, "choice", confidence_scheme),
            )
        if q.type == "score":
            return ScoreAnswer(
                score=sum(i * p for i, p in enumerate(probs)),
                probabilities={str(i): p for i, p in enumerate(probs)},
                legend={str(i): desc for i, desc in enumerate(q.criteria)},
                confidence=confidence(probs, "score", confidence_scheme),
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
