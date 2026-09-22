"""Versioned System One engine: restricted-logit readout on an MLX model.

No text is ever generated. Each question is rendered as a completion-style
prompt; the answer distribution is a softmax restricted to the lettered label
tokens at the final position. Independent execution is the stable default.
Shared-prefix execution is opt-in: native BF16 rounding can depend on batch shape.
"""

from __future__ import annotations

import math
import hashlib
import time
import threading
import resource
import sys
from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .schema import Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .calibration import (
    WORKLOAD_METHOD,
    build_workload_fits,
    label_to_target_index,
    temperature_scale,
)
from .confidence import confidence, SCHEMES, ADAPTER, ENTROPY
from .provenance import model_identity
from .data import sha256_file
from .limits import (
    EngineLimits,
    RequestValidationError,
    RequestLimitError,
    InferenceDeadlineExceeded,
)
from pathlib import Path
from .serialization import State, dumps, loads, validate_state
from .rendering import (
    LETTERS,
    as_yes_no_choice,
    label_token_ids,
    legacy_choice_prompt,
    legacy_score_prompt,
    render,
    resolve_renderer,
    resolve_readout,
    render_views,
    combine_views,
    LETTER_READOUT,
)

# Backward-compatible export for historical data scripts.
_LETTERS = LETTERS

# Content-free states for contextual calibration (Zhao et al. 2021,
# "Calibrate Before Use"): the model's answer distribution on these
# estimates its label prior, which is then divided out.
_CONTENT_FREE_STATES = ["N/A", "", "none"]

# minimum shared tokens for the KV-cache path to be worth it
_MIN_SHARED_PREFIX = 8



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
        limits: EngineLimits | None = None,
        mlx_cache_limit_bytes: int | None = None,
        workload_calibration_paths: list[str] | None = None,
        max_workload_calibrations: int = 16,
        min_calibration_examples: int = 25,
        max_calibration_examples: int = 256,
    ):
        for name, value in (
            ("max_workload_calibrations", max_workload_calibrations),
            ("min_calibration_examples", min_calibration_examples),
            ("max_calibration_examples", max_calibration_examples),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min_calibration_examples > max_calibration_examples:
            raise ValueError("min_calibration_examples exceeds the example cap")
        self.max_workload_calibrations = max_workload_calibrations
        self.min_calibration_examples = min_calibration_examples
        self.max_calibration_examples = max_calibration_examples
        if precision not in ("native", "float16", "float32"):
            raise ValueError("precision must be native, float16, or float32")
        if execution_mode not in ("independent", "shared"):
            raise ValueError("execution_mode must be independent or shared")
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if mlx_cache_limit_bytes is not None and (
            type(mlx_cache_limit_bytes) is not int or mlx_cache_limit_bytes < 0
        ):
            raise ValueError("MLX allocator cache limit must be a nonnegative integer")
        self.mlx_cache_limit_bytes = mlx_cache_limit_bytes
        if mlx_cache_limit_bytes is not None:
            mx.set_cache_limit(mlx_cache_limit_bytes)
        self.max_batch_size = max_batch_size
        self._owner_thread = threading.get_ident()
        self.limits = limits or EngineLimits()
        self.request_deadline = None
        self.readout_version = resolve_readout(readout_version, adapter_path)
        self.precision = precision
        self.execution_mode = execution_mode
        self.model_name = model_name
        self.contextual_calibration = contextual_calibration
        self.renderer_version = resolve_renderer(renderer_version, adapter_path)
        if (
            self.readout_version != LETTER_READOUT
            and self.renderer_version != "structured-v1"
        ):
            raise ValueError("candidate-v1 requires structured-v1")
        self.confidence_scheme = confidence_scheme or (
            ENTROPY if self.renderer_version == "legacy-v0" else ADAPTER
        )
        if self.confidence_scheme not in SCHEMES:
            raise ValueError("unknown confidence scheme")
        self.model, self.tokenizer, self.model_config = load(
            model_name, adapter_path=adapter_path, return_config=True
        )
        context_sizes = [
            self.model_config[key]
            for key in (
                "max_position_embeddings",
                "n_positions",
                "n_ctx",
                "context_length",
            )
            if type(self.model_config.get(key)) is int and self.model_config[key] > 0
        ]
        self.context_limit = min([self.limits.max_prompt_tokens, *context_sizes])
        if precision != "native":
            dtype = getattr(mx, precision)
            self.model.update(
                tree_map(
                    lambda x: (
                        x.astype(dtype) if mx.issubdtype(x.dtype, mx.floating) else x
                    ),
                    self.model.parameters(),
                )
            )
        self.model.eval()
        if precision != "native":
            mx.eval(self.model.parameters())
        self._label_cache: dict[str, int] = {}
        self._prior_cache: dict[str, list[float]] = {}
        self._token_cache: dict[str, list[int]] = {}
        self._input_tokens = 0
        self.backbone_identity = model_identity(model_name)
        self.adapter_digest = (
            sha256_file(Path(adapter_path) / "adapters.safetensors")
            if adapter_path
            else None
        )
        self._temperatures = {}
        self.temperature_path = temperature_path
        self._expected_prediction_config = {
            "max_batch_size": max_batch_size,
            "renderer_version": self.renderer_version,
            "readout_version": self.readout_version,
            "precision": precision,
            "execution_mode": execution_mode,
            "contextual_correction": contextual_calibration,
            "backbone_files": self.backbone_identity["files"],
            "adapter_sha256": self.adapter_digest,
        }
        if temperature_path:
            artifact = loads(Path(temperature_path).read_text())
            actual_config = dict(artifact.get("prediction_config", {}))
            # Earlier artifacts were produced by evaluators fixed at four views.
            actual_config.setdefault("max_batch_size", 4)
            if (
                artifact.get("format_version") != 1
                or artifact.get("method") != "per-primitive-temperature-v1"
                or actual_config != self._expected_prediction_config
            ):
                raise ValueError(
                    "temperature artifact does not match model/tokenizer/readout/runtime configuration"
                )
            for primitive, fit in artifact["fits"].items():
                if primitive not in ("choice", "noul", "score"):
                    raise ValueError("invalid calibration primitive")
                temperature_scale([0.5, 0.5], fit["temperature"])
                self._temperatures[primitive] = fit["temperature"]
        self.temperature_digest = (
            sha256_file(temperature_path) if temperature_path else None
        )
        self._workload_calibrations: dict[str, dict] = {}
        for path in workload_calibration_paths or []:
            self._register_workload_artifact(loads(Path(path).read_text()), path)
        self.implementation_hashes = {
            name: sha256_file(Path(__file__).parent / name)
            for name in (
                "engine.py",
                "rendering.py",
                "serialization.py",
                "schema.py",
                "calibration.py",
                "confidence.py",
            )
        }
        identity = dumps(
            [
                self.implementation_hashes,
                self.backbone_identity["files"],
                self.adapter_digest,
                self.renderer_version,
                self.readout_version,
                precision,
                execution_mode,
                max_batch_size,
                self.confidence_scheme,
                self.temperature_digest,
            ]
        )
        self.served_model_id = (
            model_name + "@" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        )

    def describe(self):
        limits = asdict(getattr(self, "limits", EngineLimits()))
        limits["max_prompt_tokens"] = getattr(
            self, "context_limit", limits["max_prompt_tokens"]
        )
        readout = getattr(self, "readout_version", LETTER_READOUT)
        limits["max_options"] = min(
            limits["max_options"], 26 if readout == LETTER_READOUT else 255
        )
        return {
            "id": getattr(self, "served_model_id", self.model_name),
            "object": "model",
            "owned_by": "local",
            "backbone": self.model_name,
            "backbone_revision": getattr(self, "backbone_identity", {}).get(
                "snapshot_revision"
            ),
            "adapter_sha256": getattr(self, "adapter_digest", None),
            "renderer_version": self.renderer_version,
            "readout_version": readout,
            "precision": getattr(self, "precision", "native"),
            "execution_mode": getattr(self, "execution_mode", "independent"),
            "shared_prefix_supported": self._head_projection() is not None,
            "confidence_scheme": getattr(self, "confidence_scheme", ENTROPY),
            "temperature_sha256": getattr(self, "temperature_digest", None),
            "workload_calibration": {
                "registered": {
                    name: artifact["sha256"]
                    for name, artifact in getattr(
                        self, "_workload_calibrations", {}
                    ).items()
                },
                "max_workloads": getattr(self, "max_workload_calibrations", 0),
                "min_examples": getattr(self, "min_calibration_examples", 0),
                "max_examples": getattr(self, "max_calibration_examples", 0),
            },
            "max_batch_size": getattr(self, "max_batch_size", 4),
            "limits": limits,
            "mlx_allocator_cache_limit_bytes": getattr(
                self, "mlx_cache_limit_bytes", None
            ),
            "implementation_sha256": getattr(self, "implementation_hashes", {}),
        }

    def runtime_stats(self):
        # Called by the owner thread only, after synchronized model readouts.
        if (
            getattr(self, "_owner_thread", threading.get_ident())
            != threading.get_ident()
        ):
            raise RuntimeError(
                "runtime statistics must be captured on the owner thread"
            )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_active_bytes": mx.get_peak_memory(),
            "mlx_allocator_cache_bytes": mx.get_cache_memory(),
            "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "prior_cache_entries": len(self._prior_cache),
            "token_cache_entries": len(self._token_cache),
            "model_input_tokens_processed": self._input_tokens,
        }

    def _check_deadline(self):
        if (
            getattr(self, "_owner_thread", threading.get_ident())
            != threading.get_ident()
        ):
            raise RuntimeError("engine must execute on the thread that loaded it")
        deadline = getattr(self, "request_deadline", None)
        if deadline is not None and time.monotonic() >= deadline:
            raise InferenceDeadlineExceeded("inference deadline exceeded")

    def _encode(self, prompt):
        self._check_deadline()
        cache = getattr(self, "_token_cache", None)
        if cache is None:
            self._token_cache = cache = {}
        if prompt in cache:
            tokens = cache.pop(prompt)
            cache[prompt] = tokens
            return tokens
        tokens = self.tokenizer.encode(prompt)
        limit = getattr(
            self,
            "context_limit",
            getattr(self, "limits", EngineLimits()).max_prompt_tokens,
        )
        if len(tokens) > limit:
            raise RequestLimitError(f"prompt exceeds {limit} tokens")
        capacity = getattr(self, "limits", EngineLimits()).max_token_cache_entries
        if capacity:
            cache[prompt] = tokens
            while len(cache) > capacity:
                cache.pop(next(iter(cache)))
        return tokens

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
        self._check_deadline()
        limit = getattr(self, "max_batch_size", 4)
        if len(items) > limit:
            return [
                row
                for start in range(0, len(items), limit)
                for row in self._score_batch(items[start : start + limit])
            ]
        token_lists = [self._encode(p) for p, _ in items]

        common = 0
        if (
            len(items) > 1
            and getattr(self, "execution_mode", "independent") == "shared"
            and self._head_projection() is not None
        ):
            limit = min(len(t) for t in token_lists) - 1  # keep suffixes non-empty
            while common < limit and len({t[common] for t in token_lists}) == 1:
                common += 1

        if common < _MIN_SHARED_PREFIX:
            out = []
            for tokens, (_, labels) in zip(token_lists, items):
                self._check_deadline()
                self._input_tokens += len(tokens)
                logits = self._last_logits(
                    mx.array(tokens)[None], mx.array([len(tokens) - 1])
                )[0]
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

    def _head_projection(self):
        # Other wrappers may apply logit caps/scales after their output head or
        # use non-KV caches. Do not bypass those transformations by duck typing.
        if getattr(self, "model_config", {}).get("model_type") not in (
            "llama",
            "qwen3",
            "smollm3",
        ):
            return None
        body = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        if head is None:
            head = getattr(getattr(body, "embed_tokens", None), "as_linear", None)
        return (body, head) if callable(body) and callable(head) else None

    def _last_logits(self, batch, last_idx, cache=None):
        """Project final positions only on verified families; otherwise use wrapper.

        Select the path before execution; never retry a mutated cache. Both
        independent and shared execution use the same final-position projection.
        """
        n = batch.shape[0]
        projection = self._head_projection()
        if projection is not None:
            body, head = projection
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

    def _prior_key(self, q: Question) -> str:
        return hashlib.sha256(
            dumps(
                [
                    self.renderer_version,
                    getattr(self, "readout_version", LETTER_READOUT),
                    asdict(q),
                ]
            ).encode()
        ).hexdigest()

    def _prior(self, q: Question) -> list[float]:
        """The model's label prior for this question, estimated as the mean
        answer distribution over content-free states. Cached per question."""
        key = self._prior_key(q)
        if key in self._prior_cache:
            value = self._prior_cache.pop(key)
            self._prior_cache[key] = value
            return value
        dists = [
            self._raw_distributions(cf, {"q": q})["q"] for cf in _CONTENT_FREE_STATES
        ]
        value = [sum(col) / len(dists) for col in zip(*dists)]
        capacity = getattr(self, "limits", EngineLimits()).max_prior_cache_entries
        if capacity:
            self._prior_cache[key] = value
            while len(self._prior_cache) > capacity:
                self._prior_cache.pop(next(iter(self._prior_cache)))
        return value

    def _apply_calibration(self, q: Question, probs: list[float]) -> list[float]:
        if not self.contextual_calibration:
            return probs
        prior = self._prior(q)
        adjusted = [p / max(pr, 1e-9) for p, pr in zip(probs, prior)]
        total = sum(adjusted)
        return [a / total for a in adjusted]

    # -- public API ------------------------------------------------------------

    def ask(
        self,
        state: State,
        questions: dict[str, Question],
        workload: str | None = None,
    ) -> dict[str, Answer]:
        selected = None
        if workload is not None:
            selected = getattr(self, "_workload_calibrations", {}).get(workload)
            if selected is None:
                raise RequestValidationError(
                    "unknown calibration workload; fit one via POST /v1/calibrations"
                )
        distributions = self._pretemperature_distributions(state, questions)
        answers: dict[str, Answer] = {}
        for qid, q in questions.items():
            probs = distributions[qid]
            if selected is not None and q.type in selected["fits"]:
                temperature = selected["fits"][q.type]["temperature"]
            else:
                # primitives without a workload fit keep the global temperature
                temperature = getattr(self, "_temperatures", {}).get(q.type)
            if temperature is not None:
                probs = temperature_scale(probs, temperature)
            answers[qid] = self._to_answer(
                q, probs, getattr(self, "confidence_scheme", ENTROPY)
            )
        return answers

    def _pretemperature_distributions(
        self, state: State, questions: dict[str, Question]
    ) -> dict[str, list[float]]:
        """Validated raw-plus-contextual distributions, before any temperature."""
        try:
            validate_state(state)
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a nonempty object")
            if any(
                not isinstance(key, str) or not isinstance(q, Question)
                for key, q in questions.items()
            ):
                raise ValueError("questions must map string IDs to Question objects")
        except (ValueError, TypeError) as exc:
            raise RequestValidationError(str(exc)) from exc
        limits = getattr(self, "limits", EngineLimits())
        if len(questions) > limits.max_questions:
            raise RequestLimitError(
                f"at most {limits.max_questions} questions are allowed"
            )
        if len(self._encode(dumps(state))) > limits.max_state_tokens:
            raise RequestLimitError(f"state exceeds {limits.max_state_tokens} tokens")
        self._request_views = self._request_rendered_tokens = 0
        try:
            self._preadmit(state, questions, limits)
            raw = self._raw_distributions(state, questions)
            return {
                qid: self._apply_calibration(q, raw[qid])
                for qid, q in questions.items()
            }
        finally:
            del self._request_views, self._request_rendered_tokens

    def _preadmit(self, state, questions, limits):
        """Reject an over-budget request before any model forward runs.

        Budget semantics are unchanged - cold contextual priors still share the
        request's view budget - but the decision moves ahead of the compute, so
        a request that cannot finish never pays for the passes it made first,
        and admission no longer depends on whether an earlier request happened
        to warm the prior cache mid-flight."""
        readout = getattr(self, "readout_version", LETTER_READOUT)
        prospective = 0
        for q in questions.values():
            try:
                mains = len(render_views(state, q, self.renderer_version, readout))
            except ValueError as exc:
                raise RequestValidationError(str(exc)) from exc
            prospective += mains
            if self.contextual_calibration and self._prior_key(q) not in self._prior_cache:
                # content-free prior passes render the same per-question view count
                prospective += mains * len(_CONTENT_FREE_STATES)
        if prospective > limits.max_views:
            raise RequestLimitError(
                f"request needs {prospective} model views (questions plus cold "
                f"calibration priors), over the {limits.max_views} limit; warm the "
                "priors, disable contextual correction, or raise max_views"
            )

    def _raw_distributions(self, state, questions):
        readout = getattr(self, "readout_version", LETTER_READOUT)
        limits = getattr(self, "limits", EngineLimits())
        views = {}
        token_count = getattr(self, "_request_rendered_tokens", 0)
        view_count = getattr(self, "_request_views", 0)
        for qid, q in questions.items():
            self._check_deadline()
            if len(q.answer_keys) > limits.max_options:
                raise RequestLimitError(
                    f"at most {limits.max_options} options are allowed"
                )
            try:
                items = render_views(state, q, self.renderer_version, readout)
            except ValueError as exc:
                raise RequestValidationError(str(exc)) from exc
            view_count += len(items)
            if view_count > limits.max_views:
                raise RequestLimitError(
                    f"request exceeds {limits.max_views} model views"
                )
            for prompt, _ in items:
                token_count += len(self._encode(prompt))
                if token_count > limits.max_request_tokens:
                    raise RequestLimitError(
                        f"request exceeds {limits.max_request_tokens} rendered tokens"
                    )
            views[qid] = items
        if hasattr(self, "_request_views"):
            self._request_views, self._request_rendered_tokens = view_count, token_count
        raw = iter(
            self._score_batch([item for items in views.values() for item in items])
        )
        return {
            qid: combine_views(questions[qid], [next(raw) for _ in items], readout)
            for qid, items in views.items()
        }

    @staticmethod
    def _to_answer(
        q: Question, probs: list[float], confidence_scheme: str = ENTROPY
    ) -> Answer:
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

    @staticmethod
    def _workload_name_ok(name) -> bool:
        return (
            isinstance(name, str)
            and 0 < len(name) <= 64
            and all(c.isalnum() or c in "-_.:" for c in name)
        )

    @staticmethod
    def _workload_artifact_digest(artifact: dict) -> str:
        body = {key: value for key, value in artifact.items() if key != "sha256"}
        return hashlib.sha256(dumps(body, sort_keys=True).encode()).hexdigest()

    def _register_workload_artifact(self, artifact: dict, path: str | None = None):
        """Admit a per-workload calibration artifact; every check fails closed."""
        origin = f" ({path})" if path else ""
        if (
            not isinstance(artifact, dict)
            or artifact.get("format_version") != 1
            or artifact.get("method") != WORKLOAD_METHOD
        ):
            raise ValueError(f"not a per-workload calibration artifact{origin}")
        if artifact.get("prediction_config") != self._expected_prediction_config:
            raise ValueError(
                "workload calibration does not match "
                f"model/tokenizer/readout/runtime configuration{origin}"
            )
        name = artifact.get("workload")
        if not self._workload_name_ok(name):
            raise ValueError(f"invalid workload name{origin}")
        if path is not None and name in self._workload_calibrations:
            raise ValueError(f"duplicate workload calibration: {name}{origin}")
        fits = artifact.get("fits")
        if not isinstance(fits, dict) or not fits:
            raise ValueError(f"workload calibration has no fits{origin}")
        for primitive, fit in fits.items():
            if primitive not in ("choice", "noul", "score"):
                raise ValueError(f"invalid calibration primitive{origin}")
            temperature_scale([0.5, 0.5], fit["temperature"])
        if artifact.get("sha256") != self._workload_artifact_digest(artifact):
            raise ValueError(f"workload calibration failed its integrity hash{origin}")
        if (
            name not in self._workload_calibrations
            and len(self._workload_calibrations) >= self.max_workload_calibrations
        ):
            raise ValueError(
                f"workload calibration capacity ({self.max_workload_calibrations}) reached"
            )
        self._workload_calibrations[name] = artifact

    def fit_workload(self, request: dict) -> dict:
        """Fit and register per-primitive temperatures for one named workload.

        Scores each labeled example through the exact serving path (raw plus
        contextual correction, before any temperature), fits scalar temperatures
        per primitive, and registers the artifact for `"calibration": name`
        requests. Diagnostics are in-sample and optimistic; the resampled
        evidence for the method is EXPERIMENTS section 15. Runs on the model
        thread and blocks other requests for its duration.
        """
        if not isinstance(request, dict):
            raise RequestValidationError("request must be an object")
        name = request.get("workload")
        if not self._workload_name_ok(name):
            raise RequestValidationError(
                "workload must be 1-64 characters of letters, digits, '-', '_', '.', ':'"
            )
        examples = request.get("examples")
        if not isinstance(examples, list) or not examples:
            raise RequestValidationError("examples must be a nonempty array")
        if len(examples) > self.max_calibration_examples:
            raise RequestLimitError(
                f"at most {self.max_calibration_examples} calibration examples are allowed"
            )
        if (
            name not in self._workload_calibrations
            and len(self._workload_calibrations) >= self.max_workload_calibrations
        ):
            raise RequestLimitError(
                f"workload calibration capacity ({self.max_workload_calibrations}) "
                "reached; refit an existing workload or restart with more capacity"
            )
        rows_by_primitive: dict[str, list[dict]] = {}
        canonical = []
        for position, item in enumerate(examples):
            if not isinstance(item, dict) or not {"state", "question", "label"} <= set(item):
                raise RequestValidationError(
                    f"examples[{position}] must be an object with state, question, and label"
                )
            spec = item["question"]
            if not isinstance(spec, dict):
                raise RequestValidationError(f"examples[{position}].question must be an object")
            try:
                question = Question(**spec)
                index = label_to_target_index(question, item["label"])
            except (ValueError, TypeError) as exc:
                raise RequestValidationError(f"examples[{position}]: {exc}") from exc
            self._check_deadline()
            probs = self._pretemperature_distributions(
                item["state"], {"q": question}
            )["q"]
            target = [0.0] * len(question.answer_keys)
            target[index] = 1.0
            rows_by_primitive.setdefault(question.type, []).append(
                {"probabilities": probs, "target": target}
            )
            canonical.append(
                [
                    item["state"],
                    {
                        "type": question.type,
                        "instructions": question.instructions,
                        "criteria": question.criteria,
                    },
                    item["label"],
                ]
            )
        try:
            fits, diagnostics = build_workload_fits(
                rows_by_primitive, self.min_calibration_examples
            )
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from exc
        artifact = {
            "format_version": 1,
            "method": WORKLOAD_METHOD,
            "workload": name,
            "created_unix": int(time.time()),
            "serving_model_id": self.served_model_id,
            "prediction_config": self._expected_prediction_config,
            "examples": {
                "count": len(examples),
                "sha256": hashlib.sha256(dumps(canonical).encode()).hexdigest(),
            },
            "fits": fits,
            "diagnostics": diagnostics,
            "limitations": (
                "In-sample diagnostics on this workload's labeled sample; valid "
                "only for this serving identity and workload. An at-bound or "
                "insufficient primitive is diagnosed, never applied. A scalar "
                "temperature cannot repair rank errors (verdict "
                "structural_warning): widen escalation instead."
            ),
        }
        artifact["sha256"] = self._workload_artifact_digest(artifact)
        self._register_workload_artifact(artifact)
        return artifact

    def respond(self, request: dict) -> dict:
        """Serve a Jev-shaped request and return a Jev-shaped response.

        Request:  {"state": JSON, "questions": {id: {type, instructions, criteria?}}}
        Response: {"model", "answers": {id: {...}}, "usage": {...}}

        Successful calls return one typed answer per question ID, without parsing
        generated text. Invalid requests or model execution errors still fail.
        """
        if not isinstance(request, dict) or "state" not in request:
            raise RequestValidationError("request must be an object containing state")
        if request.get("model") not in (
            None,
            "jev-latest",
            self.model_name,
            getattr(self, "served_model_id", self.model_name),
        ):
            raise RequestValidationError("unknown model; see GET /v1/models")
        specs = request.get("questions")
        if not isinstance(specs, dict) or not specs:
            raise RequestValidationError("questions must be a nonempty object")
        if len(specs) > getattr(self, "limits", EngineLimits()).max_questions:
            raise RequestLimitError("too many questions")
        if any(not isinstance(spec, dict) for spec in specs.values()):
            raise RequestValidationError("each question must be an object")
        try:
            questions = {qid: Question(**spec) for qid, spec in specs.items()}
        except (ValueError, TypeError) as exc:
            raise RequestValidationError(str(exc)) from exc
        workload = request.get("calibration")
        if workload is not None and not isinstance(workload, str):
            raise RequestValidationError("calibration must be a workload name")
        start_tokens = self._input_tokens
        answers = self.ask(request["state"], questions, workload=workload)
        response = {
            "model": getattr(self, "served_model_id", self.model_name),
            "answers": {qid: asdict(a) for qid, a in answers.items()},
            "usage": {
                "input_tokens": self._input_tokens - start_tokens,
                "output_tokens": 0,
            },
        }
        if workload is not None:
            response["calibration"] = {
                "workload": workload,
                "sha256": self._workload_calibrations[workload]["sha256"],
            }
        return response
