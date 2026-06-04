# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColBERT query/document encoding for vLLM pooling (PyLate-compatible)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from vllm.inputs import EngineInput
from vllm.inputs.engine import TokensInput
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.inputs import EngineInput, TextPrompt, TokensPrompt
    from vllm.renderers import TokenizeParams
    from vllm.renderers.tokenizer import TokenizerLike

logger = init_logger(__name__)

ColBERTEmbeddingMode = Literal["query", "document"]

# ColBERT ``colbert_attention_mask`` on EngineInput is metadata for parity checks
# and observability only; vLLM pooling ColBERT forwards do not consume it.
COLBERT_ATTENTION_MASK_ENFORCED_IN_FORWARD = False

COLBERT_MODEL_ARCHITECTURES = frozenset(
    {
        "HF_ColBERT",
        "ColBERTModel",
        "ColBERTModernBertModel",
        "ColBERTJinaRobertaModel",
        "ColBERTLfm2Model",
    }
)

DEFAULT_QUERY_PREFIX = "[unused0]"
DEFAULT_DOCUMENT_PREFIX = "[unused1]"
JINA_QUERY_PREFIX = "[QueryMarker]"
JINA_DOCUMENT_PREFIX = "[DocumentMarker]"


def is_colbert_pooling_model(model_config: ModelConfig) -> bool:
    """Return True if the served model is a ColBERT late-interaction encoder."""
    arch = model_config.architecture
    if arch in COLBERT_MODEL_ARCHITECTURES:
        return True
    hf_archs = getattr(model_config.hf_config, "architectures", None) or []
    return any(a in COLBERT_MODEL_ARCHITECTURES for a in hf_archs)


def colbert_parity_check_enabled() -> bool:
    return os.getenv("VLLM_COLBERT_PARITY_CHECK", "0") == "1"


def colbert_strict_mode_enabled() -> bool:
    return os.getenv("VLLM_COLBERT_STRICT_MODE", "0") == "1"


def is_colbert_late_interaction(model_config: ModelConfig) -> bool:
    """Return True if the model supports ColBERT late-interaction encoding."""
    return is_colbert_pooling_model(model_config)


def validate_colbert_embedding_mode(
    model_config: ModelConfig,
    embedding_mode: ColBERTEmbeddingMode | None,
) -> None:
    """Reject embedding_mode on non-ColBERT late-interaction models."""
    if embedding_mode is None:
        return
    if not is_colbert_late_interaction(model_config):
        raise ValueError(
            "embedding_mode is only supported for ColBERT / late interaction models"
        )


def _warn_colbert_attention_mask_not_enforced() -> None:
    if colbert_strict_mode_enabled():
        logger.warning_once(
            "ColBERT attention_mask is not enforced in transformer forward; "
            "running in validation mode only."
        )


def _resolve_token_id(tokenizer: TokenizerLike, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unk_id is not None and token_id == unk_id):
        raise ValueError(
            f"ColBERT marker token {token!r} is not in the tokenizer vocabulary."
        )
    return int(token_id)


def _load_artifact_metadata(model: str, revision: str | None) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return {}

    try:
        path = hf_hub_download(model, filename="artifact.metadata", revision=revision)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_st_colbert_config(model: str, revision: str | None) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return {}

    for filename in ("config_sentence_transformers.json", "modules.json"):
        try:
            path = hf_hub_download(model, filename=filename, revision=revision)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if filename == "modules.json" and isinstance(data, list):
                for entry in data:
                    if entry.get("type") == "sentence_transformers.models.ColBERT":
                        return entry.get("kwargs", {})
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def resolve_colbert_chat_template_hash(
    chat_template: str | None,
    *,
    model_config: ModelConfig | None = None,
) -> str:
    """SHA-256 hash of chat template (or model fallback) for cache keys."""
    if chat_template is not None:
        return hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if model_config is not None:
        payload = f"{model_config.model}\0{model_config.revision or ''}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashlib.sha256(b"none").hexdigest()


def resolve_colbert_chat_template_version(
    chat_template: str | None,
    *,
    model_config: ModelConfig | None = None,
) -> str:
    """Deprecated alias for :func:`resolve_colbert_chat_template_hash`."""
    return resolve_colbert_chat_template_hash(chat_template, model_config=model_config)


def compute_colbert_config_hash(
    config: ColBERTEncodingConfig,
    *,
    model_config: ModelConfig | None = None,
) -> str:
    """Hash ColBERT tokenization settings that affect encoded token sequences."""
    parts = [
        config.query_prefix,
        config.document_prefix,
        str(config.query_length),
        str(config.document_length),
        str(config.mask_token_id),
        str(config.pad_token_id),
        str(config.do_query_expansion),
        str(config.attend_to_expansion_tokens),
    ]
    if model_config is not None:
        parts.extend([model_config.model, model_config.revision or ""])
    payload = "\0".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonicalize_colbert_input_text(
    text: str,
    tok_params: ColBERTTokenizeParams,
    tokenizer: TokenizerLike,
) -> str:
    """Return the exact string passed to ``tokenizer.encode`` for ColBERT paths.

    Cache keys must use this value (not the raw API string) so prefix-cache
    isolation matches tokenization when the renderer applies ``do_lower_case`` or
    other pre-tokenization validators.
    """

    prompt: TextPrompt = {"prompt": text}
    prepared = tok_params.apply_pre_tokenization(tokenizer, prompt)
    return prepared["prompt"]


def compute_colbert_cache_key(
    *,
    model_name: str,
    embedding_mode: ColBERTEmbeddingMode | None,
    raw_text: str,
    chat_template_hash: str,
    colbert_config_hash: str | None = None,
) -> str:
    """Canonical prefix-cache isolation key for ColBERT renderer paths.

    ``raw_text`` must be the post-``apply_pre_tokenization`` prompt string (see
    :func:`canonicalize_colbert_input_text`), not the unnormalized API payload.
    ColBERT marker/truncate/pad steps run after tokenization and are represented
    in ``prompt_token_ids`` plus ``embedding_mode`` / ``colbert_config_hash``.
    """
    # NOTE: cache key is defined over canonical pre-tokenization text to match
    # the tokenizer input contract (not over token ids).
    config_hash = colbert_config_hash if embedding_mode is not None else "none"
    mode_part = "" if embedding_mode is None else embedding_mode
    payload = "\0".join(
        [model_name, mode_part, raw_text, chat_template_hash, config_hash]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_colbert_cache_salt(
    embedding_mode: ColBERTEmbeddingMode,
    input_text: str,
    *,
    model_name: str = "",
    chat_template_hash: str | None = None,
    chat_template_version: str | None = None,
    colbert_config_hash: str | None = None,
    request_id: str | None = None,  # noqa: ARG001 — deprecated, ignored
) -> str:
    """Backward-compatible alias for :func:`compute_colbert_cache_key`."""
    del request_id
    template_hash = chat_template_hash
    if template_hash is None:
        if chat_template_version is not None:
            template_hash = hashlib.sha256(
                chat_template_version.encode("utf-8")
            ).hexdigest()
        else:
            template_hash = resolve_colbert_chat_template_hash(None)
    return compute_colbert_cache_key(
        model_name=model_name,
        embedding_mode=embedding_mode,
        raw_text=input_text,
        chat_template_hash=template_hash,
        colbert_config_hash=colbert_config_hash,
    )


def build_colbert_attention_mask(
    token_ids: list[int],
    mode: ColBERTEmbeddingMode,
    config: ColBERTEncodingConfig,
) -> list[int]:
    """Build per-token attention mask metadata for ColBERT engine inputs.

    This mask is **not** consumed by the vLLM ColBERT transformer forward pass
    (see :data:`COLBERT_ATTENTION_MASK_ENFORCED_IN_FORWARD`). Correctness relies
    on preprocessing (markers, truncation, expansion) and ranking parity tests.
    The mask is attached for validation, observability, and PyLate parity tooling.
    """
    _warn_colbert_attention_mask_not_enforced()
    if mode == "query" and config.attend_to_expansion_tokens:
        return [1] * len(token_ids)
    return [0 if tid == config.pad_token_id else 1 for tid in token_ids]


def finalize_colbert_engine_input(
    engine_input: EngineInput,
    *,
    mode: ColBERTEmbeddingMode,
    cache_key: str,
    config: ColBERTEncodingConfig,
) -> EngineInput:
    """Return a fresh, immutable ColBERT engine input (no shared mutable token lists).

    Attaches ``colbert_attention_mask`` as metadata only; it is not applied inside
    the model forward pass. See :data:`COLBERT_ATTENTION_MASK_ENFORCED_IN_FORWARD`.
    """
    _warn_colbert_attention_mask_not_enforced()
    if engine_input.get("type") != "token":
        raise ValueError("ColBERT encoding requires token engine inputs")

    token_ids = list(engine_input["prompt_token_ids"])
    result: TokensInput = {
        "type": "token",
        "prompt_token_ids": token_ids,
        "cache_salt": cache_key,
        "colbert_embedding_mode": mode,
        "colbert_cache_key": cache_key,
        "colbert_cache_salt": cache_key,
    }
    if prompt := engine_input.get("prompt"):
        result["prompt"] = prompt
    if arrival_time := engine_input.get("arrival_time"):
        result["arrival_time"] = arrival_time
    if mode == "query" and config.attend_to_expansion_tokens:
        result["colbert_attend_to_expansion"] = True
        result["colbert_full_attention_mask"] = True
    result["colbert_attention_mask"] = build_colbert_attention_mask(
        token_ids, mode, config
    )
    return result


@dataclass(frozen=True)
class ColBERTEncodingConfig:
    query_prefix: str
    document_prefix: str
    query_prefix_id: int
    document_prefix_id: int
    query_length: int
    document_length: int
    mask_token_id: int
    pad_token_id: int
    do_query_expansion: bool
    attend_to_expansion_tokens: bool

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        tokenizer: TokenizerLike,
    ) -> ColBERTEncodingConfig:
        model = model_config.model
        revision = model_config.revision
        metadata = _load_artifact_metadata(model, revision)
        st_cfg = _load_st_colbert_config(model, revision)

        is_jina = "jina-colbert" in model.lower() or any(
            a == "ColBERTJinaRobertaModel"
            for a in (getattr(model_config.hf_config, "architectures", None) or [])
        )

        query_prefix = (
            st_cfg.get("query_prefix")
            or metadata.get("query_token_id")
            or (JINA_QUERY_PREFIX if is_jina else DEFAULT_QUERY_PREFIX)
        )
        if isinstance(query_prefix, int):
            query_prefix = tokenizer.convert_ids_to_tokens(query_prefix) or str(
                query_prefix
            )

        document_prefix = (
            st_cfg.get("document_prefix")
            or metadata.get("doc_token_id")
            or (JINA_DOCUMENT_PREFIX if is_jina else DEFAULT_DOCUMENT_PREFIX)
        )
        if isinstance(document_prefix, int):
            document_prefix = tokenizer.convert_ids_to_tokens(document_prefix) or str(
                document_prefix
            )

        query_length = (
            st_cfg.get("query_length")
            or metadata.get("query_maxlen")
            or (32 if is_jina else 32)
        )
        document_length = (
            st_cfg.get("document_length")
            or metadata.get("doc_maxlen")
            or model_config.max_model_len
        )

        attend_to_expansion = st_cfg.get("attend_to_expansion_tokens")
        if attend_to_expansion is None:
            attend_meta = metadata.get("attend_to_mask_tokens")
            attend_to_expansion = (
                bool(attend_meta) if attend_meta is not None else is_jina
            )

        do_query_expansion = st_cfg.get("do_query_expansion")
        if do_query_expansion is None:
            do_query_expansion = True

        mask_token = tokenizer.mask_token or tokenizer.pad_token or "<mask>"
        if mask_token is None:
            raise ValueError(
                "Tokenizer must define mask_token or pad_token for ColBERT."
            )
        mask_token_id = _resolve_token_id(tokenizer, mask_token)

        pad_token = tokenizer.pad_token or mask_token
        pad_token_id = _resolve_token_id(tokenizer, pad_token)

        return cls(
            query_prefix=str(query_prefix),
            document_prefix=str(document_prefix),
            query_prefix_id=_resolve_token_id(tokenizer, str(query_prefix)),
            document_prefix_id=_resolve_token_id(tokenizer, str(document_prefix)),
            query_length=int(query_length),
            document_length=int(document_length),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            do_query_expansion=bool(do_query_expansion),
            attend_to_expansion_tokens=bool(attend_to_expansion),
        )

    def max_sequence_length_for_mode(self, mode: ColBERTEmbeddingMode) -> int:
        if mode == "query":
            return self.query_length
        return self.document_length


@dataclass(frozen=True)
class ColBERTTokenizeParams:
    """TokenizeParams wrapper that applies ColBERT post-tokenization hooks."""

    base: TokenizeParams
    config: ColBERTEncodingConfig
    mode: ColBERTEmbeddingMode

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def with_kwargs(self, **tokenization_kwargs):
        return ColBERTTokenizeParams(
            base=self.base.with_kwargs(**tokenization_kwargs),
            config=self.config,
            mode=self.mode,
        )

    def get_encode_kwargs(self) -> dict:
        return dict(
            truncation=False,
            add_special_tokens=self.base.add_special_tokens,
        )

    def apply_pre_tokenization(
        self,
        tokenizer: TokenizerLike | None,
        prompt: TextPrompt,
    ) -> TextPrompt:
        prompt = dict(prompt)
        prompt = self.base.apply_pre_tokenization(tokenizer, prompt)
        prompt["colbert_embedding_mode"] = self.mode
        return cast("TextPrompt", prompt)

    def apply_post_tokenization(
        self,
        tokenizer: TokenizerLike | None,
        prompt: TokensPrompt,
    ) -> TokensPrompt:
        if tokenizer is None:
            return prompt
        prompt = dict(prompt)
        token_ids = list(prompt["prompt_token_ids"])
        encoder = ColBERTEncoder(self.config)
        token_ids = encoder.encode_token_ids(token_ids, self.mode)
        prompt["prompt_token_ids"] = token_ids
        prompt["colbert_embedding_mode"] = self.mode
        if self.mode == "query" and self.config.attend_to_expansion_tokens:
            prompt["colbert_attend_to_expansion"] = True
            prompt["colbert_full_attention_mask"] = True
        return cast("TokensPrompt", prompt)


class ColBERTEncoder:
    """PyLate-compatible ColBERT token pipeline (marker → truncate → pad)."""

    def __init__(self, config: ColBERTEncodingConfig) -> None:
        self.config = config

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        tokenizer: TokenizerLike,
    ) -> ColBERTEncoder:
        return cls(ColBERTEncodingConfig.from_model_config(model_config, tokenizer))

    def build_tokenize_params(
        self,
        mode: ColBERTEmbeddingMode,
        base: TokenizeParams,
    ) -> ColBERTTokenizeParams:
        wrapped = base.with_kwargs(
            truncation=False,
            truncate_prompt_tokens=None,
            pad_prompt_tokens=None,
        )
        return ColBERTTokenizeParams(base=wrapped, config=self.config, mode=mode)

    @staticmethod
    def insert_prefix_token(token_ids: list[int], prefix_id: int) -> list[int]:
        if not token_ids:
            return [prefix_id]
        return [token_ids[0], prefix_id, *token_ids[1:]]

    def encode_token_ids(
        self,
        token_ids: list[int],
        mode: ColBERTEmbeddingMode,
    ) -> list[int]:
        """PyLate order: tokenize (done) → marker → length truncate → query pad."""
        prefix_id = (
            self.config.query_prefix_id
            if mode == "query"
            else self.config.document_prefix_id
        )
        encoded = self.insert_prefix_token(token_ids, prefix_id)
        max_len = self.config.max_sequence_length_for_mode(mode)
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        if mode == "query" and self.config.do_query_expansion:
            pad_id = self.config.mask_token_id
            if len(encoded) < max_len:
                encoded = encoded + [pad_id] * (max_len - len(encoded))
        return encoded


def build_colbert_prompt_extras(
    mode: ColBERTEmbeddingMode,
    config: ColBERTEncodingConfig,
    *,
    model_name: str,
    raw_text: str,
    chat_template_hash: str,
    colbert_config_hash: str,
) -> dict:
    cache_key = compute_colbert_cache_key(
        model_name=model_name,
        embedding_mode=mode,
        raw_text=raw_text,
        chat_template_hash=chat_template_hash,
        colbert_config_hash=colbert_config_hash,
    )
    extras = {
        "colbert_embedding_mode": mode,
        "colbert_cache_key": cache_key,
        "colbert_cache_salt": cache_key,
        "cache_salt": cache_key,
    }
    if mode == "query" and config.attend_to_expansion_tokens:
        extras["colbert_attend_to_expansion"] = True
        extras["colbert_full_attention_mask"] = True
    return extras


def apply_embedding_mode_to_pooling_params(
    pooling_params,
    mode: ColBERTEmbeddingMode | None,
    config: ColBERTEncodingConfig | None,
) -> None:
    if mode is None:
        return
    extra = dict(pooling_params.extra_kwargs or {})
    extra["embedding_mode"] = mode
    extra["colbert_embedding_mode"] = mode
    if config is not None and mode == "query" and config.attend_to_expansion_tokens:
        extra["colbert_attend_to_expansion"] = True
        extra["colbert_full_attention_mask"] = True
    pooling_params.extra_kwargs = extra


def validate_colbert_engine_inputs(
    engine_inputs: list[EngineInput],
    config: ColBERTEncodingConfig,
) -> None:
    """Debug-only shape checks (``VLLM_COLBERT_PARITY_CHECK=1``)."""
    if not colbert_parity_check_enabled():
        return

    seen_keys: set[str] = set()
    for engine_input in engine_inputs:
        mode = engine_input.get("colbert_embedding_mode")
        if mode not in ("query", "document"):
            raise ValueError(
                "ColBERT parity check: missing colbert_embedding_mode on "
                f"{engine_input!r}"
            )
        token_ids = engine_input.get("prompt_token_ids")
        if token_ids is None:
            continue
        seq_len = len(token_ids)
        if mode == "query" and seq_len > config.query_length:
            raise ValueError(
                f"ColBERT parity check: query length {seq_len} > {config.query_length}"
            )
        if mode == "document" and seq_len > config.document_length:
            raise ValueError(
                f"ColBERT parity check: document length {seq_len} > "
                f"{config.document_length}"
            )
        cache_key = engine_input.get("colbert_cache_key") or engine_input.get(
            "colbert_cache_salt"
        )
        if cache_key is None:
            raise ValueError("ColBERT parity check: missing colbert_cache_key")
        if cache_key in seen_keys:
            logger.warning(
                "ColBERT parity check: duplicate cache key %s (may be intentional)",
                cache_key,
            )
        seen_keys.add(cache_key)
