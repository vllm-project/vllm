# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColBERT IO mixin (isolated to avoid embed <-> scoring circular imports)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from vllm import PoolingParams
from vllm.inputs import EngineInput, PromptType, TextPrompt
from vllm.model_executor.models.colbert_encoding import (
    ColBERTEmbeddingMode,
    ColBERTEncoder,
    apply_embedding_mode_to_pooling_params,
    canonicalize_colbert_input_text,
    compute_colbert_cache_key,
    compute_colbert_config_hash,
    finalize_colbert_engine_input,
    resolve_colbert_chat_template_hash,
    validate_colbert_engine_inputs,
)
from vllm.renderers import TokenizeParams
from vllm.renderers.inputs.preprocess import prompt_to_seq

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.renderers import BaseRenderer


class ColBERTIOProcessorMixin:
    model_config: ModelConfig
    renderer: BaseRenderer
    chat_template: str | None
    """Apply PyLate-compatible ColBERT encoding via renderer (per-request isolation).

    Mode-group batch rendering is execution-only: it groups ``render_cmpl`` calls
    by mode for efficiency and does not change tokenization semantics or outputs.
    Each prompt still becomes an independent :class:`~vllm.inputs.EngineInput`
    after render. Grouping does not couple KV-cache boundaries, attention graphs,
    position encoding continuity, or scheduler batch semantics.
    """

    _colbert_encoder: ColBERTEncoder | None = None

    def _colbert_chat_template_hash(self, request) -> str:
        chat_template = getattr(request, "chat_template", None)
        if chat_template is None:
            chat_template = self.chat_template
        return resolve_colbert_chat_template_hash(
            chat_template,
            model_config=self.model_config,
        )

    @staticmethod
    def _extract_prompt_text(prompt: PromptType) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            prompt_text = prompt.get("prompt")
            if isinstance(prompt_text, str):
                return prompt_text
        raise ValueError(
            "ColBERT embedding_mode requires text prompts; "
            "token prompts are not supported."
        )

    def _get_colbert_encoder(self) -> ColBERTEncoder:
        if self._colbert_encoder is None:
            self._colbert_encoder = ColBERTEncoder.from_model_config(
                self.model_config,
                self.renderer.get_tokenizer(),
            )
        return self._colbert_encoder

    def _resolve_colbert_mode(
        self,
        request_mode: ColBERTEmbeddingMode | None,
        pooling_params: PoolingParams | None,
    ) -> ColBERTEmbeddingMode | None:
        if request_mode is not None:
            return request_mode
        if pooling_params is None:
            return None
        if pooling_params.embedding_mode is not None:
            return pooling_params.embedding_mode
        extra = pooling_params.extra_kwargs or {}
        mode = extra.get("embedding_mode") or extra.get("colbert_embedding_mode")
        if mode in ("query", "document"):
            return mode
        return None

    def _render_colbert_cmpl(
        self,
        request,
        prompts: Sequence[PromptType],
        mode: ColBERTEmbeddingMode,
        pooling_params: PoolingParams | None = None,
        *,
        tok_params: TokenizeParams | None = None,
        chat_template_hash: str | None = None,
    ) -> list[EngineInput]:
        """Render one mode-homogeneous group via a single ``render_cmpl`` batch call."""
        encoder = self._get_colbert_encoder()
        config = encoder.config

        if tok_params is None:
            tok_params = request.build_tok_params(self.model_config)
        colbert_tok_params = encoder.build_tokenize_params(mode, tok_params)

        if chat_template_hash is None:
            chat_template_hash = self._colbert_chat_template_hash(request)
        model_name = self.model_config.model
        colbert_config_hash = compute_colbert_config_hash(
            config, model_config=self.model_config
        )

        tokenizer = self.renderer.get_tokenizer()
        text_prompts: list[TextPrompt] = []
        cache_keys: list[str] = []
        for prompt in prompt_to_seq(prompts):
            request_text = self._extract_prompt_text(prompt)
            input_text = canonicalize_colbert_input_text(
                request_text, colbert_tok_params, tokenizer
            )
            cache_key = compute_colbert_cache_key(
                model_name=model_name,
                embedding_mode=mode,
                raw_text=input_text,
                chat_template_hash=chat_template_hash,
                colbert_config_hash=colbert_config_hash,
            )
            cache_keys.append(cache_key)

            text_prompt: TextPrompt = {
                "prompt": input_text,
                "colbert_embedding_mode": mode,
                "colbert_cache_key": cache_key,
                "colbert_cache_salt": cache_key,
                "cache_salt": cache_key,
            }
            if mode == "query" and config.attend_to_expansion_tokens:
                text_prompt["colbert_attend_to_expansion"] = True
                text_prompt["colbert_full_attention_mask"] = True
            if isinstance(prompt, dict):
                if mm := prompt.get("mm_processor_kwargs"):
                    text_prompt["mm_processor_kwargs"] = cast(Any, mm)
                if mm_uuids := prompt.get("multi_modal_uuids"):
                    text_prompt["multi_modal_uuids"] = cast(Any, mm_uuids)
                if mm_data := prompt.get("multi_modal_data"):
                    text_prompt["multi_modal_data"] = cast(Any, mm_data)
            text_prompts.append(text_prompt)

        prompt_extras: dict[str, Any] | None = None
        if mm_kwargs := getattr(request, "mm_processor_kwargs", None):
            prompt_extras = {"mm_processor_kwargs": mm_kwargs}

        rendered = self.renderer.render_cmpl(
            text_prompts,
            cast(TokenizeParams, colbert_tok_params),
            prompt_extras=prompt_extras,
        )
        if len(rendered) != len(cache_keys):
            raise ValueError(
                "ColBERT batch render returned "
                f"{len(rendered)} engine inputs for {len(cache_keys)} prompts"
            )

        finalized_inputs: list[EngineInput] = []
        for engine_input, cache_key in zip(rendered, cache_keys):
            finalized_inputs.append(
                finalize_colbert_engine_input(
                    engine_input,
                    mode=mode,
                    cache_key=cache_key,
                    config=config,
                )
            )
        if pooling_params is not None and len(finalized_inputs) == 1:
            apply_embedding_mode_to_pooling_params(pooling_params, mode, config)
        validate_colbert_engine_inputs(finalized_inputs, config)
        return finalized_inputs

    def _render_colbert_cmpl_batch(
        self,
        request,
        prompts: Sequence[PromptType],
        modes: Sequence[ColBERTEmbeddingMode],
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        tok_params: TokenizeParams | None = None,
    ) -> list[EngineInput]:
        if len(prompts) != len(modes):
            raise ValueError("prompts and embedding modes must have the same length")

        chat_template_hash = self._colbert_chat_template_hash(request)
        indexed = list(enumerate(zip(prompts, modes)))
        results: list[EngineInput | None] = [None] * len(prompts)

        # NOTE: Batch grouping is strictly a PRE-TOKENIZATION rendering optimization.
        # It does NOT affect:
        # - KV cache reuse boundaries
        # - attention computation graph
        # - positional encoding continuity
        # - model forward batching semantics
        #
        # Each EngineInput remains fully isolated after render.
        for mode in ("query", "document"):
            group = [(i, p) for i, (p, m) in indexed if m == mode]
            if not group:
                continue
            group_prompts = [p for _, p in group]
            group_params = None
            if isinstance(pooling_params, Sequence):
                group_params = [pooling_params[i] for i, _ in group]
            elif pooling_params is not None:
                group_params = pooling_params

            rendered = self._render_colbert_cmpl(
                request,
                group_prompts,
                mode,
                group_params if not isinstance(group_params, list) else None,
                tok_params=tok_params,
                chat_template_hash=chat_template_hash,
            )
            if isinstance(group_params, list):
                encoder_config = self._get_colbert_encoder().config
                for params, engine_input in zip(group_params, rendered):
                    apply_embedding_mode_to_pooling_params(
                        params,
                        mode,
                        encoder_config,
                    )
            if len(rendered) != len(group):
                raise ValueError(
                    "ColBERT mode-group render size mismatch for "
                    f"mode={mode}: {len(rendered)} vs {len(group)}"
                )
            for (idx, _), engine_input in zip(group, rendered):
                results[idx] = engine_input

        if any(r is None for r in results):
            raise ValueError("ColBERT batch rendering failed to produce all inputs")

        engine_inputs = [r for r in results if r is not None]
        validate_colbert_engine_inputs(
            engine_inputs,
            self._get_colbert_encoder().config,
        )
        return engine_inputs
