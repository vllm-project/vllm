# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


_MIN_CAUSAL_QUERY_LEN = 2
_MIXED_WARMUP_TOKENS = 3
_PACK_GQA_SMALL_TILE_MAX_ROWS = 64


def _causal_warmup_query_lens(
    max_warmup_tokens: int, num_queries_per_kv: tuple[int, ...]
) -> tuple[int, ...]:
    """Return query lengths that cover causal FA4 warmup paths.

    FA4 treats a one-token query as noncausal, so two is the shortest causal
    query. PackGQA changes tiles when the query length times
    ``num_queries_per_kv`` exceeds 64. Half the warmup budget also covers
    long-prefill scheduling.
    """
    query_lens = {_MIN_CAUSAL_QUERY_LEN, max_warmup_tokens // 2}
    query_lens.update(
        _PACK_GQA_SMALL_TILE_MAX_ROWS // ratio + 1 for ratio in num_queries_per_kv
    )
    return tuple(
        sorted(
            query_len
            for query_len in query_lens
            if _MIN_CAUSAL_QUERY_LEN <= query_len <= max_warmup_tokens
        )
    )


def _loaded_fa4_num_queries_per_kv(vllm_config: object) -> tuple[int, ...]:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    layers = getattr(static_forward_context, "values", None)
    if not callable(layers):
        return ()
    values = []
    for layer in layers():
        impl = getattr(layer, "impl", None)
        if getattr(impl, "vllm_flash_attn_version", None) != 4:
            continue
        ratio = getattr(impl, "num_queries_per_kv", None)
        if ratio is None:
            num_heads = getattr(impl, "num_heads", 1)
            num_kv_heads = getattr(impl, "num_kv_heads", 1)
            ratio = num_heads // num_kv_heads
        values.append(ratio)
    return tuple(dict.fromkeys(values))


def _warm_fa4_mla_prefill(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not vllm_config.model_config.use_mla:
        return
    try:
        backend_cls = get_mla_prefill_backend(vllm_config)
    except ValueError:
        # Fall back to the top-k MQA prefill path.
        return
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from vllm.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)


def _warm_fa4_runtime_attention(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if vllm_config.model_config.use_mla:
        try:
            backend_cls = get_mla_prefill_backend(vllm_config)
        except ValueError:
            # Fall back to the top-k MQA prefill path.
            return
        if backend_cls.get_name() != "FLASH_ATTN":
            return

    if (
        not current_platform.is_device_capability(90)
        or vllm_config.attention_config.flash_attn_version != 4
    ):
        return

    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    if not worker.use_v2_model_runner:
        if vllm_config.model_config.use_mla:
            from vllm.v1.attention.backends.mla.flashattn_mla import (
                FlashAttnMLAMetadataBuilder,
            )

            max_warmup_tokens = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            if max_warmup_tokens < 2:
                return
            runner._dummy_run(
                min(
                    FlashAttnMLAMetadataBuilder.reorder_batch_threshold + 1,
                    max_warmup_tokens,
                ),
                force_attention=True,
                is_profile=True,
                create_mixed_batch=True,
                skip_eplb=True,
                profile_seq_lens=max_warmup_tokens // 2,
            )
            for context_len in (min(128, max_warmup_tokens), max_warmup_tokens // 2):
                runner._dummy_run(
                    2,
                    force_attention=True,
                    is_profile=True,
                    skip_eplb=True,
                    profile_seq_lens=context_len,
                    num_reqs=1,
                )
        else:
            num_queries_per_kv = _loaded_fa4_num_queries_per_kv(vllm_config)
            if not num_queries_per_kv:
                return
            max_warmup_tokens = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            for query_len in _causal_warmup_query_lens(
                max_warmup_tokens, num_queries_per_kv
            ):
                # Warm causal prefill at batch size 1.
                runner._dummy_run(
                    query_len,
                    force_attention=True,
                    is_profile=True,
                    skip_eplb=True,
                    profile_seq_lens=query_len,
                    num_reqs=1,
                )
                if (
                    runner.scheduler_config.max_num_seqs >= 2
                    and query_len < max_warmup_tokens
                ):
                    # One cached request decodes a token while a new request
                    # runs the shortest causal prefill.
                    runner._dummy_run(
                        _MIXED_WARMUP_TOKENS,
                        force_attention=True,
                        is_profile=True,
                        create_mixed_batch=True,
                        skip_eplb=True,
                        profile_seq_lens=[
                            query_len + 1,
                            _MIN_CAUSAL_QUERY_LEN,
                        ],
                    )
        return
    from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec

    if any(
        isinstance(group.kv_cache_spec, (CrossAttentionSpec, MambaSpec))
        for group in runner.kv_cache_config.kv_cache_groups
    ):
        return
    max_warmup_tokens = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.model_config.max_model_len,
    )
    if vllm_config.model_config.use_mla:
        from vllm.v1.attention.backends.mla.flashattn_mla import (
            FlashAttnMLAMetadataBuilder,
        )

        absorbed_tokens = min(
            FlashAttnMLAMetadataBuilder.reorder_batch_threshold + 1,
            max_warmup_tokens,
        )
        run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            absorbed_tokens,
            req_id_prefix=f"_fa4_mla_warmup_{absorbed_tokens}",
        )

    if vllm_config.model_config.use_mla:
        context_tokens = max_warmup_tokens // 2
        run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=_MIXED_WARMUP_TOKENS,
            decode_prompt_len=context_tokens,
            decode_scheduled_tokens=1,
            req_id_prefix=f"_fa4_warmup_{max_warmup_tokens}",
        )
    else:
        num_queries_per_kv = _loaded_fa4_num_queries_per_kv(vllm_config)
        if not num_queries_per_kv:
            return
        for query_len in _causal_warmup_query_lens(
            max_warmup_tokens, num_queries_per_kv
        ):
            mixed_warmed = False
            if query_len < max_warmup_tokens:
                mixed_warmed = run_mixed_prefill_decode_warmup(
                    runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens=_MIXED_WARMUP_TOKENS,
                    decode_prompt_len=query_len,
                    decode_scheduled_tokens=1,
                    req_id_prefix=(f"_fa4_warmup_{max_warmup_tokens}_{query_len}"),
                )
            if not mixed_warmed:
                runner._dummy_run(
                    query_len,
                    skip_eplb=True,
                    is_profile=True,
                    num_reqs=1,
                )


def _warm_inkling_fa4_rel_attention(worker: Worker) -> None:
    from vllm.models.inkling.configs import InklingMMConfig, InklingModelConfig
    from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
        INKLING_FA4_REL_ATTENTION_KERNEL,
    )

    vllm_config = worker.vllm_config
    hf_config = vllm_config.model_config.hf_config
    if not isinstance(hf_config, (InklingMMConfig, InklingModelConfig)):
        return

    INKLING_FA4_REL_ATTENTION_KERNEL.warmup(vllm_config)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    _warm_fa4_mla_prefill(worker)
    _warm_fa4_runtime_attention(worker)
    _warm_inkling_fa4_rel_attention(worker)
