# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL attention kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fa4_cutedsl_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not current_platform.is_device_capability(90):
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
        return
    if vllm_config.attention_config.flash_attn_version != 4:
        return

    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    if vllm_config.model_config.use_mla:
        try:
            backend_cls = get_mla_prefill_backend(vllm_config)
        except ValueError:
            # Fall back to the top-k MQA prefill path.
            return
        if backend_cls.get_name() != "FLASH_ATTN":
            return

        from vllm.v1.attention.backends.mla.prefill import flash_attn

        if not flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(vllm_config):
            return
        flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)

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

    # Long prefill compilation and long-context batched decode select scheduler
    # modes that the existing short-context model-runner warmup does not reach.
    num_long_decodes = min(4, runner.max_num_reqs - 1)
    decode_scheduled_tokens = 2
    run_mixed_prefill_decode_warmup(
        runner,
        worker.execute_model,
        worker.sample_tokens,
        num_long_decodes * decode_scheduled_tokens + 17,
        decode_prompt_len=max_warmup_tokens // 2,
        num_decode_reqs=num_long_decodes,
        decode_scheduled_tokens=decode_scheduled_tokens,
        req_id_prefix=f"_fa4_warmup_{max_warmup_tokens}",
    )
