# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nightly-specific eligibility for the private MiniMax target/draft sampler."""

from __future__ import annotations

import os


def qualified_type(value) -> str:
    """Keep custom processors, head implementations and model subclasses native."""
    return f"{type(value).__module__}.{type(value).__name__}"


def config_supported(config) -> bool:
    """Limit the opt-in payload to the unchanged MiniMax TP4 greedy Eagle3 lane."""
    parallel, model, spec = (
        config.parallel_config,
        config.model_config,
        config.speculative_config,
    )
    return (
        spec is not None
        and model.hf_config.model_type == "minimax_m3_vl"
        and str(model.dtype) == "torch.bfloat16"
        and str(model.head_dtype) == "torch.bfloat16"
        and parallel.tensor_parallel_size == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and not parallel.enable_batch_sharded_sampling
        and config.lora_config is None
        and config.watermark_config is None
        and spec.method == "eagle3"
        and spec.draft_sample_method == "greedy"
        and spec.num_speculative_tokens == 3
        and not spec.enable_adaptive_verification
        and not spec.parallel_drafting
    )


def head_geometry(model, group) -> tuple[int, int] | None:
    """Check the stock BF16 head and its complete, contiguous original vocabulary."""
    head = getattr(model, "lm_head", None)
    processor = getattr(model, "logits_processor", None)
    if (
        qualified_type(head)
        != "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead"
        or qualified_type(processor)
        != "vllm.model_executor.layers.logits_processor.LogitsProcessor"
    ):
        return None
    width = head.num_embeddings_per_partition
    start = head.shard_indices.org_vocab_start_index
    vocab = processor.org_vocab_size
    if (
        group.world_size != 4
        or head.tp_size != group.world_size
        or head.embedding_dim != 6144
        or str(head.weight.dtype) != "torch.bfloat16"
        or qualified_type(head.quant_method)
        not in (
            "vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod",
            "vllm.model_executor.layers.linear.UnquantizedLinearMethod",
        )
        or processor.logits_as_input
        or processor.head_dtype is not None
        and str(processor.head_dtype) != "torch.bfloat16"
        or processor.scale != 1.0
        or processor.soft_cap is not None
        or not processor.use_all_gather
        or head.num_added_embeddings != 0
        or vocab != 200064
        or processor.vocab_size != vocab
        or width != 50016
        or start != group.rank_in_group * width
        or head.shard_indices.org_vocab_end_index != min(start + width, vocab)
    ):
        return None
    return start, vocab


def can_use_target(rejection_sampler, batch, sampler_type) -> bool:
    """Mirror nightly's no-processor shortcut; otherwise fall back before the head."""
    sampler = rejection_sampler.sampler
    if (
        type(sampler) is not sampler_type
        or batch.num_reqs <= 0
        or batch.num_draft_tokens <= 0
        or rejection_sampler.num_speculative_steps != 3
        or rejection_sampler.enable_adaptive_verification
        or rejection_sampler.use_block_verification
        or rejection_sampler.watermark_key is not None
        or sampler.compute_nans
        or sampler.return_sampling_mask
        or sampler.trace_replay_state is not None
        or sampler.use_fp64_gumbel
    ):
        return False
    slots = batch.idx_mapping_np
    if (
        len(slots) != batch.num_reqs
        or any(
            slot < 0 or slot >= len(sampler.needs_logits_processing) for slot in slots
        )
        or sampler.get_logprobs_dims(slots) is not None
        or any(sampler.needs_logits_processing[slot] for slot in slots)
        or any(sampler.sampling_states.temperature.np[slot] != 0 for slot in slots)
    ):
        return False
    cu = batch.cu_num_logits_np
    return (
        len(cu) == batch.num_reqs + 1
        and cu[0] == 0
        and cu[-1] == batch.logits_indices.shape[0]
        and all(1 <= end - start <= 4 for start, end in zip(cu, cu[1:]))
    )


def configure_draft(speculator) -> bool:
    """Validate once after loading; never inspect mapping values during capture."""
    if os.getenv("MM3_FUSED_DRAFT_ARGMAX_EXPERIMENT") != "1":
        return False
    import torch

    from vllm.distributed import get_tp_group
    from vllm.logger import init_logger
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    model = speculator.model
    if (
        not current_platform.is_rocm()
        or not config_supported(speculator.vllm_config)
        or qualified_type(model)
        != "vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM"
        or getattr(speculator.compute_draft_logits, "__func__", None)
        is not DraftModelSpeculator.compute_draft_logits
        or speculator.use_local_argmax_reduction
        or speculator.draft_logits is not None
        or speculator.draft_watermarker is not None
        or speculator.use_acceptance_estimator
        or speculator.acceptance_estimator is not None
        or head_geometry(model, get_tp_group()) is None
        or model.config.draft_vocab_size != 200064
        or model.config.vocab_size != 200064
    ):
        return False
    mapping = model.draft_id_to_target_id
    if mapping is not None and (
        mapping.numel() != 200064
        or mapping.dtype != torch.int64
        or torch.count_nonzero(mapping).item() != 0
    ):
        return False
    init_logger(__name__).info(
        "N03 draft sampling enabled at sample_draft: stock BF16 head, "
        "identity vocabulary, dynamic-token compact rank-major argmax"
    )
    return True


def target_model(model, rejection_sampler, speculator, config):
    """Return a verified native inner model; do not add or replace model methods."""
    if os.getenv("MM3_SHARDED_GREEDY_VERIFY_EXPERIMENT") != "1":
        return None
    from vllm.distributed import get_tp_group
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    if (
        not current_platform.is_rocm()
        or not config_supported(config)
        or type(rejection_sampler) is not RejectionSampler
        or type(rejection_sampler.sampler) is not Sampler
        or speculator is None
        or speculator.draft_logits is not None
        or rejection_sampler.watermark_key is not None
    ):
        return None
    module = "vllm.models.minimax_m3.amd.model."
    if qualified_type(model) == module + "MiniMaxM3SparseForConditionalGeneration":
        model = model.language_model
    if (
        qualified_type(model) != module + "MiniMaxM3SparseForCausalLM"
        or head_geometry(model, get_tp_group()) is None
        or config.model_config.get_vocab_size() != 200064
    ):
        return None
    return model
