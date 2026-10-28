# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuning and compiling the specialized decode's kernels.

The model state calls ``warm_up`` once the executor is built, before memory
profiling and CUDA graph capture, so no kernel compiles or tunes inside a
capture. FlashInfer's tuner times each tactic on every TP rank and reduces the
timings over the TP group, so every rank makes the same calls in the same
order.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch

import vllm.utils.flashinfer as fi_utils
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context

from .. import FUSED_MHC_MAX_ROWS, LOW_BATCH_MAX_ROWS
from ..attention.aux_projections import warmup_aux_projection_gemms
from ..attention.mxfp8 import (
    pinned_tactics,
    prefer_pinned_tactics,
    run_zeros,
    shape_representatives,
)
from ..attention.step import AttentionStep, key_weight, side_weights
from ..logits.lm_head import autotune_low_batch_lm_head, warmup_low_batch_lm_head
from ..moe.step import deferred_finalize
from .decode import SpecializedDecode


def warm_up(decode: SpecializedDecode, vllm_config: VllmConfig) -> None:
    """Tune (unless FlashInfer autotuning is off) and compile the executor's
    GEMMs for every row count it runs them at."""
    steps = [plan.attention for plan in decode.plans]
    linears = shape_representatives(
        linear
        for step in steps
        for linear in (step.attn.fused_wqa_wkv, step.attn.wq_b, step.attn.wo_b)
    )
    rows = min(vllm_config.scheduler_config.max_num_batched_tokens, FUSED_MHC_MAX_ROWS)
    prefer_pinned_tactics()
    with torch.inference_mode():
        if decode.config.autotune:
            with _tuning():
                with pinned_tactics():
                    for linear in linears:
                        run_zeros(linear, LOW_BATCH_MAX_ROWS)
                autotune_low_batch_lm_head(decode.lm_head)
                _deferred_moe(decode, vllm_config, rows, tune=True)
        else:
            _deferred_moe(decode, vllm_config, rows, tune=False)

        for linear in linears:
            for m in range(1, LOW_BATCH_MAX_ROWS + 1):
                run_zeros(linear, m)
        # The publishing wo_b compiles the tactics selected above; one layer
        # per shape compiles every layer's kernels.
        publishing = {
            tuple(step.attn.wo_b.weight.shape): step
            for step in steps
            if step.publisher is not None
        }
        for step in publishing.values():
            assert step.publisher is not None
            for m in range(1, LOW_BATCH_MAX_ROWS + 1):
                step.publisher.precompile(step.attn.wo_b, m)
        warmup_aux_projection_gemms(_aux_weights(steps))
        warmup_low_batch_lm_head(decode.lm_head)
    if decode.config.jit_warmup:
        _warm_up_mhc(decode, vllm_config, rows)


@contextmanager
def _tuning() -> Iterator[None]:
    """FlashInfer's tuning context, its timings reduced over the TP group."""
    from flashinfer.autotuner import set_autotune_process_group

    group = get_tp_group()
    set_autotune_process_group(group.cpu_group if group.world_size > 1 else None)
    try:
        with fi_utils.autotune(tune_mode=True):
            yield
    finally:
        set_autotune_process_group(None)


def _aux_weights(steps: list[AttentionStep]) -> list[torch.Tensor]:
    """The weights the layers run on ``ll_bf16``."""
    weights = [w for step in steps if step.aux_gemm for w in side_weights(step.attn)]
    for step in steps:
        wk = key_weight(step.attn)
        if wk is not None and step.key_gemm:
            weights.append(wk)
    return weights


def _deferred_moe(
    decode: SpecializedDecode, vllm_config: VllmConfig, rows: int, tune: bool
) -> None:
    """One deferring MoE layer on ``rows`` random rows with the finalize
    deferred, as the executor runs it above ``LOW_BATCH_MAX_ROWS`` rows:
    tunes its experts over the serving buckets, and allocates what the
    deferred launch allocates on first use before memory profiling."""
    moe = next((plan.moe for plan in decode.plans if plan.moe.deferrable), None)
    if moe is None:
        return
    runner = moe.runner
    weight = next(p for p in runner.parameters() if p.is_cuda)
    hidden = torch.randn(
        rows,
        decode.backbone.config.hidden_size,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    input_ids = torch.zeros(rows, dtype=torch.int64, device=weight.device)
    tuning = (
        fi_utils.autotune(
            tuning_buckets=fi_utils.flashinfer_get_hybrid_num_tokens_buckets(rows),
            round_up=True,
        )
        if tune
        else nullcontext()
    )
    with (
        set_forward_context(None, vllm_config),
        deferred_finalize(runner.moe_config, rows),
        tuning,
    ):
        runner._forward_impl(hidden, hidden, hidden, input_ids)


def _warm_up_mhc(decode: SpecializedDecode, vllm_config: VllmConfig, rows: int) -> None:
    """Compile the split mHC pre-norm kernels the boundaries run, for every row
    count the executor takes; the layers register them for fewer rows."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_post_pre_splits,
    )
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

    layer = decode.plans[0].layer
    hidden, hc_mult = layer.hidden_size, layer.hc_mult
    hc_stream = {
        "rms_numel": hc_mult * hidden,
        "extra_splits": mhc_fused_post_pre_splits(hidden, hc_mult),
    }
    variants = [
        {"rms_numel": hidden, "use_pre_mix_in": False, "extra_splits": ()},
        {**hc_stream, "use_pre_mix_in": False},
        {**hc_stream, "use_pre_mix_in": True},
    ]
    if vllm_config.speculative_config is not None:
        variants.append({**hc_stream, "use_pre_mix_in": True, "write_aux": True})
    for variant in variants:
        for split_mode in ("stats", "input"):
            MHC_PRE_NORM_KERNEL.warmup(
                split_mode=split_mode,
                max_tokens=rows,
                hidden_size=hidden,
                rms_eps=layer.rms_norm_eps,
                hc_pre_eps=layer.hc_eps,
                hc_sinkhorn_eps=layer.hc_eps,
                hc_post_mult_value=layer.hc_post_alpha,
                sinkhorn_repeat=layer.hc_sinkhorn_iters,
                norm_eps=layer.rms_norm_eps,
                hc_mult=hc_mult,
                **variant,
            )
