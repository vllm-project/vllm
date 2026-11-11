# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MHCFusedPostPreOp on FlashInfer's fused mHC kernel (`flashinfer.mega_mhc`).

One persistent kernel computes a sublayer's mHC post block, the next
sublayer's pre block and its RMSNorm: the work the TileLang path does in two
launches for up to 32 tokens, and in post + projection GEMM + pre beyond that.
FlashInfer ships it as the experimental `prepare_mega_mhc` API; its
`sf_layout="bf16"` programs serve hidden size 4096 with four residual
streams in normal (non-delayed) mode. The model decides which steps use it.

A FlashInfer plan binds the tensors it is prepared with, so a plan is prepared
for each call. The kernel reads the FP32 projection through its three BF16
pieces (`fn_split`), which the model owns and refreshes when its weights
change. The cross-CTA counters and the scratch live in vLLM's workspace
manager. FlashInfer JIT-builds the program a token count selects; the JIT
warmup (`MegaMHCKernel`) builds every program before serving.
"""

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# The shape FlashInfer's BF16-output mega mHC programs serve.
HIDDEN_SIZE = 4096
HC_MULT = 4
# FlashInfer before #8199 refuses more than 64 tokens per SM in one launch.
_MAX_TOKENS_PER_SM = 64

__all__ = [
    "MEGA_MHC_KERNEL",
    "mhc_fused_post_pre_flashinfer_mega",
    "mega_mhc_supported",
    "split_mhc_fn",
]


@functools.cache
def _flashinfer_mega_mhc():
    """`(prepare_mega_mhc, host rules module)`, or None if FlashInfer lacks them."""
    try:
        from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as rules
        from flashinfer.jit import env as jit_env
        from flashinfer.mega_mhc import prepare_mega_mhc
    except ImportError:
        return None
    if not hasattr(rules, "mhc_infer_num_splits"):
        # A FlashInfer whose mega mHC has no BF16-output (hidden 4096) program.
        return None
    # FlashInfer's PyPI release wheels leave out the experimental kernel
    # sources (flashinfer-ai/flashinfer#7977); nightly and source installs
    # have them. Without them the kernel can't be JIT-built.
    sources = jit_env.FLASHINFER_CSRC_DIR / rules.GENERATED_ROOT.removeprefix("csrc/")
    if not sources.is_dir():
        logger.info(
            "FlashInfer was installed without its experimental kernel sources; "
            "mHC keeps TileLang."
        )
        return None
    return prepare_mega_mhc, rules


def mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    """Whether FlashInfer has a fused mHC program for this shape and device."""
    if (hidden_size, hc_mult) != (HIDDEN_SIZE, HC_MULT):
        return False
    if not current_platform.is_cuda():
        return False
    fi = _flashinfer_mega_mhc()
    if fi is None:
        return False
    rules = fi[1]
    arch = rules.CAPABILITIES.get(current_platform.get_device_capability())
    return arch is not None and rules.supported(arch, HIDDEN_SIZE, "bf16")


def split_mhc_fn(fn: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """The three BF16 pieces of the FP32 projection `fn` the kernel reads.

    Writes into `out` when given, so captured CUDA graphs see the new values.
    """
    return _flashinfer_mega_mhc()[1].split_fn(fn.detach(), out=out)


@functools.cache
def _launch_grid(num_tokens: int, num_sms: int) -> tuple[int, int]:
    """(split count, grid) FlashInfer's plan uses for `num_tokens`."""
    rules = _flashinfer_mega_mhc()[1]
    splits = rules.mhc_infer_num_splits(num_tokens, num_sms)
    return splits, rules.mhc_infer_num_ctas(num_tokens, num_sms, splits)


@functools.cache
def _max_scratch_floats(num_sms: int) -> int:
    # Not monotonic in the token count, so take the maximum over every launch.
    rules = _flashinfer_mega_mhc()[1]
    return max(
        rules.mhc_infer_scratch_floats(t, _launch_grid(t, num_sms)[0])
        for t in range(1, _MAX_TOKENS_PER_SM * num_sms + 1)
    )


def _counters(device: torch.device, num_sms: int):
    """Zeroed (split barriers, launch epochs) per launch grid, or None.

    Every CTA advances its own epoch slot, so only launches of one grid may
    share a pair. All pairs are made on the first call of each ubatch, before
    CUDA graph capture locks the workspace.
    """
    if not is_workspace_manager_initialized():
        return None

    def allocate() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        words = _flashinfer_mega_mhc()[1].SPLIT_BARRIER_WORDS
        # From 4 tokens per SM on, the grid is the SM count.
        grids = {_launch_grid(t, num_sms)[1] for t in range(1, 4 * num_sms + 1)}
        return {
            grid: (
                torch.zeros(words, dtype=torch.uint64, device=device),
                torch.zeros(num_sms, dtype=torch.uint64, device=device),
            )
            for grid in grids
        }

    return current_workspace_manager().get_persistent_resource(
        ("flashinfer_mega_mhc", device), allocate
    )


def mhc_fused_post_pre_flashinfer_mega(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    fn_split: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MHCFusedPostPreOp (with its RMSNorm) as one FlashInfer launch.

    Same contract as `mhc_fused_post_pre_tilelang`, plus `fn_split`, the
    BF16 pieces of `fn` (`split_mhc_fn`).
    """
    num_tokens = residual.numel() // (HC_MULT * HIDDEN_SIZE)
    prepare_mega_mhc, rules = _flashinfer_mega_mhc()
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, HC_MULT, HIDDEN_SIZE).contiguous()
    device = residual.device
    _, num_sms = rules.device_facts(device.index)
    counters = _counters(device, num_sms)
    scratch = None
    if counters is not None:
        (scratch,) = current_workspace_manager().get_simultaneous(
            ((_max_scratch_floats(num_sms),), torch.float32)
        )

    x_flat = x.reshape(num_tokens, HIDDEN_SIZE).contiguous()
    post_flat = post_layer_mix.reshape(num_tokens, HC_MULT, 1).contiguous()
    comb_flat = comb_res_mix.reshape(num_tokens, HC_MULT, HC_MULT).contiguous()
    norm_weight = norm_weight.contiguous()
    out = {
        "new_residual": torch.empty_like(residual_flat),
        "new_post_mix": torch.empty(
            num_tokens, HC_MULT, 1, dtype=torch.float32, device=device
        ),
        "new_comb_res_mix": torch.empty(
            num_tokens, HC_MULT, HC_MULT, dtype=torch.float32, device=device
        ),
        "y_bf16": torch.empty(
            num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
        ),
    }
    # FlashInfer before #8199 takes at most 64 tokens per SM in one launch, so
    # larger batches run as several launches over consecutive token ranges;
    # mHC is per token, so this is exact.
    step = _MAX_TOKENS_PER_SM * num_sms
    for lo in range(0, num_tokens, step):
        hi = min(lo + step, num_tokens)
        splits, grid = _launch_grid(hi - lo, num_sms)
        split_barriers, launch_epochs = (
            (None, None) if counters is None else counters[grid]
        )
        plan = prepare_mega_mhc(
            x=x_flat[lo:hi],
            residual=residual_flat[lo:hi],
            post_mix=post_flat[lo:hi],
            comb_res_mix=comb_flat[lo:hi],
            shifted_prev_mix=None,
            fn=fn,
            mix_scales=hc_scale,
            mix_bases=hc_base,
            rmsnorm_weight=norm_weight,
            hc_mult=HC_MULT,
            hc_norm_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_post_scale=hc_post_mult_value,
            sinkhorn_eps=hc_sinkhorn_eps,
            num_sinkhorn_iters=sinkhorn_repeat,
            rmsnorm_eps=norm_eps,
            rmsnorm_scale=1.0,
            sf_layout="bf16",
            out={name: tensor[lo:hi] for name, tensor in out.items()},
            scratch=(
                None
                if scratch is None
                else scratch[: rules.mhc_infer_scratch_floats(hi - lo, splits)]
            ),
            split_barriers=split_barriers,
            launch_epochs=launch_epochs,
            fnp=fn_split,
        )
        # Counters of another grid would leave the cooperative kernel waiting.
        if plan.config["num_launch_sms"] != grid:
            raise RuntimeError(
                f"FlashInfer mega mHC launches {plan.config['num_launch_sms']} "
                f"CTAs for {hi - lo} tokens; its host rules give {grid}."
            )
        plan.run()
    return (
        out["new_residual"].view(*outer_shape, HC_MULT, HIDDEN_SIZE),
        out["new_post_mix"].view(*outer_shape, HC_MULT, 1),
        out["new_comb_res_mix"].view(*outer_shape, HC_MULT, HC_MULT),
        out["y_bf16"].view(*outer_shape, HIDDEN_SIZE),
    )


def _mhc_fused_post_pre_flashinfer_mega_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    fn_split: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    return (
        torch.empty_like(residual),
        residual.new_empty((*outer_shape, hc_mult, 1), dtype=torch.float32),
        residual.new_empty((*outer_shape, hc_mult, hc_mult), dtype=torch.float32),
        residual.new_empty((*outer_shape, hidden_size), dtype=torch.bfloat16),
    )


direct_register_custom_op(
    op_name="mhc_fused_post_pre_flashinfer_mega",
    op_func=mhc_fused_post_pre_flashinfer_mega,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_flashinfer_mega_fake,
)


class MegaMHCKernel(VllmJitKernel["MegaMHCKernel.CompileKey"]):
    """Builds every FlashInfer mega mHC program the op can select.

    FlashInfer picks the program from the token count, and builds it on first
    use. Startup reaches only the profile run and the CUDA graph sizes, so a
    program for a size between them would be built mid-serving.
    """

    @dataclass(frozen=True)
    class CompileKey:
        program: str
        defines: tuple[tuple[str, Any], ...]
        num_sms: int
        # A launch size that selects the program; not part of its identity.
        num_tokens: int = field(compare=False)

    def get_warmup_keys(self, *, max_num_batched_tokens: int) -> list[CompileKey]:
        rules = _flashinfer_mega_mhc()[1]
        arch, num_sms = rules.device_facts(torch.accelerator.current_device_index())
        launch = inspect.signature(rules.select_program).parameters
        max_tokens = min(max_num_batched_tokens, _MAX_TOKENS_PER_SM * num_sms)
        keys: dict[MegaMHCKernel.CompileKey, None] = {}
        # Chunk remainders make every launch size up to max_tokens reachable.
        for t in range(1, max_tokens + 1):
            splits = _launch_grid(t, num_sms)[0]
            kwargs = {"regime": rules.mhc_infer_tail_program(t, num_sms)}
            if "tokens" in launch:
                kwargs.update(tokens=t, deterministic=False)
            program, defines = rules.select_program(
                HIDDEN_SIZE, splits, False, False, "bf16", arch, **kwargs
            )
            key = self.CompileKey(program, tuple(sorted(defines.items())), num_sms, t)
            keys.setdefault(key, None)
        return list(keys)

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        prepare_mega_mhc, rules = _flashinfer_mega_mhc()
        t, h = compile_key.num_tokens, HIDDEN_SIZE
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        fp32 = dict(dtype=torch.float32, device="cuda")
        # Preparing a plan builds (or loads) its program; it is not run.
        plan = prepare_mega_mhc(
            x=torch.zeros(t, h, **bf16),
            residual=torch.zeros(t, HC_MULT, h, **bf16),
            post_mix=torch.zeros(t, HC_MULT, 1, **fp32),
            comb_res_mix=torch.zeros(t, HC_MULT, HC_MULT, **fp32),
            shifted_prev_mix=None,
            fn=torch.zeros(rules.FN_ROWS, HC_MULT * h, **fp32),
            mix_scales=torch.zeros(3, **fp32),
            mix_bases=torch.zeros(rules.FN_ROWS, **fp32),
            rmsnorm_weight=torch.zeros(h, **bf16),
            sf_layout="bf16",
            fnp=torch.zeros(rules.FNP_PIECES, rules.FNP_ROWS, HC_MULT * h, **bf16),
        )
        if plan.program != compile_key.program:
            raise RuntimeError(
                f"FlashInfer mega mHC selected {plan.program} for "
                f"{t} tokens; warmup expected {compile_key.program}."
            )
        self._compiled_cache[compile_key] = plan.library_path


MEGA_MHC_KERNEL = MegaMHCKernel()
