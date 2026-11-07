# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MHCFusedPostPreOp on FlashInfer's fused mHC kernel (``flashinfer.mega_mhc``).

One persistent kernel computes a sublayer's mHC post block, the next
sublayer's pre block and its RMSNorm: the work the TileLang path does in two
launches for up to 32 tokens, and in post + projection GEMM + pre beyond that.
FlashInfer ships it as the experimental ``prepare_mega_mhc`` API; its
``sf_layout="bf16"`` program serves hidden size 4096 with four residual
streams in normal (non-delayed) mode, on the architectures FlashInfer lists
for it (SM107a; SM100a/SM103a in newer releases). One launch takes at most 64
tokens per SM (the binding checks it), so larger batches are split into
several launches; mHC is per token, so the split is exact.

A FlashInfer plan binds the tensors it is prepared with, and the activations
are new tensors on every call, so a plan is prepared for each call (under CUDA
graphs that happens once, at capture). The state the kernel keeps between
calls is owned here and handed to every plan:

* ``fnp``: the FP32 projection split into three BF16 pieces, built once per
  projection weight. The weights are parameters and do not change after
  loading.
* ``scratch``: per-launch partial sums; one buffer per device, sized for the
  largest token count seen so far.
* ``split_barriers`` / ``launch_epochs``: cross-CTA counters. Every CTA reads
  and advances its own epoch, so launches may share a pair only if they use
  the same grid. FlashInfer derives the grid from the token count and there
  are only a handful of distinct grids, so one pair is kept per grid.

Sharing is safe because launches on a stream run one after another. The
buffers are allocated on the first eager call (vLLM's profile run comes before
CUDA graph capture) and are never freed, since captured graphs keep their
addresses.

"""

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# The shape FlashInfer's BF16-output mega mHC program serves.
HIDDEN_SIZE = 4096
HC_MULT = 4

__all__ = [
    "mhc_fused_post_pre_flashinfer_mega",
    "use_flashinfer_mega_mhc",
]


@functools.cache
def _flashinfer_mega_mhc():
    """``(prepare_mega_mhc, host rules module)``, or None if FlashInfer lacks them."""
    try:
        from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as rules
        from flashinfer.mega_mhc import prepare_mega_mhc
    except ImportError:
        return None
    if not hasattr(rules, "mhc_infer_num_splits"):
        # A FlashInfer whose mega mHC has no BF16-output (hidden 4096) program.
        return None
    return prepare_mega_mhc, rules


@functools.cache
def _device_supported(device_index: int) -> bool:
    fi = _flashinfer_mega_mhc()
    if fi is None:
        logger.warning_once(
            "VLLM_MHC_USE_FLASHINFER_MEGA=1 but this FlashInfer has no "
            "flashinfer.mega_mhc BF16-output program; using TileLang."
        )
        return False
    rules = fi[1]
    capability = torch.cuda.get_device_capability(device_index)
    arch = rules.CAPABILITIES.get(capability)
    if arch is None or not rules.supported(arch, HIDDEN_SIZE, "bf16"):
        logger.warning_once(
            "VLLM_MHC_USE_FLASHINFER_MEGA=1 but flashinfer.mega_mhc has no "
            "hidden-%d program for compute capability %s; using TileLang.",
            HIDDEN_SIZE,
            capability,
        )
        return False
    logger.info_once(
        "MHCFusedPostPreOp uses FlashInfer mega mHC (%s, hidden %d).",
        arch,
        HIDDEN_SIZE,
    )
    return True


def use_flashinfer_mega_mhc(
    residual: torch.Tensor, norm_weight: torch.Tensor | None
) -> bool:
    """Whether MHCFusedPostPreOp should run FlashInfer's fused kernel.

    The kernel always applies the RMSNorm, so calls without ``norm_weight``
    keep the TileLang path, as do shapes the kernel does not serve.
    """
    if not envs.VLLM_MHC_USE_FLASHINFER_MEGA or norm_weight is None:
        return False
    if (
        residual.shape[-1] != HIDDEN_SIZE
        or residual.shape[-2] != HC_MULT
        or not residual.is_cuda
    ):
        return False
    num_tokens = residual.numel() // (HC_MULT * HIDDEN_SIZE)
    if num_tokens == 0 or num_tokens < envs.VLLM_MHC_FLASHINFER_MEGA_MIN_TOKENS:
        return False
    return _device_supported(residual.device.index)


def _forbid_capture(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"FlashInfer mega mHC: {what} during CUDA graph capture. This "
            "happens on the first eager call, which vLLM's profile run makes "
            "before capturing."
        )


class _Workspace:
    """Per-device state shared by every plan (see the module docstring)."""

    def __init__(self, device: torch.device, rules) -> None:
        _forbid_capture("allocating the shared workspace")
        self.device = device
        self.rules = rules
        _, self.num_sms = rules.device_facts(device.index)
        # From 4 tokens per SM on, the grid is the SM count, so this range
        # yields every grid FlashInfer can pick.
        grids = {self.grid(t)[1] for t in range(1, 4 * self.num_sms + 1)}
        grids.add(self.num_sms)
        self.counters = {grid: self._new_counters() for grid in sorted(grids)}
        self.scratch = torch.empty(0, dtype=torch.float32, device=device)
        # Replaced scratch buffers: captured graphs may still point at them.
        self.retired: list[torch.Tensor] = []
        self.fnp: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    def _new_counters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(
                self.rules.SPLIT_BARRIER_WORDS, dtype=torch.uint64, device=self.device
            ),
            torch.zeros(self.num_sms, dtype=torch.uint64, device=self.device),
        )

    def grid(self, num_tokens: int) -> tuple[int, int]:
        """(split count, grid) FlashInfer's plan will use for ``num_tokens``."""
        splits = self.rules.mhc_infer_num_splits(num_tokens, self.num_sms)
        return splits, self.rules.mhc_infer_num_ctas(num_tokens, self.num_sms, splits)

    def counters_for(self, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        counters = self.counters.get(grid)
        if counters is None:
            _forbid_capture(f"allocating counters for a grid of {grid} CTAs")
            counters = self.counters[grid] = self._new_counters()
        return counters

    def scratch_view(self, floats: int) -> torch.Tensor:
        if floats > self.scratch.numel():
            _forbid_capture(f"growing the scratch buffer to {floats} floats")
            self.retired.append(self.scratch)
            self.scratch = torch.empty(
                max(floats, 2 * self.scratch.numel()),
                dtype=torch.float32,
                device=self.device,
            )
        return self.scratch[:floats]

    def fnp_for(self, fn: torch.Tensor) -> torch.Tensor:
        key = (fn.data_ptr(), tuple(fn.shape))
        fnp = self.fnp.get(key)
        if fnp is None:
            _forbid_capture("splitting an mHC projection into BF16 pieces")
            fnp = self.fnp[key] = self.rules.split_fn(fn)
        return fnp


_WORKSPACES: dict[int, _Workspace] = {}


def _workspace(device: torch.device, rules) -> _Workspace:
    ws = _WORKSPACES.get(device.index)
    if ws is None:
        ws = _WORKSPACES[device.index] = _Workspace(device, rules)
    return ws


def mhc_fused_post_pre_flashinfer_mega(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
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

    Same contract as ``mhc_fused_post_pre_tilelang``: returns residual_cur,
    post_mix_cur, comb_mix_cur and the RMSNorm'd layer_input_cur.
    """
    fi = _flashinfer_mega_mhc()
    assert fi is not None, "use_flashinfer_mega_mhc() must gate this op"
    prepare_mega_mhc, rules = fi
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, HC_MULT, HIDDEN_SIZE).contiguous()
    num_tokens = residual_flat.shape[0]
    device = residual.device
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)

    ws = _workspace(device, rules)
    x_flat = x.reshape(num_tokens, HIDDEN_SIZE).contiguous()
    post_flat = post_layer_mix.reshape(num_tokens, HC_MULT, 1).contiguous()
    comb_flat = comb_res_mix.reshape(num_tokens, HC_MULT, HC_MULT).contiguous()
    norm_weight = norm_weight.contiguous()
    fnp = ws.fnp_for(fn)
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
    # The program's binding requires num_ctas * 64 >= num_tokens and its grid
    # never exceeds the SM count, so one launch serves at most 64 tokens per SM
    # (13568 on a 212-SM R200). Larger batches, i.e. big prefill steps, run as
    # several launches over consecutive token ranges; mHC is per token, so this
    # is exact.
    step = 64 * ws.num_sms
    for lo in range(0, num_tokens, step):
        hi = min(lo + step, num_tokens)
        splits, grid = ws.grid(hi - lo)
        split_barriers, launch_epochs = ws.counters_for(grid)
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
            scratch=ws.scratch_view(rules.mhc_infer_scratch_floats(hi - lo, splits)),
            split_barriers=split_barriers,
            launch_epochs=launch_epochs,
            fnp=fnp,
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
