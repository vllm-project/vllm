# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token x per-channel FP8 GEMM through AITER's FlyDSL preshuffle kernels
for the token counts where hipBLASLt's row-wise ``torch._scaled_mm`` is far from
the roofline (decode and small prefill chunks), keeping ``torch._scaled_mm``
for the rest.  Tuned per (N, K) and token-count range in ``_CONFIGS``; a shape
without an entry is not handled by this kernel (the row-wise torch kernel
takes it).  Needs a second, 16x16 pre-shuffled copy of the weight.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .pytorch import RowWiseTorchFP8ScaledMMLinearKernel, _get_num_tokens
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig

logger = init_logger(__name__)

# (tile_m, tile_n, tile_k, use_async_copy, waves_per_eu, xcd_swizzle, lds_stage, split_k)
_Cfg = tuple[int, int, int, int, int, int, int, int]

# (N, K) -> [(max_tokens, cfg or None), ...] ascending: the first entry with
# tokens <= max_tokens wins; None (and more tokens than the last entry) means
# torch._scaled_mm.  Gemma-4-E4B gate_up_proj (N=20480, K=2560) and down_proj
# (N=2560, K=10240) on MI350P (gfx950, 128 CUs), tuned by timing candidates in a
# CUDA graph with rotating weights (iter-10 microbench/tune).
_CONFIGS: dict[tuple[int, int], list[tuple[int, _Cfg | None]]] = {
    (20480, 2560): [
        (32, (32, 64, 128, 1, 3, 0, 1, 1)),
        (64, (64, 192, 128, 0, 3, 0, 2, 1)),
        (128, (128, 192, 128, 1, 0, 0, 2, 1)),
        (160, (160, 192, 128, 1, 1, 0, 2, 1)),
        (192, None),
        (256, (128, 320, 128, 0, 0, 4, 2, 1)),
        (288, (160, 192, 128, 1, 1, 0, 2, 1)),
        (384, (128, 256, 128, 1, 2, 4, 2, 1)),
        (480, None),
        (512, (128, 256, 128, 1, 2, 4, 2, 1)),
    ],
    (2560, 10240): [
        (8, (16, 32, 1024, 0, 2, 0, 2, 1)),
        (16, (16, 32, 1024, 0, 1, 0, 1, 1)),
        (32, (32, 128, 256, 0, 2, 0, 2, 4)),
        (112, (128, 64, 256, 1, 1, 0, 2, 2)),
        (192, (64, 128, 256, 1, 1, 4, 2, 2)),
        (256, None),
        (320, (64, 64, 256, 1, 1, 4, 2, 1)),
    ],
}


# (N, K) -> {tokens: hipBLASLt solution index} for the CUDA-graph capture sizes
# where a specific row-wise hipBLASLt solution (hipb_findallsols, timed per M)
# beats both torch._scaled_mm's heuristic pick and the FlyDSL ranges above.
# Solution indices belong to the installed hipBLASLt build.
_HIPB_SOLUTIONS: dict[tuple[int, int], dict[int, int]] = {
    (20480, 2560): {
        168: 33923,
        176: 33923,
        184: 33923,
        192: 33923,
        224: 33819,
        256: 33818,
        416: 33797,
        448: 33819,
    },
    (2560, 10240): {
        40: 33832,
        48: 33832,
        56: 33832,
        64: 33824,
        72: 33831,
        80: 33831,
        88: 33831,
        96: 33831,
        120: 33830,
        128: 33822,
        200: 33827,
        208: 33822,
        232: 33822,
        240: 33822,
        248: 33822,
        256: 33822,
        336: 34120,
        352: 34120,
        368: 33830,
        384: 33804,
    },
}
_hipb_ready = False
_hipb_failed: set[tuple[int, int, int]] = set()


def _ensure_hipb() -> None:
    global _hipb_ready
    if not _hipb_ready:
        import aiter

        aiter.hipb_create_extension()
        _hipb_ready = True


def _lookup(m: int, n: int, k: int) -> _Cfg | None:
    for max_m, cfg in _CONFIGS.get((n, k), ()):
        if m <= max_m:
            return cfg
    return None


def _flydsl_ptpc_gemm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    B_shuffled: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    m = A.shape[0]
    n = B_shuffled.shape[0]
    k = A.shape[1]
    sol = _HIPB_SOLUTIONS.get((n, k), {}).get(m)
    if sol is not None and (m, n, k) not in _hipb_failed:
        try:
            import aiter

            _ensure_hipb()
            return aiter.hipb_mm(
                A,
                B,
                solution_index=sol,
                out_dtype=torch.bfloat16,
                scaleA=As,
                scaleB=Bs.t(),
                bpreshuffle=False,
            )
        except Exception as e:
            _hipb_failed.add((m, n, k))
            logger.warning_once("hipBLASLt solution %d failed for M=%d N=%d K=%d: %s", sol, m, n, k, e)
    cfg = _lookup(m, n, k)
    if cfg is None:
        return torch._scaled_mm(
            A,
            B,
            out_dtype=torch.bfloat16,
            scale_a=As,
            scale_b=Bs.t(),
        )
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    tile_m, tile_n, tile_k, async_copy, wpe, xcd, lds_stage, split_k = cfg
    out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    flydsl_preshuffle_gemm_a8(
        A,
        B_shuffled,
        As,
        Bs,
        out,
        tile_m,
        tile_n,
        tile_k,
        async_copy,
        wpe,
        xcd,
        lds_stage,
        True,
        split_k=split_k,
    )
    return out


def _flydsl_ptpc_gemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    B_shuffled: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (A.shape[0], B_shuffled.shape[0]), dtype=torch.bfloat16, device=A.device
    )


direct_register_custom_op(
    op_name="rocm_flydsl_ptpc_fp8_gemm",
    op_func=_flydsl_ptpc_gemm_impl,
    fake_impl=_flydsl_ptpc_gemm_fake,
)


class FlyDSLPerTokenFp8ScaledMMLinearKernel(RowWiseTorchFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not envs.VLLM_ROCM_USE_FLYDSL_PTPC_GEMM:
            return False, "disabled by VLLM_ROCM_USE_FLYDSL_PTPC_GEMM=0."
        ok, reason = RowWiseTorchFP8ScaledMMLinearKernel.is_supported(
            compute_capability
        )
        if not ok:
            return ok, reason
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False, "the tuned FlyDSL configs are for gfx950."
        try:
            import aiter.ops.flydsl.gemm_kernels  # noqa: F401
        except Exception:
            return False, "requires aiter with FlyDSL support."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        ok, reason = RowWiseTorchFP8ScaledMMLinearKernel.can_implement(c)
        if not ok:
            return ok, reason
        if not (
            c.activation_quant_key.scale.group_shape.is_per_token()
            and c.weight_quant_key.scale.group_shape.is_per_channel()
        ):
            return False, "requires per token activation and per channel weight scales."
        if c.out_dtype is not torch.bfloat16:
            return False, "requires bfloat16 output dtype."
        if c.weight_shape is None:
            return False, "weight_shape is required."
        n, k = c.weight_shape
        if (n, k) not in _CONFIGS:
            return False, f"no tuned FlyDSL configuration for N={n}, K={k}."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_name, *_ = self.layer_param_names
        w, *_ = self._get_layer_params(layer)
        # w is the [K, N] view of the [N, K] quantized weight
        from aiter.ops.shuffle import shuffle_weight

        _ensure_hipb()
        shuffled = shuffle_weight(w.t().contiguous(), layout=(16, 16))
        layer.register_parameter(
            f"{w_name}_shuffled", torch.nn.Parameter(shuffled, requires_grad=False)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        w_shuffled = getattr(layer, f"{self.layer_param_names[0]}_shuffled")

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            x_data, x_s = qa.data, qa.scale
            orig_shape, orig_dtype = qa.orig_shape, qa.orig_dtype
        else:
            assert isinstance(x, torch.Tensor)
            x_data = x
            orig_shape, orig_dtype = x.shape, x.dtype

        x_2d = x_data.view(-1, x_data.shape[-1])
        output_shape = [*orig_shape[:-1], w.shape[1]]
        out_dtype = orig_dtype if self.config.out_dtype is None else self.config.out_dtype

        x_2d_q = x_2d
        if qa is None:
            x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)
        if bias is not None or out_dtype != torch.bfloat16:
            return self.apply_scaled_mm(
                A=x_2d_q,
                B=w,
                out_dtype=out_dtype,
                As=x_s,
                Bs=w_s,
                bias=bias,
                output_shape=output_shape,
            )
        if x_s.dim() == 1:
            x_s = x_s.view(-1, 1)
        output = torch.ops.vllm.rocm_flydsl_ptpc_fp8_gemm(
            x_2d_q, w, w_shuffled, x_s, w_s
        )
        num_tokens = _get_num_tokens(output_shape)
        return torch.narrow(output, 0, 0, num_tokens).view(*output_shape)
