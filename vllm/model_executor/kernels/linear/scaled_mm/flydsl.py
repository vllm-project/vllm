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


_MX_GATE_UP = (20480, 2560)  # merged gate/up of Gemma-4-E4B
_MX_DOWN = (2560, 10240)
_MX_UNIT_SCALES: dict[tuple, torch.Tensor] = {}



def _mx_unit_scale(device: torch.device, n: int, k: int) -> torch.Tensor:
    key = (device, n, k)
    t = _MX_UNIT_SCALES.get(key)
    if t is None:
        t = torch.full(
            (-(-n // 256) * 256 * (k // 32),), 127, dtype=torch.uint8, device=device
        )
        _MX_UNIT_SCALES[key] = t
    return t


def _flydsl_gelu_mlp_mx_impl(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    w1: torch.Tensor,
    w1_shuffled: torch.Tensor,
    w1_s: torch.Tensor,
    w1_il: torch.Tensor,
    w1_s_il: torch.Tensor,
    w1_il_dec: torch.Tensor,
    w1_s_il_dec: torch.Tensor,
    w2: torch.Tensor,
    w2_shuffled: torch.Tensor,
    w2_s: torch.Tensor,
    w2_unit: torch.Tensor,
) -> torch.Tensor:
    m = x_q.shape[0]
    inter = w2_shuffled.shape[1]
    dec_min = envs.VLLM_ROCM_FLYDSL_MX_MLP_DECODE_MIN_TOKENS
    if dec_min and dec_min <= m <= 256 and w1_il_dec.numel() > 0:
        from aiter.ops.flydsl.gemm_a8w8_gelu_mx import flydsl_gate_up_gelu_mx_dec
        from aiter.ops.flydsl.gemm_mxfp8_down_ws import flydsl_mx_down_ws

        act = torch.empty((-(-m // 16) * 16 * inter,), dtype=x_q.dtype, device=x_q.device)
        act_sc = torch.empty(-(-m // 256) * 256 * (inter // 32), dtype=torch.uint8, device=x_q.device)
        flydsl_gate_up_gelu_mx_dec(x_q, w1_il_dec, x_s, w1_s_il_dec, act, act_sc)
        out = torch.empty((m, w2_shuffled.shape[0]), dtype=torch.bfloat16, device=x_q.device)
        flydsl_mx_down_ws(act, act_sc, w2_shuffled, w2_s.reshape(-1), out)
        return out
    if m >= envs.VLLM_ROCM_FLYDSL_MX_MLP_MIN_TOKENS:
        from aiter.ops.flydsl.gemm_a8w8_gelu_mx import (
            flydsl_gate_up_gelu_mx,
            flydsl_mx_down,
        )

        act = torch.empty((m, inter), dtype=x_q.dtype, device=x_q.device)
        act_sc = torch.empty(
            -(-m // 256) * 256 * (inter // 32), dtype=torch.uint8, device=x_q.device
        )
        flydsl_gate_up_gelu_mx(x_q, w1_il, x_s, w1_s_il, act, act_sc)
        out = torch.empty((m, w2_shuffled.shape[0]), dtype=torch.bfloat16, device=x_q.device)
        flydsl_mx_down(act, act_sc, w2_shuffled, w2_unit, w2_s.reshape(-1), out)
        return out
    from vllm.model_executor.layers.fusion.gelu_tanh_mul_fp8_quant import (
        _gelu_tanh_mul_fp8_per_token_quant_impl,
    )

    y = _flydsl_ptpc_gemm_impl(x_q, w1, w1_shuffled, x_s, w1_s)
    q, s = _gelu_tanh_mul_fp8_per_token_quant_impl(y)
    return _flydsl_ptpc_gemm_impl(q, w2, w2_shuffled, s, w2_s)


def _flydsl_gelu_mlp_mx_fake(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    w1: torch.Tensor,
    w1_shuffled: torch.Tensor,
    w1_s: torch.Tensor,
    w1_il: torch.Tensor,
    w1_s_il: torch.Tensor,
    w1_il_dec: torch.Tensor,
    w1_s_il_dec: torch.Tensor,
    w2: torch.Tensor,
    w2_shuffled: torch.Tensor,
    w2_s: torch.Tensor,
    w2_unit: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x_q.shape[0], w2_shuffled.shape[0]), dtype=torch.bfloat16, device=x_q.device
    )


direct_register_custom_op(
    op_name="rocm_flydsl_gelu_mlp_mx",
    op_func=_flydsl_gelu_mlp_mx_impl,
    fake_impl=_flydsl_gelu_mlp_mx_fake,
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
        if envs.VLLM_ROCM_FLYDSL_MX_MLP and (w.shape[1], w.shape[0]) == _MX_GATE_UP:
            from aiter.ops.flydsl.gemm_a8w8_gelu_mx import prepare_gate_up_weight

            _, w_s_name, *_ = self.layer_param_names
            w_il, w_s_il = prepare_gate_up_weight(
                w.t().contiguous(), getattr(layer, w_s_name).reshape(-1)
            )
            layer.register_parameter(
                f"{w_name}_il", torch.nn.Parameter(w_il, requires_grad=False)
            )
            layer.register_parameter(
                f"{w_name}_scale_il", torch.nn.Parameter(w_s_il, requires_grad=False)
            )
            if envs.VLLM_ROCM_FLYDSL_MX_MLP_DECODE_MIN_TOKENS:
                from aiter.ops.flydsl.gemm_a8w8_gelu_mx import prepare_gate_up_weight_dec

                w_il_dec, w_s_il_dec = prepare_gate_up_weight_dec(
                    w.t().contiguous(), getattr(layer, w_s_name).reshape(-1)
                )
            else:
                w_il_dec = torch.empty(0, dtype=w.dtype, device=w.device)
                w_s_il_dec = torch.empty(0, dtype=torch.float32, device=w.device)
            layer.register_parameter(
                f"{w_name}_il_dec", torch.nn.Parameter(w_il_dec, requires_grad=False)
            )
            layer.register_parameter(
                f"{w_name}_scale_il_dec", torch.nn.Parameter(w_s_il_dec, requires_grad=False)
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


def _gelu_mlp_mx_layers(gate_up_proj, down_proj, act_fn):
    from vllm.model_executor.layers.activation import GeluAndMul

    if not envs.VLLM_ROCM_FLYDSL_MX_MLP:
        return None
    if not (isinstance(act_fn, GeluAndMul) and act_fn.approximate == "tanh"):
        return None
    k1 = getattr(getattr(gate_up_proj, "quant_method", None), "fp8_linear", None)
    k2 = getattr(getattr(down_proj, "quant_method", None), "fp8_linear", None)
    if not (
        isinstance(k1, FlyDSLPerTokenFp8ScaledMMLinearKernel)
        and isinstance(k2, FlyDSLPerTokenFp8ScaledMMLinearKernel)
    ):
        return None
    w1_name = k1.layer_param_names[0]
    w2_name = k2.layer_param_names[0]
    w1_il = getattr(gate_up_proj, f"{w1_name}_il", None)
    w2_sh = getattr(down_proj, f"{w2_name}_shuffled", None)
    if w1_il is None or w2_sh is None:
        return None
    if tuple(w2_sh.shape) != _MX_DOWN or getattr(gate_up_proj, "bias", None) is not None:
        return None
    if getattr(down_proj, "bias", None) is not None or down_proj.tp_size != 1:
        return None
    return k1, k2, w1_name, w2_name, w1_il, w2_sh


def gelu_mlp_mx_supported(gate_up_proj, down_proj, act_fn) -> bool:
    """True when maybe_fused_gelu_mlp_mx takes a bf16 activation through the FlyDSL MX MLP."""
    return _gelu_mlp_mx_layers(gate_up_proj, down_proj, act_fn) is not None


def maybe_fused_gelu_mlp_mx(
    gate_up_proj, down_proj, act_fn, x: torch.Tensor | None, x_quant=None
):
    """The gated MLP ``down(gelu_tanh(gate) * up)`` of two FP8 per-token/per-channel layers through the FlyDSL
    fused gate_up + MX-quant and MX down kernels for prefill-size token counts (smaller ones keep the unfused
    path inside the op). Returns None when the layers or the activation do not qualify. ``x_quant`` is an
    already quantized (fp8 data, per-token scale) activation, used instead of quantizing ``x``."""
    layers = _gelu_mlp_mx_layers(gate_up_proj, down_proj, act_fn)
    if layers is None or (x_quant is None and x.dtype != torch.bfloat16):
        return None
    k1, k2, w1_name, w2_name, w1_il, w2_sh = layers
    w1, w1_s, x_s, x_s_ub = k1._get_layer_params(gate_up_proj)
    w2, w2_s, *_ = k2._get_layer_params(down_proj)
    if x_quant is not None:
        x_q, x_s = x_quant
    else:
        x_2d = x.view(-1, x.shape[-1])
        x_q, x_s = k1.quant_fp8(x_2d, x_s, x_s_ub)
    if x_s.dim() == 1:
        x_s = x_s.view(-1, 1)
    out = torch.ops.vllm.rocm_flydsl_gelu_mlp_mx(
        x_q,
        x_s,
        w1,
        getattr(gate_up_proj, f"{w1_name}_shuffled"),
        w1_s,
        w1_il,
        getattr(gate_up_proj, f"{w1_name}_scale_il"),
        getattr(gate_up_proj, f"{w1_name}_il_dec"),
        getattr(gate_up_proj, f"{w1_name}_scale_il_dec"),
        w2,
        w2_sh,
        w2_s,
        _mx_unit_scale(w2_sh.device, *_MX_DOWN),
    )
    return out.view(*x_q.shape[:-1], out.shape[-1])
