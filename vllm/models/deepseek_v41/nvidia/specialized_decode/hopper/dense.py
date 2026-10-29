# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch dense FP8 linears of the Hopper executor.

A stock SM90 FP8 linear is Humming W8A16: BF16 activations times FP8 weights
with UE8M0 scales per 32 k. ``DenseFp8Linear`` runs one, for up to
``MAX_ROWS`` rows, on the kernel ``DISPATCH`` picks for its seam and row count,
reading the linear's Humming weight and scales in place:

- the CuTe DSL kernel (``dense_gemm_cute``) with a geometry (``Cute``), which
  can also publish its output into the fused all-reduce + mHC mailbox;
- Humming with a tuned tiling (``use_pdl`` set), or with its own heuristic
  tiling (``HEURISTIC``) and ``use_pdl`` set.

The choices up to 48 rows were measured in a PDL decode-layer chain on H200 at
TP4, the 96-, 144- and 192-row Humming tilings in isolation; a row count between two
entries takes the larger entry's choice. The CuTe kernel stages its rows of X
in shared memory, which bounds it to 48 rows. Every kernel here
uses programmatic dependent launch. CuTe kernels compile, and Humming kernels
JIT-compile, on first use: the executor warms both up before CUDA graph
capture (``warm_up``).

``wo_a`` is the exception to reading in place: the stock SM90 layer keeps it
only as BF16, dequantized at load. ``fp8_wo_a`` rebuilds the exact FP8 weight
and its scales in the Humming layout, a second copy of the weight.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Rows of the largest call a seam takes.
MAX_ROWS = 192


class Cute(NamedTuple):
    """A CuTe DSL launch geometry (``DenseFp8Gemm``)."""

    tw: int
    S: int
    NS: int
    wm: int
    kg: int
    sk: int
    prec: int
    # The CTAs of a K slice per cluster, which multicast X to each other.
    cn: int = 1


def _humming(
    mma_type: str,
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
    num_stages: int,
    *,
    ctas_per_sm: int = 1,
    stream_k: bool = False,
    raster_group_m: int | None = None,
) -> dict:
    """A Humming tuning config; ``wgmma`` ones use TMA and warp specialization."""
    wgmma = mma_type == "wgmma"
    config = {
        "mma_type": mma_type,
        "block_shape": list(block_shape),
        "warp_shape": list(warp_shape),
        "num_stages": num_stages,
        "use_stream_k": stream_k,
        "num_ctas_per_sm": ctas_per_sm,
        "use_tma": wgmma,
        "use_warp_spec": wgmma,
        "use_pdl": True,
    }
    if raster_group_m is not None:
        config.update(use_f16_accum=False, raster_group_m=raster_group_m)
    return config


# Humming's own tiling for the call, with use_pdl set.
HEURISTIC: dict = {}
_M8 = _humming("mma", (8, 64, 128), (8, 32, 32), 8)
_M8S6 = _humming("mma", (8, 64, 128), (8, 32, 32), 6)
_M8C2 = _humming("mma", (8, 64, 128), (8, 32, 32), 8, ctas_per_sm=2)
_M16S6 = _humming("mma", (16, 64, 128), (16, 32, 32), 6)
_M16 = _humming("mma", (16, 64, 128), (16, 32, 32), 8)
_M16C2 = _humming("mma", (16, 64, 128), (16, 32, 32), 8, ctas_per_sm=2)
_M16K256 = _humming("mma", (16, 64, 256), (16, 32, 64), 4, stream_k=True)


def _wgmma(
    rows: int, block_k: int, stages: int, stream_k: bool, ctas_per_sm: int = 1
) -> dict:
    return _humming(
        "wgmma",
        (rows, 128, block_k),
        (rows, 32, 64),
        stages,
        ctas_per_sm=ctas_per_sm,
        stream_k=stream_k,
        raster_group_m=1,
    )


_WG48 = _wgmma(48, 128, 4, True)

# Up to 8 rows the ring holds a CTA's whole weight slice, all loaded during
# the previous kernel (``DenseFp8Gemm`` issues weights before its grid wait).
_WQA_F16 = Cute(2, 8, 5, 2, 2, 8, 0)
_WQA_BF16 = Cute(1, 16, 3, 2, 4, 4, 1)
_WQB = Cute(1, 16, 5, 2, 4, 1, 0)
_IWQB = Cute(1, 16, 4, 2, 4, 2, 0)
_WOA_F16 = Cute(1, 16, 4, 2, 4, 4, 0)
_WOA_BF16 = Cute(1, 16, 3, 2, 4, 3, 1)
_WOB_F16 = Cute(1, 16, 4, 2, 4, 2, 0)
_WOB_F16N = Cute(1, 16, 4, 1, 2, 2, 0)
_WOB_BF16 = Cute(1, 16, 4, 1, 2, 1, 1)
_WOB_BF16N = Cute(1, 16, 3, 2, 4, 2, 1)
_GU_F16 = Cute(1, 8, 4, 1, 2, 8, 0)
_GU_F16S6 = Cute(1, 8, 6, 1, 2, 8, 0)
_GU_BF16 = Cute(1, 8, 4, 1, 2, 8, 1)
_WKV = Cute(2, 16, 4, 2, 2, 2, 0)

# Per seam, rows -> choice.
DISPATCH: dict[str, dict[int, Cute | dict]] = {
    "wqa_wkv": {
        1: Cute(1, 16, 5, 2, 4, 4, 0),
        **dict.fromkeys(range(2, 9), _WQA_F16),
        9: Cute(2, 8, 4, 2, 2, 8, 1),
        **dict.fromkeys((*range(10, 17), 24), _WQA_BF16),
        48: Cute(1, 8, 4, 1, 2, 4, 1),
        96: _wgmma(96, 64, 4, True),
        144: _wgmma(144, 64, 4, True),
        192: _wgmma(96, 64, 3, True, 2),
    },
    "wq_b": {
        **dict.fromkeys(range(1, 9), _WQB),
        **dict.fromkeys((9, 11, 12, 13, 14, 16), _M16S6),
        **dict.fromkeys((10, 15), _M16),
        24: Cute(1, 16, 3, 2, 4, 1, 1),
        48: Cute(1, 16, 4, 1, 2, 1, 1),
        96: _wgmma(96, 128, 4, False),
        144: _wgmma(144, 128, 4, False),
        192: _wgmma(96, 64, 4, False),
    },
    "index_wq_b": {
        **dict.fromkeys((1, 2, 3, 5), _IWQB),
        **dict.fromkeys((4, 6, 8), _M8),
        7: _M8S6,
        **dict.fromkeys((9, 15), _M16S6),
        **dict.fromkeys((*range(10, 15), 16, 24, 48), _M16),
        96: _wgmma(96, 64, 3, True, 2),
        144: _wgmma(144, 64, 3, True, 2),
    },
    "wo_a": {
        **dict.fromkeys((1, 2, 3, 4, 6, 8), _WOA_F16),
        5: Cute(1, 16, 4, 1, 2, 4, 0),
        7: Cute(2, 8, 4, 2, 2, 8, 0),
        **dict.fromkeys((9, 10), Cute(1, 16, 3, 2, 4, 4, 1)),
        **dict.fromkeys((11, 12, 14, 15, 16), _WOA_BF16),
        **dict.fromkeys((13, 24), Cute(1, 16, 4, 1, 2, 3, 1)),
        48: Cute(1, 8, 4, 1, 2, 3, 1),
    },
    "wo_b": {
        **dict.fromkeys((1, 2, 3, 4, 7, 8), _WOB_F16),
        **dict.fromkeys((5, 6), _WOB_F16N),
        **dict.fromkeys((9, 10, 11, 14), _WOB_BF16),
        **dict.fromkeys((12, 13, 15, 16), _WOB_BF16N),
        24: Cute(1, 16, 3, 2, 4, 1, 1),
        48: _WG48,
        96: _wgmma(96, 64, 3, True, 2),
        144: _wgmma(144, 64, 3, True, 2),
        192: _wgmma(96, 128, 4, False),
    },
    "shared_gate_up": {
        **dict.fromkeys((1, 2, 3, 5, 8), _GU_F16),
        **dict.fromkeys((4, 6, 7), _GU_F16S6),
        **dict.fromkeys((*range(9, 14), 15), _GU_BF16),
        14: _M16K256,
        **dict.fromkeys((16, 24), Cute(1, 8, 3, 1, 2, 8, 1)),
        48: Cute(1, 8, 4, 1, 2, 4, 1),
        96: _wgmma(96, 64, 3, True),
        144: _wgmma(144, 64, 3, True),
        192: _wgmma(96, 128, 4, True),
    },
    "shared_down": {
        **dict.fromkeys(range(1, 9), _M8C2),
        **dict.fromkeys(range(9, 17), _M16C2),
        24: HEURISTIC,
        48: Cute(1, 16, 4, 1, 2, 1, 1),
        96: _wgmma(96, 128, 4, False),
        144: _wgmma(144, 64, 4, False),
        192: _wgmma(96, 128, 4, False),
    },
    "engram_wkv": dict.fromkeys(range(1, 7), _WKV),
    # The DSpark draft's [15360 -> 5120] context projection.
    "main_proj": {
        **dict.fromkeys(
            range(1, 9), _humming("mma", (8, 256, 64), (8, 32, 64), 4, stream_k=True)
        ),
        **dict.fromkeys(
            range(9, 17), _humming("mma", (16, 256, 64), (16, 32, 64), 4, stream_k=True)
        ),
        **dict.fromkeys(
            (24, 48), _humming("mma", (48, 256, 64), (48, 32, 64), 8, stream_k=True)
        ),
    },
}

# From 6 rows, the CTAs of a K slice share X by cluster multicast: 4 per
# cluster where the K split allows (measured -0.7 to -1.2 us per layer at 6-16
# rows in the decode-layer chain; no gain at 1 row).
for _seam in ("wq_b", "wo_b", "wo_a"):
    for _rows, _c in DISPATCH[_seam].items():
        if isinstance(_c, Cute) and 6 <= _rows <= 16:
            DISPATCH[_seam][_rows] = _c._replace(cn=min(4, 8 // _c.sk))


def choice(seam: str, rows: int) -> Cute | dict | None:
    """The kernel of ``seam`` for ``rows`` rows, or None above its table."""
    table = DISPATCH[seam]
    above = [m for m in table if m >= rows]
    return table[min(above)] if rows > 0 and above else None


def _staged_rows(k: int, sk: int) -> int:
    return ((k // 16 + sk - 1) // sk + 3) // 2 * 2


def cute_fits(c: Cute, rows: int, n: int, k: int) -> bool:
    """Whether geometry ``c`` launches ``rows`` x ``n`` x ``k`` (tiling,
    registers, at least two 16-k rows per split and shared memory)."""
    nt = (rows + 7) // 8
    if (4 * c.tw) % c.wm or c.S % (2 * c.kg) or n % (64 * c.tw * c.cn):
        return False
    if c.sk * c.cn > 8:
        return False
    if c.wm * nt * 4 > 96 or (k // 16) // c.sk < 2:
        return False
    mpad = 8 * nt
    smem = (
        c.NS * c.S * c.tw * 1088
        + mpad * (_staged_rows(k, c.sk) * 32 + 16)
        + c.kg * mpad * (64 * c.tw + 4) * 4
        + max(c.sk - 1, 0) * mpad * 16 * c.tw * 16
        + 4096
    )
    return smem <= 227 * 1024 - 2048


@cache
def _compiled(nt: int, c: Cute, xr: int, pub: bool):
    import cutlass
    import cutlass.cute as cute

    from ...ops.cute_dsl.primitives import current_cu_stream
    from .dense_gemm_cute import DenseFp8Gemm

    def fake(dtype):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    kernel = DenseFp8Gemm(
        nt, c.tw, c.S, c.NS, c.wm, c.kg, c.sk, xr, c.prec, int(pub), c.cn
    )
    tensors = [
        fake(cutlass.BFloat16),
        fake(cutlass.Int32),
        fake(cutlass.Uint8),
        fake(cutlass.BFloat16),
    ]
    ints = [cutlass.Int32(1)] * 8 + [cutlass.Int64(0), cutlass.Int64(0)]
    ints += [cutlass.Int32(0), cutlass.Int32(0)]
    return cute.compile(
        kernel, *tensors, *ints, current_cu_stream(), options="--enable-tvm-ffi"
    )


class DenseFp8Linear:
    """A low-batch call of one Humming W8A16 linear (or the grouped ``wo_a``).

    Args:
        seam: the ``DISPATCH`` key.
        weight: Humming int32 ``[Kp / 16, Np * 4]``.
        weight_scale: Humming UE8M0 ``[Kp / 32, Np]``.
        n: output columns.
        k: input columns (per group).
        humming: the linear's Humming kernel (``layer_config``,
            ``compute_config``, ``locks``), or None for CuTe only.
        n_group: grouped mode: output columns ``[g * n_group, (g + 1) *
            n_group)`` read input columns ``[g * k, (g + 1) * k)``.

    """

    def __init__(
        self,
        seam: str,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        n: int,
        k: int,
        humming=None,
        n_group: int | None = None,
    ) -> None:
        assert n % 64 == 0 and k % 32 == 0
        self.seam, self.n, self.k, self.n_group = seam, n, k, n_group
        self.weight, self.weight_scale = weight, weight_scale
        self._w = weight.view(torch.int32).reshape(-1)
        self._s = weight_scale.view(torch.uint8).reshape(-1)
        self._np = weight.shape[-1] // 4
        self.humming = humming

    @classmethod
    def of(cls, seam: str, linear: nn.Module) -> "DenseFp8Linear | None":
        """The call of a stock Humming W8A16 ``linear``, or None for another
        kernel."""
        from vllm.model_executor.kernels.linear.mxfp8.humming import (
            HummingMxfp8LinearKernel,
        )

        method = getattr(linear, "scheme", getattr(linear, "quant_method", None))
        kernel = getattr(method, "kernel", None)
        if (
            type(kernel) is not HummingMxfp8LinearKernel
            or getattr(linear, "bias", None) is not None
        ):
            return None
        n = sum(linear.output_partition_sizes)
        k = getattr(linear, "input_size_per_partition", linear.input_size)
        if n % 64 or k % 32:
            return None
        return cls(seam, linear.weight, linear.weight_scale, n, k, kernel)

    def _choice(self, rows: int) -> Cute | dict | None:
        c = choice(self.seam, rows)
        if isinstance(c, Cute):
            return c if cute_fits(c, rows, self.n, self.k) else None
        return c if self.humming is not None else None

    def serves(self, rows: int) -> bool:
        """Whether a call of ``rows`` rows runs here."""
        return self._choice(rows) is not None

    def publishes(self, rows: int) -> bool:
        """Whether a call of ``rows`` rows can publish (CuTe)."""
        return isinstance(self._choice(rows), Cute)

    def __call__(self, x: torch.Tensor, all_reduce=None) -> torch.Tensor:
        """``x @ W^T`` (BF16 ``[rows, k]``, or ``[rows, groups, k]`` in grouped
        mode, unit column stride); with ``all_reduce`` (a ``BoundaryAllReduce``)
        also published into its mailbox, which then only collects it."""
        rows = x.shape[0]
        c = self._choice(rows)
        assert c is not None
        out = torch.empty(rows, self.n, dtype=torch.bfloat16, device=x.device)
        if not isinstance(c, Cute):
            assert all_reduce is None
            self._humming(x, out, c)
            return out
        self._cute(x, out, c, all_reduce)
        if all_reduce is not None:
            all_reduce.mark_published(out)
        return out

    def warm_up(self, rows: int, publish: bool = False) -> None:
        """Compile the CuTe kernel, or JIT-compile Humming by one eager call, of
        a ``rows``-row call, before CUDA graph capture."""
        c = self._choice(rows)
        if isinstance(c, Cute):
            _compiled((rows + 7) // 8, c, _staged_rows(self.k, c.sk), publish)
        elif c is not None:
            x = torch.zeros(
                rows, self.k, dtype=torch.bfloat16, device=self.weight.device
            )
            self._humming(
                x, torch.empty(rows, self.n, dtype=x.dtype, device=x.device), c
            )

    def _humming(self, x: torch.Tensor, out: torch.Tensor, config: dict) -> None:
        from vllm.utils.humming import get_heuristics_config, humming_forward

        kernel = self.humming
        if not config:
            config = dict(
                get_heuristics_config(kernel.layer_config, shape_m=x.shape[0]),
                use_pdl=True,
            )
        humming_forward(
            kernel.layer_config,
            inputs=x,
            weight=self.weight,
            weight_scale=self.weight_scale,
            locks=kernel.locks,
            outputs=out,
            compute_config=kernel.compute_config,
            tuning_config=config,
        )

    def _cute(self, x: torch.Tensor, out: torch.Tensor, c: Cute, all_reduce) -> None:
        from ...ops.cute_dsl.primitives import current_cu_stream

        rows = x.shape[0]
        assert x.stride(-1) == 1 and out.stride(-1) == 1
        if x.dim() == 3:
            assert self.n_group is not None and x.shape[1] * self.n_group == self.n
            group_stride = x.stride(1)
            span = (x.shape[1] - 1) * group_stride + x.shape[2]
        else:
            group_stride = self.k if self.n_group else 0
            span = x.shape[1]
        fn = _compiled(
            (rows + 7) // 8, c, _staged_rows(self.k, c.sk), all_reduce is not None
        )
        mc = sst = rank_cap = tp_cap = 0
        if all_reduce is not None:
            assert all_reduce.hidden_size == self.n and rows <= all_reduce.capacity
            mc, sst = all_reduce.publish_target()
            rank_cap = all_reduce.rank * all_reduce.capacity
            tp_cap = all_reduce.tp * all_reduce.capacity
        ldx, ldy = x.stride(0), out.stride(0)
        fn(
            x.as_strided(((rows - 1) * ldx + span,), (1,)),
            self._w,
            self._s,
            out.as_strided(((rows - 1) * ldy + out.shape[1],), (1,)),
            rows,
            self.n,
            self.k,
            self._np,
            ldx,
            ldy,
            self.n_group or self.n,
            group_stride,
            mc,
            sst,
            rank_cap,
            tp_cap,
            current_cu_stream(),
        )


def fp8_wo_a(wo_a: nn.Module, n_groups: int) -> DenseFp8Linear | None:
    """The grouped FP8 call of the stock SM90 BF16 ``wo_a`` (``[n_groups *
    o_lora_rank, k]``), or None when the weight is not an exact MXFP8 dequant.

    Each 1 x 32 group's scale is the smallest power of two that keeps it within
    the e4m3 range; any exact split gives the same products, since the kernel
    applies the scale exactly. The FP8 weight and scales are converted to the
    Humming layout as the stock Humming linears' are.
    """
    w = wo_a.weight
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] % n_groups:
        return None
    n, k = w.shape
    if n % 64 or k % 32:
        return None
    groups = w.float().view(n, k // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    fp8 = (groups / torch.exp2(exponent).unsqueeze(-1)).to(torch.float8_e4m3fn)
    back = fp8.float() * torch.exp2(exponent).unsqueeze(-1)
    if not torch.equal(back.view(n, k).to(w.dtype), w):
        logger.warning_once(
            "The Hopper executor keeps wo_a on the BF16 bmm: its weight is not "
            "an exact MXFP8 dequant."
        )
        return None
    weight, scale = _humming_layout(
        fp8.view(n, k), (exponent + 127).to(torch.uint8), n, k
    )
    return DenseFp8Linear("wo_a", weight, scale, n, k, n_group=n // n_groups)


def _humming_layout(
    fp8: torch.Tensor, scale: torch.Tensor, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp8`` [n, k] and UE8M0 ``scale`` [n, k / 32] in the Humming W8A16
    layout, converted as ``HummingMxfp8LinearKernel`` converts a linear."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        convert_linear_layer_to_humming_standard,
        prepare_humming_linear_layer_config,
        quant_key_to_input_schema,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    layer = nn.Module()
    layer.input_size_per_partition = layer.input_size = k
    layer.output_partition_sizes = [n]
    layer.params_dtype = torch.bfloat16
    layer.has_bias = False
    layer.weight = nn.Parameter(fp8, requires_grad=False)
    layer.weight_scale = nn.Parameter(
        scale.view(torch.float8_e8m0fnu), requires_grad=False
    )
    convert_linear_layer_to_humming_standard(
        layer=layer, name_map={"weight": "weight", "weight_scale": "weight_scale"}
    )
    prepare_humming_linear_layer_config(
        layer,
        {
            "quant_method": "humming",
            "dtype": "float8e4m3",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        },
        input_schema=quant_key_to_input_schema(kMxfp8Dynamic),
    )
    return layer.weight.data, layer.weight_scale.data
