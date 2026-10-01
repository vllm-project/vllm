# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Epilogue fusion configuration (EFC) for the Lamport TP all-reduce.

Modeled on CUTLASS's Blackwell EFC example
(examples/python/CuTeDSL/cute/blackwell/efc): an epilogue is one Python
function ``epilogue(cfg, *params)`` whose parameter annotations say how the
kernel indexes each tensor. The same function is run three ways:

- ``ANALYSIS``: with dummies, to find which tensors (and which hc streams) are
  read or written and how many row reductions the kernel needs.
- ``DEVICE``: inside the consumer kernel, after the cross-rank reduction, on
  each thread's fragments of the token row (``Frag``).
- ``TORCH``: on [M, hidden] FP32 torch tensors, as the reference.

Unlike a GEMM tile epilogue, a token row is owned by one CTA cluster, so the
epilogue may also reduce over the row (``row_sum``, for RMSNorm) or over
aligned groups of 16..256 elements (``group_max``, for NVFP4 and per-group FP8
scales). All loads are hoisted ahead of the kernel's PDL wait; stores happen
where the epilogue makes them.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
from typing import Any, ClassVar

import torch


class Phase(enum.Enum):
    ANALYSIS = enum.auto()
    DEVICE = enum.auto()
    TORCH = enum.auto()


class Kind:
    """An epilogue parameter annotation: how the kernel indexes the tensor.
    Its methods are the interface the epilogue sees; each phase passes its
    own implementation."""

    is_tensor = True


class Row(Kind):
    """[M, hidden] per token. Reads BF16; writes BF16, FP8 E4M3 or packed
    E2M1 (uint8 [M, hidden // 2], low nibble first)."""

    def load(self) -> Any:
        raise NotImplementedError

    def store(self, value: Any) -> None:
        raise NotImplementedError


class Streams(Kind):
    """[M, S, hidden] BF16; ``x[s]`` is a ``Row``; ``len(x)`` is S."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, stream: int) -> Row:
        raise NotImplementedError


class Weight(Kind):
    """[hidden] BF16, broadcast over tokens."""

    def load(self) -> Any:
        raise NotImplementedError


class PerToken(Kind):
    """[M, *shape] FP32; ``x[i, ...]`` is one scalar per token."""

    def __getitem__(self, index: Any) -> Any:
        raise NotImplementedError


class DeviceScalar(Kind):
    """A one-element FP32 device tensor, e.g. a static quant scale."""

    def load(self) -> Any:
        raise NotImplementedError


class Scalar(Kind):
    """An FP32 passed by value; the epilogue receives the value itself."""

    is_tensor = False


class GroupScale(Kind):
    """One value per (token, group of ``group_size`` elements), set by a
    subclass. Writes FP32, FP8 E4M3 or UE8M0 (uint8). ``swizzled`` selects the
    128x4 tcgen05 scale-factor layout (``scaled_fp4_quant``'s default),
    otherwise [M, hidden // group_size] row-major."""

    group_size: ClassVar[int]
    swizzled: ClassVar[bool] = False

    def store(self, value: Any) -> None:
        raise NotImplementedError


class NVFP4Scale(GroupScale):
    group_size = 16
    swizzled = True


class Group128Scale(GroupScale):
    group_size = 128


BYTE_DTYPES = (
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float4_e2m1fn_x2,
    torch.float8_e8m0fnu,
)


@dataclasses.dataclass
class Param:
    name: str
    kind: Kind
    dtype: torch.dtype | None = None
    shape: tuple[int, ...] = ()
    read: bool = False
    written: bool = False
    streams_read: set[int] = dataclasses.field(default_factory=set)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def _as_kind(annotation) -> Kind:
    if not (isinstance(annotation, type) and issubclass(annotation, Kind)):
        raise TypeError(f"epilogue parameters need a Kind annotation, got {annotation}")
    if issubclass(annotation, GroupScale) and not hasattr(annotation, "group_size"):
        raise TypeError("annotate with a GroupScale subclass that sets group_size")
    return annotation()


def _squeeze_trailing(shape: tuple[int, ...]) -> tuple[int, ...]:
    while len(shape) > 1 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


class Epilogue:
    """An epilogue function bound to concrete dtypes and per-token shapes."""

    def __init__(self, fn, hidden: int, examples: dict[str, Any]) -> None:
        signature = inspect.signature(fn, eval_str=True)
        names = list(signature.parameters)
        if not names or names[0] != "cfg":
            raise TypeError("the first epilogue parameter must be `cfg`")
        self.fn = fn
        self.hidden = hidden
        self.params = [
            Param(name, _as_kind(signature.parameters[name].annotation))
            for name in names[1:]
        ]
        missing = {p.name for p in self.params} - examples.keys()
        if missing:
            raise TypeError(f"missing epilogue arguments: {sorted(missing)}")
        for param in self.params:
            self._bind(param, examples[param.name])
        self.num_row_reductions = 0
        fn(_AnalysisConfig(self), *(_analysis_proxy(p) for p in self.params))
        if not any(p.written for p in self.params):
            raise ValueError("the epilogue stores nothing")

    def _bind(self, param: Param, value) -> None:
        kind, hidden = param.kind, self.hidden
        if not kind.is_tensor:
            if not isinstance(value, (int, float)):
                raise TypeError(f"{param.name} must be a Python float")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{param.name} must be a tensor")
        param.dtype = value.dtype
        if isinstance(kind, Row):
            width = hidden // 2 if _is_fp4(value.dtype) else hidden
            if value.dim() != 2 or value.shape[1] != width:
                raise ValueError(f"{param.name} must be [M, {width}]")
        elif isinstance(kind, Streams):
            if value.dim() != 3 or value.shape[2] != hidden:
                raise ValueError(f"{param.name} must be [M, S, {hidden}]")
            param.shape = (value.shape[1],)
        elif isinstance(kind, Weight):
            if value.shape != (hidden,):
                raise ValueError(f"{param.name} must be [{hidden}]")
        elif isinstance(kind, PerToken):
            if value.dtype != torch.float32:
                raise ValueError(f"{param.name} must be FP32")
            param.shape = _squeeze_trailing(tuple(value.shape[1:]))
        elif isinstance(kind, DeviceScalar):
            if value.numel() != 1 or value.dtype != torch.float32:
                raise ValueError(f"{param.name} must be a one-element FP32 tensor")
        elif isinstance(kind, GroupScale):
            if hidden % kind.group_size or kind.group_size % 8:
                raise ValueError("group_size must divide hidden and be a multiple of 8")
            if kind.group_size > 256:
                raise ValueError("group_size must be at most 256")
            if kind.swizzled and kind.group_size != 16:
                raise ValueError("the swizzled layout is for NVFP4 (group_size 16)")
        if value.dtype not in (torch.bfloat16, torch.float32, *BYTE_DTYPES):
            raise ValueError(f"{param.name}: unsupported dtype {value.dtype}")

    def reference(self, accum: torch.Tensor, values: dict[str, Any]) -> None:
        """Evaluate the epilogue in torch on the FP32 all-reduced ``accum``,
        writing into the output tensors in ``values``."""
        cfg = _TorchConfig(self, accum)
        self.fn(cfg, *(_torch_proxy(cfg, p, values[p.name]) for p in self.params))


def _is_fp4(dtype: torch.dtype) -> bool:
    return dtype in (torch.uint8, torch.float4_e2m1fn_x2)


class Config:
    """The ``cfg`` an epilogue receives. Arithmetic on values uses Python
    operators; everything else goes through these methods."""

    phase: Phase
    hidden: float

    def accum(self):
        """The FP32 cross-rank sum of the BF16 contributions."""
        raise NotImplementedError

    def zeros(self):
        raise NotImplementedError

    def round(self, x, dtype: torch.dtype):
        """Round to ``dtype`` and back to FP32 (RN; satfinite for FP8)."""
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def maximum(self, x, y):
        raise NotImplementedError

    def minimum(self, x, y):
        raise NotImplementedError

    def clamp(self, x, lo: float, hi: float):
        return self.minimum(self.maximum(x, lo), hi)

    def where(self, cond, x, y):
        raise NotImplementedError

    def rsqrt(self, x):
        raise NotImplementedError

    def rcp_approx(self, x):
        """``rcp.approx.ftz.f32``; the torch reference divides exactly."""
        raise NotImplementedError

    def ue8m0_ceil(self, x):
        """The power of two at or above ``max(|x|, 1e-10)``."""
        raise NotImplementedError

    def row_sum(self, x):
        raise NotImplementedError

    def group_max(self, x, group_size: int):
        raise NotImplementedError


class _Dummy:
    """Stands in for every value during analysis."""

    def _any(self, *_):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _any
    __truediv__ = __rtruediv__ = __neg__ = __abs__ = _any
    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _any
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self):
        raise TypeError("epilogues cannot branch on runtime values; use cfg.where")


_DUMMY = _Dummy()


class _AnalysisConfig(Config):
    phase = Phase.ANALYSIS

    def __init__(self, epilogue: Epilogue) -> None:
        self.epilogue = epilogue
        self.hidden = float(epilogue.hidden)

    def _value(self, *_args, **_kwargs):
        return _DUMMY

    accum = zeros = round = abs = maximum = minimum = where = _value
    rsqrt = rcp_approx = ue8m0_ceil = group_max = _value

    def row_sum(self, x):
        self.epilogue.num_row_reductions += 1
        return _DUMMY


class _AnalysisRow:
    def __init__(self, param: Param, stream: int | None = None) -> None:
        self.param, self.stream = param, stream

    def load(self):
        if _is_byte(self.param.dtype):
            raise TypeError(f"{self.param.name}: only BF16 rows can be loaded")
        self.param.read = True
        if self.stream is not None:
            self.param.streams_read.add(self.stream)
        return _DUMMY

    def store(self, _value) -> None:
        self.param.written = True


class _AnalysisStreams:
    def __init__(self, param: Param) -> None:
        self.param = param

    def __len__(self) -> int:
        return self.param.shape[0]

    def __getitem__(self, stream: int) -> _AnalysisRow:
        if not 0 <= stream < len(self):
            raise IndexError(f"{self.param.name}[{stream}]")
        return _AnalysisRow(self.param, stream)


class _AnalysisPerToken:
    def __init__(self, param: Param) -> None:
        self.param = param
        self.shape = param.shape

    def __getitem__(self, index):
        _flat_index(self.param, index)
        self.param.read = True
        return _DUMMY


def _is_byte(dtype) -> bool:
    return dtype in BYTE_DTYPES


def _flat_index(param: Param, index) -> int:
    index = index if isinstance(index, tuple) else (index,)
    if len(index) != len(param.shape):
        raise IndexError(f"{param.name} takes {len(param.shape)} indices")
    flat = 0
    for i, n in zip(index, param.shape):
        if not 0 <= i < n:
            raise IndexError(f"{param.name}[{index}]")
        flat = flat * n + i
    return flat


def _analysis_proxy(param: Param):
    kind = param.kind
    if isinstance(kind, Row | Weight | DeviceScalar | GroupScale):
        return _AnalysisRow(param)
    if isinstance(kind, Streams):
        return _AnalysisStreams(param)
    if isinstance(kind, PerToken):
        return _AnalysisPerToken(param)
    return _DUMMY


# ---------------------------------------------------------------- torch phase


class _TorchConfig(Config):
    phase = Phase.TORCH

    def __init__(self, epilogue: Epilogue, accum: torch.Tensor) -> None:
        self.epilogue = epilogue
        self.hidden = float(epilogue.hidden)
        self._accum = accum.float()
        self.m = accum.shape[0]

    def accum(self):
        return self._accum

    def zeros(self):
        return torch.zeros_like(self._accum)

    def round(self, x, dtype):
        if dtype == torch.float8_e4m3fn:
            x = x.clamp(-448.0, 448.0)
        return torch.as_tensor(x).to(dtype).float()

    def abs(self, x):
        return x.abs()

    def maximum(self, x, y):
        return torch.maximum(torch.as_tensor(x), torch.as_tensor(y).to(x.device))

    def minimum(self, x, y):
        return torch.minimum(torch.as_tensor(x), torch.as_tensor(y).to(x.device))

    def where(self, cond, x, y):
        return torch.where(cond, x, y)

    def rsqrt(self, x):
        return torch.rsqrt(x)

    def rcp_approx(self, x):
        return 1.0 / torch.as_tensor(x, dtype=torch.float32)

    def ue8m0_ceil(self, x):
        x = x.abs().clamp_min(1e-10)
        return torch.exp2(torch.ceil(torch.log2(x)))

    def row_sum(self, x):
        return x.sum(-1, keepdim=True)

    def group_max(self, x, group_size):
        m, h = x.shape
        groups = x.view(m, h // group_size, group_size).amax(-1, keepdim=True)
        return groups.expand(m, h // group_size, group_size).reshape(m, h)


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """RN-even, saturating FP32 -> E2M1 nibbles."""
    a = x.abs()
    code = torch.zeros_like(a, dtype=torch.uint8)
    # Upper bounds of each code under ties-to-even.
    for bound, inclusive in (
        (0.25, True),
        (0.75, False),
        (1.25, True),
        (1.75, False),
        (2.5, True),
        (3.5, False),
        (5.0, True),
    ):
        code += ((a > bound) if inclusive else (a >= bound)).to(torch.uint8)
    # The sign survives rounding to zero, as with cvt.rn.satfinite.e2m1x2.
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def nvfp4_swizzled_offsets(rows: int, cols: int, device) -> torch.Tensor:
    """Flat byte offsets of [rows, cols] scale factors in the 128x4 layout
    (``cvt_quant_to_fp4_get_sf_out_offset``)."""
    num_k_tiles = (cols + 3) // 4
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        ((r >> 7) * num_k_tiles + (c >> 2)) * 512
        + (r & 31) * 16
        + ((r >> 5) & 3) * 4
        + (c & 3)
    )


class _TorchRow:
    def __init__(self, cfg: _TorchConfig, param: Param, tensor: torch.Tensor):
        self.cfg, self.param, self.tensor = cfg, param, tensor

    def load(self):
        return self.tensor.float()

    def store(self, value) -> None:
        dtype, tensor = self.param.dtype, self.tensor
        value = torch.as_tensor(value, device=tensor.device).float()
        value = value.expand(self.cfg.m, self.cfg.epilogue.hidden)
        if _is_fp4(dtype):
            codes = _e2m1_codes(value)
            packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
            tensor.view(torch.uint8).copy_(packed)
        elif dtype == torch.float8_e4m3fn:
            tensor.copy_(value.clamp(-448.0, 448.0).to(dtype))
        else:
            tensor.copy_(value.to(dtype))


class _TorchWeight(_TorchRow):
    def load(self):
        return self.tensor.float().view(1, -1)


class _TorchDeviceScalar(_TorchRow):
    def load(self):
        return self.tensor.float().view(1, 1)


class _TorchStreams:
    def __init__(self, cfg: _TorchConfig, param: Param, tensor: torch.Tensor):
        self.cfg, self.param, self.tensor = cfg, param, tensor

    def __len__(self) -> int:
        return self.param.shape[0]

    def __getitem__(self, stream: int) -> _TorchRow:
        return _TorchRow(self.cfg, self.param, self.tensor[:, stream])


class _TorchPerToken:
    def __init__(self, cfg: _TorchConfig, param: Param, tensor: torch.Tensor):
        self.param = param
        self.shape = param.shape
        self.flat = tensor.reshape(tensor.shape[0], -1)

    def __getitem__(self, index):
        i = _flat_index(self.param, index)
        return self.flat[:, i : i + 1]


class _TorchGroupScale(_TorchRow):
    def store(self, value) -> None:
        kind: GroupScale = self.param.kind  # type: ignore[assignment]
        m, hidden = self.cfg.m, self.cfg.epilogue.hidden
        g = kind.group_size
        value = torch.as_tensor(value).float().expand(m, hidden)[:, ::g]
        dtype = self.param.dtype
        if dtype == torch.float32:
            bits = value
        elif dtype == torch.float8_e4m3fn:
            bits = value.clamp(-448.0, 448.0).to(dtype).view(torch.uint8)
        else:
            bits = ((value.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        if kind.swizzled:
            offsets = nvfp4_swizzled_offsets(m, hidden // g, value.device)
            self.tensor.view(torch.uint8).view(-1)[offsets.view(-1)] = bits.reshape(-1)
        else:
            self.tensor.view(bits.dtype).view(m, hidden // g).copy_(bits)


def _torch_proxy(cfg: _TorchConfig, param: Param, value):
    kind = param.kind
    if not kind.is_tensor:
        return float(value)
    if isinstance(kind, Row):
        return _TorchRow(cfg, param, value)
    if isinstance(kind, Streams):
        return _TorchStreams(cfg, param, value)
    if isinstance(kind, Weight):
        return _TorchWeight(cfg, param, value)
    if isinstance(kind, PerToken):
        return _TorchPerToken(cfg, param, value)
    if isinstance(kind, DeviceScalar):
        return _TorchDeviceScalar(cfg, param, value)
    return _TorchGroupScale(cfg, param, value)
