# SPDX-License-Identifier: Apache-2.0
"""Asymmetric K16/V8 KV codec.

K is stored at FP16 or BF16 (the model's native dtype), V is
quantized to FP8 e4m3fn with per-tensor / per-layer-head /
per-page-head scales.  The codec is pure PyTorch; GPU support is
inherited automatically from the device of the input tensors.

Quant convention:
  scale = max(|V|) / finfo(fp8).max
  q     = clamp(V / scale, -finfo.max, +finfo.max).to(fp8)
  V_hat = q.to(fp32) * scale
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.v1.kv_codec.encoded_kv import (
    CodecHashes,
    EncodedKV,
    ScaleScope,
    deserialize_header,
    serialize_header,
)
from lmcache.v1.kv_codec.errors import (
    CodecMismatchError,
    UnsupportedConfigError,
)

# Sentinel scale used when an entire tensor / head / page is all
# zero.  Picking 1.0 keeps `dequant(quant(0)) == 0` exact and avoids
# a 0/0 NaN.
_ZERO_SCALE_SENTINEL = 1.0


def _tensor_to_bytes_fast(t: torch.Tensor) -> bytes:
    """Convert a contiguous CPU tensor to bytes via a single C-level memcpy.

    Reinterpreting the tensor as uint8 and calling numpy's ``.tobytes()``
    emits exactly the tensor's bytes -- including for tensors whose dtype
    numpy doesn't natively know about (FP8, BF16), and crucially for
    slices, where ``t.untyped_storage()`` would return the underlying
    storage of the parent tensor (including sibling-slice bytes) rather
    than just this slice's bytes.

    Much faster than ``bytes(tensor.untyped_storage())``, which routes
    through Python's bytes() constructor and iterates byte-by-byte at the
    Python level.
    """
    return t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def _fp8_max(dtype: torch.dtype) -> float:
    """FP8 representable absolute max via torch.finfo."""
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise UnsupportedConfigError(
            f"compute_v_scales: dtype {dtype} is not an FP8 dtype"
        )
    return float(torch.finfo(dtype).max)


def compute_v_scales(
    v: torch.Tensor,
    scope: ScaleScope,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    head_axis: int = -2,
    page_axis: Optional[int] = None,
) -> torch.Tensor:
    """Compute FP32 V scales according to the requested scope.

    Args:
        v: the V tensor.  Layout-agnostic; `head_axis` and
            `page_axis` index the dimensions for the chosen scope.
        scope: PER_TENSOR / PER_LAYER_HEAD / PER_PAGE_HEAD.
        fp8_dtype: float8_e4m3fn or float8_e5m2.
        head_axis: which dim is "kv head" for per-head scales.
            Default -2 covers (..., heads, dim) layouts.
        page_axis: which dim is "page" for per-page scales.  Required
            when scope is PER_PAGE_HEAD.

    Returns:
        FP32 scale tensor whose shape depends on scope:
            PER_TENSOR     -> ()
            PER_LAYER_HEAD -> (n_heads,)
            PER_PAGE_HEAD  -> (n_pages, n_heads)
    """
    qmax = _fp8_max(fp8_dtype)
    v32 = v.detach().to(torch.float32)

    if scope == ScaleScope.PER_TENSOR:
        amax = v32.abs().amax()
        return torch.where(
            amax > 0,
            amax / qmax,
            torch.tensor(_ZERO_SCALE_SENTINEL, dtype=torch.float32, device=v.device),
        ).reshape(())

    if scope == ScaleScope.PER_LAYER_HEAD:
        # reduce over all dims except head_axis
        ndim = v32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        reduce_dims = tuple(d for d in range(ndim) if d != head)
        amax = v32.abs().amax(dim=reduce_dims)  # (n_heads,)
        scale = torch.where(
            amax > 0,
            amax / qmax,
            torch.full_like(amax, _ZERO_SCALE_SENTINEL),
        )
        return scale

    if scope == ScaleScope.PER_PAGE_HEAD:
        if page_axis is None:
            raise UnsupportedConfigError(
                "compute_v_scales(PER_PAGE_HEAD) requires page_axis"
            )
        ndim = v32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        page = page_axis if page_axis >= 0 else ndim + page_axis
        keep = (page, head)
        reduce_dims = tuple(d for d in range(ndim) if d not in keep)
        amax = v32.abs().amax(dim=reduce_dims)
        # amax is shaped per the surviving dims in their original
        # order; permute to (n_pages, n_heads) for the on-disk shape.
        if page > head:
            # head dim came first after reduction; transpose
            amax = amax.transpose(-1, -2)
        scale = torch.where(
            amax > 0,
            amax / qmax,
            torch.full_like(amax, _ZERO_SCALE_SENTINEL),
        )
        return scale

    if scope == ScaleScope.EXTERNAL:
        raise UnsupportedConfigError(
            "compute_v_scales does not produce EXTERNAL scales; the "
            "caller is expected to attach them."
        )

    raise UnsupportedConfigError(f"unknown scale scope: {scope}")


def quantize_v_fp8(
    v: torch.Tensor,
    scales: torch.Tensor,
    scope: ScaleScope,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    head_axis: int = -2,
    page_axis: Optional[int] = None,
) -> torch.Tensor:
    """Quantize V to FP8 using the given scales.

    Saturates at +/- finfo(fp8).max (via clamp before cast).
    Returns a tensor with dtype=fp8_dtype, same shape as v.
    """
    qmax = _fp8_max(fp8_dtype)
    v32 = v.detach().to(torch.float32)

    if scope == ScaleScope.PER_TENSOR:
        s = scales.to(torch.float32).reshape(())
        scaled = v32 / s
    elif scope == ScaleScope.PER_LAYER_HEAD:
        ndim = v32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        # broadcast (n_heads,) along all other dims
        view_shape = [1] * ndim
        view_shape[head] = scales.shape[0]
        s = scales.to(torch.float32).view(*view_shape)
        scaled = v32 / s
    elif scope == ScaleScope.PER_PAGE_HEAD:
        if page_axis is None:
            raise UnsupportedConfigError(
                "quantize_v_fp8(PER_PAGE_HEAD) requires page_axis"
            )
        ndim = v32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        page = page_axis if page_axis >= 0 else ndim + page_axis
        view_shape = [1] * ndim
        view_shape[page] = scales.shape[0]
        view_shape[head] = scales.shape[1]
        s = scales.to(torch.float32).view(*view_shape)
        scaled = v32 / s
    else:
        raise UnsupportedConfigError(f"unsupported scope: {scope}")

    # Saturate, then cast.
    saturated = torch.clamp(scaled, min=-qmax, max=qmax)
    return saturated.to(fp8_dtype)


def dequantize_v_fp8(
    q: torch.Tensor,
    scales: torch.Tensor,
    scope: ScaleScope,
    *,
    out_dtype: torch.dtype = torch.float16,
    head_axis: int = -2,
    page_axis: Optional[int] = None,
) -> torch.Tensor:
    """Inverse of quantize_v_fp8.  Returns out_dtype."""
    q32 = q.detach().to(torch.float32)
    if scope == ScaleScope.PER_TENSOR:
        s = scales.to(torch.float32).reshape(())
        out = q32 * s
    elif scope == ScaleScope.PER_LAYER_HEAD:
        ndim = q32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        view_shape = [1] * ndim
        view_shape[head] = scales.shape[0]
        s = scales.to(torch.float32).view(*view_shape)
        out = q32 * s
    elif scope == ScaleScope.PER_PAGE_HEAD:
        if page_axis is None:
            raise UnsupportedConfigError(
                "dequantize_v_fp8(PER_PAGE_HEAD) requires page_axis"
            )
        ndim = q32.ndim
        head = head_axis if head_axis >= 0 else ndim + head_axis
        page = page_axis if page_axis >= 0 else ndim + page_axis
        view_shape = [1] * ndim
        view_shape[page] = scales.shape[0]
        view_shape[head] = scales.shape[1]
        s = scales.to(torch.float32).view(*view_shape)
        out = q32 * s
    else:
        raise UnsupportedConfigError(f"unsupported scope: {scope}")
    return out.to(out_dtype)


class AsymK16V8Codec:
    """Concrete codec for FP16/BF16-K + FP8-V asymmetric KV cache.

    The codec is stateless aside from configuration (fp8 variant,
    expected scale scope).  Instances are cheap; one per
    (kv_storage_codec, kv_scale_scope) pair is fine.
    """

    def __init__(
        self,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_dtype: torch.dtype = torch.float32,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
    ):
        # Default is PER_TENSOR because PER_PAGE_HEAD also needs a
        # page_axis argument every time `encode` is called.  Callers
        # operating on paged caches construct the codec with
        # `scale_scope=PER_PAGE_HEAD` and pass `page_axis` to encode.
        if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise UnsupportedConfigError(
                f"AsymK16V8Codec: fp8_dtype must be float8_e4m3fn "
                f"(or e5m2 reserved); got {fp8_dtype}"
            )
        if fp8_dtype == torch.float8_e5m2:
            raise UnsupportedConfigError(
                "float8_e5m2 is reserved-but-not-implemented in v1; use float8_e4m3fn"
            )
        self.fp8_dtype = fp8_dtype
        self.scale_dtype = scale_dtype
        self.scale_scope = scale_scope

    def encode(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        hashes: Optional[CodecHashes] = None,
        layer_id: int = -1,
        chunk_id: int = -1,
        chunk_size: int = -1,
        page_size: int = -1,
        kv_head_count: int = -1,
        head_dim: int = -1,
        head_axis: int = -2,
        page_axis: Optional[int] = None,
        precomputed_v_scales: Optional[torch.Tensor] = None,
        precomputed_v_quant: Optional[torch.Tensor] = None,
    ) -> EncodedKV:
        """Encode an asymmetric (K, V) pair into one EncodedKV blob.

        If `precomputed_v_quant` is provided it is used as-is: the
        caller has already produced V in FP8 and we copy bytes
        without re-quanting.  Otherwise V is quantized here from
        FP16/BF16.

        `precomputed_v_scales` must be supplied alongside
        `precomputed_v_quant` since we cannot recompute scales from
        already-quantized V.

        K is always copied at its native dtype; we never quantize K
        in this codec (that's the whole asymmetric point).
        """
        if k.shape != v.shape:
            raise UnsupportedConfigError(
                f"encode: K and V shapes must match; got K={tuple(k.shape)} "
                f"vs V={tuple(v.shape)}"
            )

        if precomputed_v_quant is not None:
            if precomputed_v_scales is None:
                raise UnsupportedConfigError(
                    "encode: precomputed_v_quant requires precomputed_v_scales"
                )
            if precomputed_v_quant.dtype != self.fp8_dtype:
                raise UnsupportedConfigError(
                    f"encode: precomputed_v_quant dtype "
                    f"{precomputed_v_quant.dtype} != codec "
                    f"{self.fp8_dtype}"
                )
            v_quant = precomputed_v_quant
            v_scales = precomputed_v_scales.to(self.scale_dtype)
        else:
            v_scales = compute_v_scales(
                v,
                self.scale_scope,
                fp8_dtype=self.fp8_dtype,
                head_axis=head_axis,
                page_axis=page_axis,
            ).to(self.scale_dtype)
            v_quant = quantize_v_fp8(
                v,
                v_scales,
                self.scale_scope,
                fp8_dtype=self.fp8_dtype,
                head_axis=head_axis,
                page_axis=page_axis,
            )

        # Ensure all payloads are contiguous and CPU for the byte
        # write.  The caller may be on GPU; serialize_header runs on
        # CPU bytes either way.
        #
        # `_tensor_to_bytes_fast` reinterprets via `view(uint8).numpy()`
        # which respects the contiguous view's stride and emits exactly
        # the slice's bytes, so `.contiguous()` after `.to("cpu")` is
        # sufficient -- no additional `.clone()` is needed.
        k_cpu = k.detach().to("cpu").contiguous()
        v_cpu = v_quant.detach().to("cpu").contiguous()
        s_cpu = v_scales.detach().to("cpu").contiguous()
        k_bytes = _tensor_to_bytes_fast(k_cpu)
        v_bytes = _tensor_to_bytes_fast(v_cpu)
        s_bytes = _tensor_to_bytes_fast(s_cpu)

        enc = EncodedKV(
            k_dtype=k.dtype,
            v_dtype=self.fp8_dtype,
            scale_dtype=self.scale_dtype,
            scale_scope=self.scale_scope,
            hashes=hashes if hashes is not None else CodecHashes(),
            layer_id=layer_id,
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            page_size=page_size,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            scale_shape=tuple(v_scales.shape),
            k_payload_len=len(k_bytes),
            v_payload_len=len(v_bytes),
            scale_payload_len=len(s_bytes),
            payload=k_bytes + v_bytes + s_bytes,
        )
        enc.header_bytes = serialize_header(enc)
        return enc

    def to_bytes(self, enc: EncodedKV) -> bytes:
        """Concatenate header + payload for on-disk write."""
        if enc.header_bytes is None:
            enc.header_bytes = serialize_header(enc)
        # `enc.payload` may be a memoryview on the read path; coerce
        # to bytes so the returned object satisfies the bytes contract
        # callers rely on.
        if isinstance(enc.payload, memoryview):
            return enc.header_bytes + bytes(enc.payload)
        return enc.header_bytes + enc.payload

    def from_bytes(
        self,
        buf: bytes,
        *,
        expected_hashes: Optional[CodecHashes] = None,
    ) -> EncodedKV:
        """Parse a byte buffer into an EncodedKV.

        Cross-config gates: if `expected_hashes` is provided, any
        non-empty mismatch raises `CodecMismatchError`.  Empty
        strings are wildcards on either side.
        """
        enc = deserialize_header(buf)
        if expected_hashes is not None:
            self._check_hash_match(enc.hashes, expected_hashes)
        return enc

    @staticmethod
    def _check_hash_match(got: CodecHashes, expected: CodecHashes) -> None:
        for key in CodecHashes._CHECK_ORDER:
            ev = getattr(expected, key)
            gv = getattr(got, key)
            if ev and gv and ev != gv:
                raise CodecMismatchError(
                    f"codec hash mismatch on {key!r}: encoded={gv!r}, expected={ev!r}"
                )

    def decode(
        self,
        enc: EncodedKV,
        *,
        out_k_dtype: Optional[torch.dtype] = None,
        out_v_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode an EncodedKV into (K, V_quant_or_dequant, scales).

        If `out_v_dtype` is None, V is returned at its native FP8
        dtype (storage_only=False, native asym pathway).  If
        `out_v_dtype` is FP16/BF16, V is dequantized into that
        dtype.

        K is returned at `out_k_dtype` if specified, else at the
        encoded dtype.
        """
        if out_k_dtype is None:
            out_k_dtype = enc.k_dtype
        device = device or torch.device("cpu")

        k_off = 0
        v_off = enc.k_payload_len
        s_off = v_off + enc.v_payload_len
        k_bytes = enc.payload[k_off:v_off]
        v_bytes = enc.payload[v_off:s_off]
        s_bytes = enc.payload[s_off:]

        # K: read-only view over the source bytes, then own the
        # destination via .clone() (single memcpy total).
        k_tensor = torch.frombuffer(k_bytes, dtype=enc.k_dtype).clone()
        if out_k_dtype != enc.k_dtype:
            k_tensor = k_tensor.to(out_k_dtype)
        k_tensor = k_tensor.to(device)

        # Scales
        scale_tensor = (
            torch.frombuffer(s_bytes, dtype=enc.scale_dtype)
            .clone()
            .reshape(enc.scale_shape)
        )
        scale_tensor = scale_tensor.to(device)

        # V: read as fp8, optionally dequantize
        v_q = torch.frombuffer(v_bytes, dtype=enc.v_dtype).clone().to(device)

        if out_v_dtype is None or out_v_dtype == enc.v_dtype:
            return k_tensor, v_q, scale_tensor

        # Dequantize.  Caller specifies a higher-precision dtype.
        if not out_v_dtype.is_floating_point:
            raise UnsupportedConfigError(
                f"decode: out_v_dtype {out_v_dtype} is not floating-point"
            )
        v_dq = dequantize_v_fp8(
            v_q,
            scale_tensor,
            enc.scale_scope,
            out_dtype=out_v_dtype,
        )
        return k_tensor, v_dq, scale_tensor
