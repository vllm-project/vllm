# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the CuTe KDA RecoverSSM kernels (SM10x tensor cores).

* ``CuteRecoverSSMVerify``: the speculative verify of one KDA layer with the short
  convolution fused in (``cutedsl/verify_ws.py`` for the dim-first conv state,
  ``cutedsl/verify.py`` otherwise). It writes the outputs, the rolled conv state
  and the verify records, like ``causal_conv1d_update`` followed by
  ``kda_recoverssm_verify``.
* ``CuteRecoverSSMStateCommit``: the state fold of a prepared commit plan for all
  KDA layers in one launch (``cutedsl/commit.py``), replacing the Triton fold of
  ``KDARecoverSSMCommitContext.commit`` outside align mode.

A compiled CuTe kernel accepts any tensor without checking it against the
signature it was compiled for, so every entry point checks dtypes and layouts
before launching. The kernels compile when the KV cache is bound, before CUDA
graph capture; nothing compiles under capture.
"""

from collections.abc import Sequence
from functools import cache
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

K = 128  # KDA head dim (keys = values)
CONV_WIDTH = 4
MAX_WINDOW = 8
COMMIT_HEADS_PER_CTA = 8


def cute_recoverssm_unsupported_reason(
    *,
    num_heads: int,
    head_dim: int,
    window: int,
    conv_width: int,
    has_conv_bias: bool,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    state_dtype: torch.dtype,
    conv_weight_dtype: torch.dtype,
    lower_bound: float | None,
) -> str | None:
    """Why the CuTe verify cannot serve a KDA layer (None: it can)."""
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return "it needs SM10x tensor cores (tcgen05)"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return "the CuTe DSL (nvidia-cutlass-dsl) is not installed"
    if head_dim != K or conv_width != CONV_WIDTH:
        return f"it needs head_dim {K} and a width-{CONV_WIDTH} convolution"
    if has_conv_bias:
        return "it does not apply a convolution bias"
    if not 1 <= window <= MAX_WINDOW:
        return f"it verifies 1 to {MAX_WINDOW} tokens per step"
    if num_heads < 1:
        return "the layer has no local heads"
    if lower_bound is None:
        return "it needs a bounded KDA gate (gate_lower_bound)"
    if (activation_dtype, conv_state_dtype, state_dtype) != (torch.bfloat16,) * 3:
        return "it needs bfloat16 activations, conv state and SSM state"
    if conv_weight_dtype not in (torch.bfloat16, torch.float32):
        return "it needs a bfloat16 or float32 convolution weight"
    return None


def _architecture(device: torch.device) -> str:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        architecture,
    )

    return architecture(device)


def _ctas(device: torch.device, per_sm: int) -> int:
    return per_sm * torch.cuda.get_device_properties(device).multi_processor_count


@cache
def _compile_verify(
    heads: int,
    window: int,
    arch: str,
    ctas: int,
    warp_specialized: bool,
    conv_dim_first: bool,
    conv_weight_dtype: torch.dtype,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_fake,
    )

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    fakes = (
        dynamic_fake(2, bf16),  # raw q / k / v projection [T, 3HK]
        dynamic_fake(2, bf16),  # raw gate [T, HK]
        dynamic_fake(2, bf16),  # raw beta [T, H]
        dynamic_fake(1, f32),  # A_log [H]
        dynamic_fake(1, f32),  # dt_bias [HK]
        dynamic_fake(4, bf16),  # SSM state [slots, H, V, K]
        dynamic_fake(4, f32),  # corrections [slots, H, Q, V]
        dynamic_fake(4, bf16),  # key / gate records [slots, H, Q, 2K]
        dynamic_fake(2, bf16),  # output [T, HV]
        dynamic_fake(1, i32),  # query start [B + 1]
        dynamic_fake(1, i32),  # state slots [B]
        dynamic_fake(3, bf16),  # conv state
        dynamic_fake(2, conv_weight_dtype),  # conv weight [3HK, 4]
        dynamic_fake(1, i32),  # accepted tokens [B]
        dynamic_fake(1, i32),  # work counter [1]
    )
    if warp_specialized:
        from .cutedsl.verify_ws import KdaTcVerify2

        kernel: Any = KdaTcVerify2(heads, window, ctas, conv_weight_dtype == f32)
    else:
        from .cutedsl.verify import KdaTcVerify

        kernel = KdaTcVerify(
            heads,
            window,
            conv=True,
            ctas=ctas,
            conv_dim_first=conv_dim_first,
            conv_weight_f32=conv_weight_dtype == f32,
        )
    return cute.compile(
        kernel,
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def _check(name: str, tensor: torch.Tensor, dtype: torch.dtype, ndim: int) -> None:
    if tensor.dtype != dtype or tensor.ndim != ndim or tensor.stride(-1) != 1:
        raise ValueError(
            f"KDA RecoverSSM CuTe kernel: {name} must be a {ndim}-D {dtype} tensor "
            f"with a unit last stride; got {tensor.dtype} shape "
            f"{tuple(tensor.shape)} strides {tensor.stride()}"
        )


def _aligned(*tensors: torch.Tensor) -> bool:
    return all(tensor.data_ptr() % 16 == 0 for tensor in tensors)


class CuteRecoverSSMVerify:
    """The CuTe verify for one KDA layer's shapes (compiled kernels are shared)."""

    def __init__(
        self,
        *,
        num_heads: int,
        window: int,
        conv_dim_first: bool,
        conv_weight_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_heads = num_heads
        self.window = window
        self.conv_dim_first = conv_dim_first
        self.conv_weight_dtype = conv_weight_dtype
        self.device = device
        # Each launch takes one ticket per work item and leaves the counter
        # zeroed. One per layer, so verifies on different streams never share it.
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._compiled: dict[bool, Any] = {}

    def compile(self) -> None:
        """Compile the variants this layer launches (outside graph capture)."""
        from .cutedsl import verify, verify_ws

        arch = _architecture(self.device)
        variants = [False] + ([True] if self.conv_dim_first else [])
        for warp_specialized in variants:
            per_sm = verify_ws.CTAS_PER_SM if warp_specialized else verify.CTAS_PER_SM
            self._compiled[warp_specialized] = _compile_verify(
                self.num_heads,
                self.window,
                arch,
                _ctas(self.device, per_sm),
                warp_specialized,
                self.conv_dim_first,
                self.conv_weight_dtype,
            )

    def __call__(
        self,
        *,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        state: torch.Tensor,
        correction: torch.Tensor,
        key_gate: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Verify ``state_indices.shape[0]`` requests. ``mixed_qkv`` [T, 3HK] is
        the raw projection (before the convolution), ``raw_gate`` [T, HK],
        ``raw_beta`` [T, H], ``out`` [T, HV]; ``conv_state`` is the stored
        [slots, 3HK, Q + 2] (dim first) or [slots, Q + 2, 3HK] page view."""
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            dynamic_view,
            stream,
        )

        H, Q = self.num_heads, self.window
        tokens = mixed_qkv.shape[0]
        batch = state_indices.shape[0]
        slots = state.shape[0]
        for name, tensor, dtype, ndim in (
            ("mixed_qkv", mixed_qkv, torch.bfloat16, 2),
            ("raw_gate", raw_gate, torch.bfloat16, 2),
            ("raw_beta", raw_beta, torch.bfloat16, 2),
            ("A_log", A_log, torch.float32, 1),
            ("dt_bias", dt_bias, torch.float32, 1),
            ("state", state, torch.bfloat16, 4),
            ("correction", correction, torch.float32, 4),
            ("key_gate", key_gate, torch.bfloat16, 4),
            ("out", out, torch.bfloat16, 2),
            ("query_start_loc", query_start_loc, torch.int32, 1),
            ("state_indices", state_indices, torch.int32, 1),
            ("num_accepted_tokens", num_accepted_tokens, torch.int32, 1),
            ("conv_state", conv_state, torch.bfloat16, 3),
            ("conv_weight", conv_weight, self.conv_weight_dtype, 2),
        ):
            _check(name, tensor, dtype, ndim)
        conv_len = Q + CONV_WIDTH - 2
        conv_shape = (slots, 3 * H * K, conv_len)
        if not self.conv_dim_first:
            conv_shape = (slots, conv_len, 3 * H * K)
        expected = (
            ("mixed_qkv", mixed_qkv, (tokens, 3 * H * K)),
            ("raw_gate", raw_gate, (tokens, H * K)),
            ("raw_beta", raw_beta, (tokens, H)),
            ("A_log", A_log, (H,)),
            ("dt_bias", dt_bias, (H * K,)),
            ("state", state, (slots, H, K, K)),
            ("correction", correction, (slots, H, Q, K)),
            ("key_gate", key_gate, (slots, H, Q, 2 * K)),
            ("out", out, (tokens, H * K)),
            ("query_start_loc", query_start_loc, (batch + 1,)),
            ("conv_state", conv_state, conv_shape),
            ("conv_weight", conv_weight, (3 * H * K, CONV_WIDTH)),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"KDA RecoverSSM CuTe verify: {name} has shape "
                    f"{tuple(tensor.shape)}, expected {shape}"
                )
        if num_accepted_tokens.shape[0] < batch:
            raise ValueError("KDA RecoverSSM CuTe verify: too few accepted counts")
        if tokens > batch * Q:
            raise ValueError("KDA RecoverSSM CuTe verify: more tokens than slots")
        # The state streams in by TMA: K-element rows, 16-byte aligned slots.
        if (
            state.stride()[1:] != (K * K, K, 1)
            or state.stride(0) * 2 % 16
            or not _aligned(state)
        ):
            raise ValueError(
                f"KDA RecoverSSM CuTe verify: state strides {state.stride()} need "
                "contiguous heads and 16-byte aligned slots"
            )
        if batch == 0 or tokens == 0:
            return
        warp_specialized = self.conv_dim_first and self._bulk_copy_aligned(
            mixed_qkv, raw_gate, dt_bias, conv_state, conv_weight
        )
        compiled = self._compiled.get(warp_specialized)
        if compiled is None:
            raise RuntimeError(
                "KDA RecoverSSM CuTe verify launched before compile() (it must "
                "not compile under CUDA graph capture)"
            )
        compiled(
            *map(
                dynamic_view,
                (
                    mixed_qkv,
                    raw_gate,
                    raw_beta,
                    A_log,
                    dt_bias,
                    state,
                    correction,
                    key_gate,
                    out,
                    query_start_loc,
                    state_indices,
                    conv_state,
                    conv_weight,
                    num_accepted_tokens,
                    self.counter,
                ),
            ),
            cutlass.Float32(lower_bound),
            cutlass.Int32(NULL_BLOCK_ID),
            stream(self.device),
        )

    def _bulk_copy_aligned(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
    ) -> bool:
        """The warp-specialized verify bulk-copies 16-byte aligned q / k / v and
        gate rows, conv state channel blocks (contiguous per slot), conv taps and
        dt_bias; otherwise the plain tcgen05 verify runs."""
        conv_len = self.window + CONV_WIDTH - 2
        return (
            _aligned(mixed_qkv, raw_gate, dt_bias, conv_state, conv_weight)
            and mixed_qkv.stride(0) % 8 == 0
            and raw_gate.stride(0) % 8 == 0
            and conv_weight.stride() == (CONV_WIDTH, 1)
            and conv_state.stride()[1:] == (conv_len, 1)
            and conv_state.stride(0) * conv_state.element_size() % 16 == 0
        )


@cache
def _compile_commit(heads: int, window: int, arch: str, param_dtype: torch.dtype):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_fake,
    )

    from .cutedsl.commit import KdaCommit

    i32, i64 = torch.int32, torch.int64
    fakes = (
        dynamic_fake(1, i64),  # state base addresses [L]
        dynamic_fake(1, i64),  # correction base addresses [L]
        dynamic_fake(1, i64),  # key / gate base addresses [L]
        dynamic_fake(1, i64),  # state slot strides [L]
        dynamic_fake(1, i64),  # correction slot strides [L]
        dynamic_fake(1, i64),  # key / gate slot strides [L]
        dynamic_fake(2, param_dtype),  # A_log [L, H]
        dynamic_fake(3, param_dtype),  # dt_bias [L, H, K]
        dynamic_fake(1, i32),  # source slots [B]
        dynamic_fake(1, i32),  # commit lengths [B]
        dynamic_fake(1, i32),  # destination slots [B]
    )
    return cute.compile(
        KdaCommit(heads, window),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def compile_state_commit(num_heads: int, window: int, device: torch.device) -> None:
    """Compile the CuTe state commit (outside graph capture)."""
    if num_heads % COMMIT_HEADS_PER_CTA == 0:
        _compile_commit(num_heads, window, _architecture(device), torch.float32)


class CuteRecoverSSMStateCommit:
    """The CuTe state fold for a ``KDARecoverSSMCommitContext``'s layers.

    Every argument but the source slots is fixed per context and batch size, so
    their CuTe views are built once: a commit runs after sampling, outside CUDA
    graphs, and its host time is on every decode step."""

    def __init__(self, context: Any) -> None:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            dynamic_view,
        )

        state = context.checkpoints[0]
        _, heads, _, _ = state.shape
        self.compiled = _compile_commit(
            heads, context.spec_query_len, _architecture(state.device), torch.float32
        )
        self.context = context
        self.device = state.device
        self.constants = tuple(
            dynamic_view(tensor)
            for tensor in (
                context.state_base_addrs,
                context.correction_cache_base_addrs,
                context.kg_cache_base_addrs,
                context.state_block_strides,
                context.correction_cache_block_strides,
                context.kg_cache_block_strides,
                context.A_log,
                context.dt_bias,
            )
        )
        self._plans: dict[int, tuple[Any, Any]] = {}

    @staticmethod
    def unsupported_reason(
        states: Sequence[torch.Tensor],
        corrections: Sequence[torch.Tensor],
        key_gates: Sequence[torch.Tensor],
        A_log: torch.Tensor,
        window: int,
        lower_bound: float | None,
    ) -> str | None:
        state = states[0]
        _, heads, value_dim, key_dim = state.shape
        if (value_dim, key_dim) != (K, K) or not 1 <= window <= MAX_WINDOW:
            return f"it needs {K} x {K} heads and 1 to {MAX_WINDOW} tokens"
        if heads % COMMIT_HEADS_PER_CTA:
            return f"it folds {COMMIT_HEADS_PER_CTA} heads per CTA"
        if lower_bound is None or A_log.dtype != torch.float32:
            return "it needs a bounded gate and float32 gate parameters"
        for tensors, dtype, inner in (
            (states, torch.bfloat16, (K * K, K, 1)),
            (corrections, torch.float32, (window * K, K, 1)),
            (key_gates, torch.bfloat16, (window * 2 * K, 2 * K, 1)),
        ):
            for tensor in tensors:
                # Pages are bulk-copied 16 bytes at a time.
                if (
                    tensor.dtype != dtype
                    or tensor.stride()[1:] != inner
                    or tensor.stride(0) * tensor.element_size() % 16
                    or not _aligned(tensor)
                ):
                    return (
                        f"a {tensor.dtype} page view with strides "
                        f"{tensor.stride()} is not a 16-byte aligned {dtype} one "
                        f"with inner strides {inner}"
                    )
        return None

    def __call__(self, state_indices: torch.Tensor, batch: int) -> None:
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            dynamic_view,
            stream,
        )

        context = self.context
        plan = self._plans.get(batch)
        if plan is None:
            plan = (
                dynamic_view(context.commit_lens[:batch]),
                dynamic_view(context.final_state_indices[:batch]),
            )
            self._plans[batch] = plan
        self.compiled(
            *self.constants,
            dynamic_view(state_indices[:batch]),
            *plan,
            cutlass.Float32(context.lower_bound),
            cutlass.Int32(NULL_BLOCK_ID),
            stream(self.device),
        )
