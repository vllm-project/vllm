# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the CuTe KDA RecoverSSM kernels (SM10x tensor cores).

* ``CuteRecoverSSMVerify``: the speculative verify of one KDA layer with the short
  convolution fused in (``cutedsl/verify_ws.py`` for the dim-first conv state,
  ``cutedsl/verify.py`` otherwise). It writes the outputs, the rolled conv state
  and the verify records, like ``causal_conv1d_update`` followed by
  ``kda_recoverssm_verify``.
* ``cute_recoverssm_commit``: the state fold of a prepared commit plan for all
  KDA layers in one launch (``cutedsl/commit.py``), replacing the Triton fold of
  ``KDARecoverSSMCommitContext.commit``.

The kernels compile when the KV cache is bound, before CUDA graph capture;
nothing compiles under capture.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
        KDARecoverSSMRecords,
    )

K = 128  # KDA head dim (keys = values)
CONV_WIDTH = 4
MAX_WINDOW = 8


def can_use_cute_recoverssm(
    *,
    head_dim: int,
    window: int,
    conv_width: int,
    has_conv_bias: bool,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    state_dtype: torch.dtype,
    conv_weight_dtype: torch.dtype,
    lower_bound: float | None,
) -> tuple[bool, str | None]:
    """Whether the CuTe verify and commit can serve a KDA layer, and if not,
    why."""
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False, "it needs SM10x tensor cores (tcgen05)"
    if head_dim != K or conv_width != CONV_WIDTH:
        return False, f"it needs head_dim {K} and a width-{CONV_WIDTH} convolution"
    if has_conv_bias:
        return False, "it does not apply a convolution bias"
    if not 1 <= window <= MAX_WINDOW:
        return False, f"it verifies 1 to {MAX_WINDOW} tokens per step"
    if lower_bound is None:
        return False, "it needs a bounded KDA gate (gate_lower_bound)"
    if (activation_dtype, conv_state_dtype, state_dtype) != (torch.bfloat16,) * 3:
        return False, "it needs bfloat16 activations, conv state and SSM state"
    if conv_weight_dtype not in (torch.bfloat16, torch.float32):
        return False, "it needs a bfloat16 or float32 convolution weight"
    return True, None


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
        dynamic_fake(4, f32),  # corrections [rows, H, Q, V]
        dynamic_fake(4, bf16),  # key / gate records [rows, H, Q, 2K]
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
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


def _aligned(*tensors: torch.Tensor) -> bool:
    return all(tensor.data_ptr() % 16 == 0 for tensor in tensors)


def _check_layouts(state: torch.Tensor, *tensors: torch.Tensor) -> None:
    """The KV cache pages, records and parameters the CuTe kernels read: the
    state streams in by TMA (K-element rows of contiguous heads, slots a multiple
    of 16 bytes apart), and every tensor starts on a 16-byte boundary."""
    if state.stride()[1:] != (K * K, K, 1) or state.stride(0) * 2 % 16:
        raise ValueError(
            f"KDA RecoverSSM CuTe kernels: state strides {state.stride()} need "
            "contiguous heads and 16-byte aligned slots"
        )
    if not _aligned(state, *tensors):
        raise ValueError(
            "KDA RecoverSSM CuTe kernels: the KV cache pages, records and "
            "parameters must start on 16-byte boundaries"
        )


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
        # zeroed for the next one.
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._compiled: dict[bool, Any] = {}
        # Page and parameter layouts already checked (fixed per KV cache).
        self._checked: set[tuple] = set()

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
        records: "KDARecoverSSMRecords",
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        out: torch.Tensor,
    ) -> bool:
        """Verify ``state_indices.shape[0]`` requests. ``mixed_qkv`` [T, 3HK] is
        the raw projection (before the convolution), ``raw_gate`` [T, HK],
        ``raw_beta`` [T, H], ``out`` [T, HV]; ``conv_state`` is the stored
        [slots, 3HK, Q + 2] (dim first) or [slots, Q + 2, 3HK] page view.
        Request ``b`` writes its verify records to row ``b`` of ``records``.

        Returns False without launching when an activation does not start on a
        16-byte boundary (e.g. for an odd token offset into projection rows that
        are not 16-byte aligned): the caller then runs the Triton verify."""
        correction, key_gate = records.correction, records.key_gate
        static = (state, A_log, dt_bias, correction, key_gate, conv_state, conv_weight)
        key = tuple((t.data_ptr(), t.stride()) for t in static)
        if key not in self._checked:
            _check_layouts(*static)
            self._checked.add(key)
        if not _aligned(
            mixed_qkv,
            raw_gate,
            raw_beta,
            out,
            query_start_loc,
            state_indices,
            num_accepted_tokens,
        ):
            return False
        warp_specialized = self.conv_dim_first and self._bulk_copy_aligned(
            mixed_qkv, raw_gate, conv_state, conv_weight
        )
        # TVM-FFI launch on the current stream, straight from the torch tensors.
        self._compiled[warp_specialized](
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
            lower_bound,
            NULL_BLOCK_ID,
        )
        return True

    def _bulk_copy_aligned(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
    ) -> bool:
        """The warp-specialized verify bulk-copies q / k / v and gate rows (16-byte
        aligned rows), conv state channel blocks (contiguous per slot) and conv
        taps; otherwise the plain tcgen05 verify runs."""
        conv_len = self.window + CONV_WIDTH - 2
        return (
            mixed_qkv.stride(0) % 8 == 0
            and raw_gate.stride(0) % 8 == 0
            and conv_weight.stride() == (CONV_WIDTH, 1)
            and conv_state.stride()[1:] == (conv_len, 1)
            and conv_state.stride(0) * conv_state.element_size() % 16 == 0
        )


@cache
def _compile_commit(
    heads: int, window: int, arch: str, param_dtype: torch.dtype, align: bool
):
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
        dynamic_fake(1, i64),  # correction row strides [L]
        dynamic_fake(1, i64),  # key / gate row strides [L]
        dynamic_fake(2, param_dtype),  # A_log [L, H]
        dynamic_fake(3, param_dtype),  # dt_bias [L, H, K]
        dynamic_fake(1, i32),  # source slots [B]
        dynamic_fake(1, i32),  # commit lengths [B]
        dynamic_fake(1, i32),  # destination slots [B]
        dynamic_fake(1, i32),  # boundary slots [B]
        dynamic_fake(1, i32),  # boundary lengths [B]
    )
    return cute.compile(
        KdaCommit(heads, window, align),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


def compile_state_commit(
    num_heads: int, window: int, device: torch.device, align: bool
) -> None:
    """Compile the CuTe state commit (before serving: it runs eagerly)."""
    _compile_commit(num_heads, window, _architecture(device), torch.float32, align)


def cute_recoverssm_commit(
    context: "KDARecoverSSMCommitContext",
    state_indices: torch.Tensor,
    batch: int,
    align: bool,
) -> None:
    """Fold the accepted tokens of a prepared commit plan into the state of all
    of ``context``'s layers in one launch. It runs after sampling, outside CUDA
    graphs, on the current stream (TVM-FFI)."""
    state = context.checkpoints[0]
    assert context.lower_bound is not None
    compiled = _compile_commit(
        state.shape[1],
        context.spec_query_len,
        _architecture(state.device),
        torch.float32,
        align,
    )
    compiled(
        context.state_base_addrs,
        context.correction_cache_base_addrs,
        context.kg_cache_base_addrs,
        context.state_block_strides,
        context.correction_cache_block_strides,
        context.kg_cache_block_strides,
        context.A_log,
        context.dt_bias,
        state_indices[:batch],
        context.commit_lens[:batch],
        context.final_state_indices[:batch],
        context.boundary_state_indices[:batch],
        context.boundary_recovery_lens[:batch],
        context.lower_bound,
        NULL_BLOCK_ID,
    )
