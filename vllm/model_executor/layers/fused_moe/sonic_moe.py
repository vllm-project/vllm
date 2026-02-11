# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sonic MoE integration for Hopper GPUs.

Sonic MoE uses swiglu format with rows interleaved as:
  [gate_row0, up_row0, gate_row1, up_row1, ...]
and computes: silu(gate) * up.

vLLM's `silu_and_mul` computes:
  silu(gate) * up
with the convention [gate, up] along the last dimension.

Weight permutation is required during loading to convert between formats.
See: https://github.com/Dao-AILab/sonic-moe/issues/12
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

logger = init_logger(__name__)

_sonicmoe_available: bool | None = None


def _check_sonicmoe_available() -> bool:
    global _sonicmoe_available
    if _sonicmoe_available is not None:
        return _sonicmoe_available

    try:
        # NOTE: Importing the functional API will import the SonicMoE package and
        # its runtime deps (e.g. quack, nvidia-cutlass-dsl).
        from sonicmoe.functional.forward import (  # noqa: F401
            _down_projection_forward,
            _router_forward,
            _up_projection_forward,
        )

        _sonicmoe_available = True
        logger.info("Sonic MoE is available")
    except Exception:
        _sonicmoe_available = False
        # The import may fail with ImportError/ModuleNotFoundError (missing deps)
        # or SyntaxError (unsupported Python version).
        logger.debug(
            "Sonic MoE not available: failed to import sonicmoe functional API",
            exc_info=True,
        )

    return _sonicmoe_available


def _is_hopper_gpu() -> bool:
    if not current_platform.is_cuda():
        return False
    # Hopper is SM90 (compute capability 9.0)
    return current_platform.is_device_capability(90)


def is_sonic_moe_supported() -> bool:
    if not _is_hopper_gpu():
        logger.debug("Sonic MoE requires Hopper GPU (H100/H200)")
        return False
    # Avoid importing SonicMoE (and its heavy deps like quack/cutlass) on
    # non-Hopper systems.
    return _check_sonicmoe_available()


def is_valid_sonic_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> bool:
    if not is_sonic_moe_supported():
        return False

    # SonicMoE's cutedsl kernels have additional shape constraints (asserted
    # inside the upstream implementation). Validate them here so callers can
    # reliably fall back without hard failures.
    #
    # In SonicMoE's pinned commit, `H` is the hidden dim (x.shape[1]) and must
    # be >= 512 and divisible by 64. The intermediate dim after GLU (`I`) must
    # also be divisible by 64.
    H = hidden_states.size(1)
    if H < 512 or H % 64 != 0:
        logger.debug("Sonic MoE: hidden_dim H=%d must be >=512 and divisible by 64", H)
        return False

    if not hidden_states.is_contiguous():
        logger.debug("Sonic MoE: hidden_states not contiguous")
        return False

    if not w1.is_contiguous() or not w2.is_contiguous():
        logger.debug("Sonic MoE: weights not contiguous")
        return False

    if w1.dim() != 3 or w2.dim() != 3:
        logger.debug("Sonic MoE: expected 3D weights")
        return False

    if w1.size(0) != num_experts or w2.size(0) != num_experts:
        logger.debug("Sonic MoE: num_experts mismatch")
        return False

    if w1.size(2) != hidden_states.size(1):
        logger.debug("Sonic MoE: w1 K dimension mismatch")
        return False

    two_n = w1.size(1)
    if two_n % 2 != 0:
        logger.debug("Sonic MoE: w1 second dim must be even (2N)")
        return False
    intermediate_dim = two_n // 2
    if intermediate_dim % 64 != 0:
        logger.debug(
            "Sonic MoE: intermediate dim I=%d must be divisible by 64",
            intermediate_dim,
        )
        return False

    if w2.size(1) != hidden_states.size(1) or w2.size(2) != two_n // 2:
        logger.debug("Sonic MoE: w2 shape mismatch")
        return False

    if top_k > 16:
        logger.debug("Sonic MoE: top_k > 16 not optimized")
        return False

    supported_dtypes = {torch.float16, torch.bfloat16}
    if hidden_states.dtype not in supported_dtypes:
        logger.debug("Sonic MoE: unsupported dtype %s", hidden_states.dtype)
        return False

    return True


def permute_weights_for_sonic(w: torch.Tensor) -> torch.Tensor:
    """
    Permute weights from vLLM's silu_and_mul format to Sonic's swiglu format.

    vLLM format: [gate, up] -> silu(gate) * up
    Sonic format: [interleaved] -> silu(gate) * up (even=gate, odd=up)

    Conversion: interleave [gate, up] so that even indices are Sonic `gate` and
    odd indices are Sonic `up`.

    Reference: https://github.com/Dao-AILab/sonic-moe/issues/12
    """
    if not w.is_contiguous():
        w = w.contiguous()
    E, two_N, K = w.shape
    N = two_N // 2
    # vLLM provides [gate, up]; Sonic expects [gate0, up0, gate1, up1, ...].
    w_reshaped = w.view(E, 2, N, K)
    w_interleaved = w_reshaped.permute(0, 2, 1, 3)
    return w_interleaved.reshape(E, two_N, K).contiguous()


class SonicMoeExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Sonic MoE experts implementation for Hopper GPUs.

    Uses Sonic MoE's optimized kernels for up/down projections.
    Requires weight permutation for swiglu compatibility.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig = FUSED_MOE_UNQUANTIZED_CONFIG,
        weights_prepermuted: bool = False,
    ):
        super().__init__(moe_config, quant_config)
        self.out_dtype = moe_config.in_dtype
        self.weights_prepermuted = weights_prepermuted
        self._w1_sonic: torch.Tensor | None = None
        self._w2_sonic: torch.Tensor | None = None
        self._w1_id: int = -1
        self._w2_id: int = -1

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_sonic_moe_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ("silu", "silu_and_mul")

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.is_sequence_parallel
        )

    def supports_expert_map(self) -> bool:
        return False  # TODO: Verify Sonic MoE expert mapping support

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, max(N, K))
        # NOTE: In the non-chunked case, the modular-kernel runtime may reuse the
        # same underlying storage for `workspace13` and the final `output`.
        # Keep `y2` (down-projection output) out of `workspace13` to avoid any
        # read/write aliasing with `output` during router reduction.
        #
        # Also keep `y1` and `y2` backed by compact contiguous storage. Slicing a
        # wider 2D workspace (pitched layout) yields stride(0) != num_cols, and
        # SonicMoE's CuTe wrappers reject that layout.
        workspace2 = (M * topk * (activation_out_dim + K),)
        output = (M, K)
        return (workspace1, workspace2, output)

    def _ensure_weights_ready(
        self, w1: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w1_id != id(w1) or self._w2_id != id(w2):
            w1_interleaved = (
                w1 if self.weights_prepermuted else permute_weights_for_sonic(w1)
            )
            # SonicMoE expects:
            # w1: (I, H, E) with stride(H) == 1 and stride_order (2, 0, 1)
            # w2: (H, I, E) with stride(I) == 1 and stride_order (2, 0, 1)
            #
            # vLLM provides:
            # w1: (E, 2I, H) in silu_and_mul format (after swiglu interleave)
            # w2: (E, H, I)
            #
            # A pure permute produces the stride pattern SonicMoE validates via
            # mark_layout_dynamic(leading_dim=1) + mark_compact_shape_dynamic(...).
            self._w1_sonic = w1_interleaved.permute(1, 2, 0)
            self._w2_sonic = w2.contiguous().permute(1, 2, 0)
            self._w1_id = id(w1)
            self._w2_id = id(w2)
        assert self._w1_sonic is not None and self._w2_sonic is not None
        return self._w1_sonic, self._w2_sonic

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply Sonic MoE computation.

        Orchestrates:
        1. Weight conversion (cached)
        2. Up projection with GLU activation
        3. Down projection
        4. Router-weighted combination
        """
        if expert_map is not None:
            raise ValueError("Sonic MoE does not support expert_map/EP.")
        if activation not in ("silu", "silu_and_mul"):
            raise ValueError(
                f"Sonic MoE only supports silu/silu_and_mul activation, "
                f"got {activation}"
            )

        w1_sonic, w2_sonic = self._ensure_weights_ready(w1, w2)

        try:
            from sonicmoe.functional.forward import (
                _down_projection_forward,
                _router_forward,
                _up_projection_forward,
            )
        except Exception as e:
            raise RuntimeError(
                "Sonic MoE functional API not available.\n"
                "This requires installing SonicMoE from source (GitHub) plus its "
                "dependencies (quack/cutlass), and running on a Hopper GPU "
                "(H100/H200) with a compatible CUDA toolchain.\n"
                "Install (example): clone SonicMoE, pip install -r requirements.txt, "
                "then pip install -e .\n"
                f"Original import error: {type(e).__name__}: {e}"
            ) from e

        M, K = hidden_states.shape
        two_n, k_from_w1, num_experts = w1_sonic.shape
        topk = topk_ids.shape[1]
        if k_from_w1 != K:
            raise ValueError(f"Sonic MoE expects w1 last dim {K}, got {k_from_w1}")
        if two_n % 2 != 0:
            raise ValueError(f"Sonic MoE expects w1 second dim to be even, got {two_n}")
        n = two_n // 2
        if (
            w2_sonic.size(0) != K
            or w2_sonic.size(1) != n
            or w2_sonic.size(2) != num_experts
        ):
            raise ValueError(
                "Sonic MoE expects w2 shape (K, N, E) with "
                f"K={K}, N={n}, E={num_experts}, got {tuple(w2_sonic.shape)}"
            )

        # TODO(https://github.com/vllm-project/vllm/issues/31578): use router logits
        selected_experts = topk_ids.flatten()
        # SonicMoE expects `x_gather_idx` in the order produced by sorting the
        # flattened expert ids (equivalent to `torch.argsort(topk_ids.view(-1))`).
        s_scatter_idx = torch.argsort(selected_experts).to(torch.int32)

        expert_frequency = selected_experts.bincount(minlength=num_experts).to(
            torch.int32
        )
        # SonicMoE expects an exclusive prefix sum with a leading 0. The i'th
        # expert reads from [offset[i], offset[i + 1]).
        expert_offsets = torch.empty(
            (num_experts + 1,), device=expert_frequency.device, dtype=torch.int32
        )
        expert_offsets[0] = 0
        expert_offsets[1:] = expert_frequency.cumsum(-1)
        expert_schedule_order = None

        x_gather_idx = s_scatter_idx // topk
        s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
        s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
            s_scatter_idx.numel(),
            device=s_scatter_idx.device,
            dtype=s_scatter_idx.dtype,
        )

        z = workspace13[: M * topk, :two_n].view(M * topk, two_n)
        # workspace2 is a flat buffer. Carve compact matrices for y1/y2.
        y1_numel = M * topk * n
        y2_numel = M * topk * K
        y1 = workspace2[:y1_numel].view(M * topk, n)
        y2 = workspace2[y1_numel : y1_numel + y2_numel].view(M * topk, K)

        # SonicMoE custom op expects a string for activation_type.
        act_type = "swiglu"
        stream_id = int(torch.cuda.current_stream().cuda_stream)

        try:
            _up_projection_forward(
                x=hidden_states,
                w1=w1_sonic,
                z=z,
                y1=y1,
                b1=None,
                expert_frequency_offset=expert_offsets,
                expert_schedule_order=expert_schedule_order,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
                activation_type=act_type,
                is_glu_activation=True,
                is_inference_mode_enabled=False,
            )
        except Exception as e:
            # nvidia-cutlass-dsl has known failure modes where a raised exception
            # causes pytest's traceback formatting to segfault while repr()-ing
            # internal CuTe objects. Re-raise a plain RuntimeError to keep CI
            # failures actionable.
            raise RuntimeError(
                "SonicMoE up-projection failed.\n"
                f"Original error: {type(e).__name__}: {e}\n"
                f"x: shape={tuple(hidden_states.shape)} "
                f"stride={hidden_states.stride()} dtype={hidden_states.dtype}\n"
                f"w1: shape={tuple(w1_sonic.shape)} "
                f"stride={w1_sonic.stride()} dtype={w1_sonic.dtype}\n"
                f"z: shape={tuple(z.shape)} stride={z.stride()} dtype={z.dtype}\n"
                f"y1: shape={tuple(y1.shape)} stride={y1.stride()} dtype={y1.dtype}\n"
            ) from None

        try:
            _down_projection_forward(
                w2=w2_sonic,
                y1=y1,
                y2=y2,
                b2=None,
                expert_frequency_offset=expert_offsets,
                expert_schedule_order=expert_schedule_order,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
            )
        except Exception as e:
            raise RuntimeError(
                "SonicMoE down-projection failed.\n"
                f"Original error: {type(e).__name__}: {e}\n"
                f"w2: shape={tuple(w2_sonic.shape)} "
                f"stride={w2_sonic.stride()} dtype={w2_sonic.dtype}\n"
                f"y1: shape={tuple(y1.shape)} stride={y1.stride()} dtype={y1.dtype}\n"
                f"y2: shape={tuple(y2.shape)} stride={y2.stride()} dtype={y2.dtype}\n"
            ) from None

        # apply_router_weight_on_input only supported for topk=1
        # (consistent with MoEPrepareAndFinalizeNoEP)
        if apply_router_weight_on_input:
            if topk != 1:
                raise ValueError(
                    "apply_router_weight_on_input is only supported for topk=1"
                )
            topk_scores = torch.ones_like(topk_weights).flatten()
        else:
            # SonicMoE expects router scores in the original (token-major)
            # flattened order; `s_reverse_scatter_idx` maps that order to the
            # expert-grouped order of y2.
            topk_scores = topk_weights.flatten()
        try:
            _router_forward(
                y2=y2,
                o=output,
                topk_scores=topk_scores,
                s_reverse_scatter_idx=s_reverse_scatter_idx,
                num_activated_expert_per_token_offset=torch.arange(
                    0, M * topk + 1, topk, device=output.device, dtype=torch.int32
                ),
                varlen_K_max=topk,
                H=K,
                is_varlen_K=False,
            )
        except Exception as e:
            raise RuntimeError(
                "SonicMoE router reduction failed.\n"
                f"Original error: {type(e).__name__}: {e}\n"
                f"y2: shape={tuple(y2.shape)} stride={y2.stride()} dtype={y2.dtype}\n"
                f"o: shape={tuple(output.shape)} "
                f"stride={output.stride()} dtype={output.dtype}\n"
            ) from None


def sonic_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sonic MoE forward pass using modular kernel infrastructure.
    """
    if not is_sonic_moe_supported():
        raise RuntimeError(
            "Sonic MoE is not supported on this system. "
            "Requires: SonicMoE + Hopper GPU (H100/H200)"
        )

    dtype = hidden_states.dtype
    num_experts = w1.size(0)
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk_ids.size(1),
        hidden_dim=hidden_states.size(1),
        intermediate_size_per_partition=w1.size(1) // 2,
        num_local_experts=num_experts,
        num_logical_experts=(
            global_num_experts if global_num_experts > 0 else num_experts
        ),
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=dtype,
        device=hidden_states.device,
        routing_method=RoutingMethodType.TopK,
        is_act_and_mul=True,
    )
    fused_experts = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        SonicMoeExperts(moe_config=moe_config),
    )

    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
