# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import (
    _interleave_gate_up_rows,
    make_test_weights,
    per_token_cast_to_fp8,
)
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_experts
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.utils.import_utils import (
    has_aiter,
    has_deep_ep,
    has_deep_ep_v2,
    has_deep_gemm,
    has_mori,
)
from vllm.utils.math_utils import next_power_of_2

from .mk_objects import (
    TestMoEQuantConfig,
    expert_info,
    make_fused_experts,
    prepare_finalize_info,
)
from .parallel_utils import ProcessGroupInfo


def _describe_tensor(t: torch.Tensor | None, name: str) -> str:
    if t is None:
        return f"{name} : None"
    else:
        return f"{name} : {t.shape} {t.dtype} {t.device}"


@dataclass
class Config:
    Ms: list[int] | int
    K: int
    N: int
    E: int
    topks: list[int] | int
    dtype: torch.dtype
    quant_config: TestMoEQuantConfig | None

    prepare_finalize_type: mk.FusedMoEPrepareAndFinalize
    fused_experts_type: mk.FusedMoEExperts

    world_size: int

    activation: MoEActivation = MoEActivation.SILU

    torch_trace_dir_path: str | None = None

    # Force AiterExperts's hidden_pad/intermediate_pad computation
    # (`experts/rocm_aiter_moe.py`) to diverge from the padded K/N sizes above.
    # None (default) preserves today's behavior: FusedMoEConfig defaults both
    # to the (unpadded) K/intermediate_size_per_partition, so hidden_pad and
    # intermediate_pad come out to 0.
    # See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").
    hidden_dim_unpadded: int | None = None
    intermediate_size_per_partition_unpadded: int | None = None

    def __post_init__(self):
        if self.quant_config is None:
            self.quant_config = TestMoEQuantConfig(None, False, False, None)

    def describe(self) -> str:
        s = ""
        s += "== Config:\n"
        s += f" world_size={self.world_size}\n"
        s += f" PF={self.prepare_finalize_type.__name__}\n"
        s += f" FE={self.fused_experts_type.__name__}\n"
        s += f" E={self.E}\n"
        s += f" Ms={self.Ms}\n"
        s += f" N={self.N}\n"
        s += f" K={self.K}\n"
        s += f" topk={self.topks}\n"
        s += f" dtype={self.dtype}\n"
        s += " Quant:\n"
        if self.quant_config is not None:
            s += f"     q_dtype={self.quant_dtype}\n"
            s += f"     w_dtype={self.weight_dtype}\n"
            s += f"     q_block_shape={self.quant_block_shape}\n"
            s += f"     q_per_out_ch_quant={self.is_per_out_ch_quant}\n"
            s += f"     q_per_act_token={self.is_per_act_token_quant}\n"
        else:
            s += "     quant=None\n"
        return s

    @property
    def M(self) -> int:
        assert isinstance(self.Ms, int)
        return self.Ms

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        assert self.quant_config is not None
        return self.quant_config.quant_dtype

    @property
    def weight_dtype(self) -> torch.dtype | str | None:
        """Weight quant dtype, defaulting to quant_dtype when unset (mirrors
        FusedMoEQuantConfig.make()'s own weight_dtype fallback)."""
        assert self.quant_config is not None
        wd = self.quant_config.weight_dtype
        return wd if wd is not None else self.quant_config.quant_dtype

    @property
    def is_per_act_token_quant(self) -> bool:
        assert self.quant_config is not None
        return self.quant_config.per_act_token_quant

    @property
    def is_per_tensor_act_quant(self) -> bool:
        return not self.is_per_act_token_quant and self.quant_block_shape is None

    @property
    def is_per_out_ch_quant(self) -> bool:
        assert self.quant_config is not None
        return self.quant_config.per_out_ch_quant

    @property
    def quant_block_shape(self) -> list[int] | None:
        assert self.quant_config is not None
        return self.quant_config.block_shape

    @property
    def topk(self) -> int:
        assert isinstance(self.topks, int)
        return self.topks

    @property
    def num_local_experts(self) -> int:
        return self.E // self.world_size

    def make_env_data(self) -> tuple[VllmConfig, dict[Any, Any]]:
        """Make env data for vllm launch."""
        vllm_config = VllmConfig()
        vllm_config.model_config = SimpleNamespace(
            enforce_eager=True,
            is_moe=True,
        )
        vllm_config.parallel_config.data_parallel_size = self.world_size
        vllm_config.parallel_config.enable_expert_parallel = True

        env_dict = {
            "VLLM_USE_DEEP_GEMM": str(int(self.needs_deep_gemm())),
        }

        vllm_config.parallel_config.all2all_backend = self.all2all_backend()

        return vllm_config, env_dict

    def fp8_quant_key_pair(self) -> tuple[QuantKey, QuantKey]:
        """Derive the (weight_quant_key, activation_quant_key) pair an FP8
        quant config of this shape corresponds to (either OCP or FNUZ FP8,
        see ``current_platform.fp8_dtype()``)."""
        if self.quant_block_shape is not None:
            return kFp8Static128BlockSym, kFp8Dynamic128Sym
        if self.is_per_out_ch_quant:
            return (
                kFp8StaticChannelSym,
                kFp8DynamicTokenSym
                if self.is_per_act_token_quant
                else kFp8StaticTensorSym,
            )
        return (
            kFp8StaticTensorSym,
            kFp8DynamicTensorSym
            if self.is_per_act_token_quant
            else kFp8StaticTensorSym,
        )

    def mxfp4_quant_key_pair(self) -> tuple[QuantKey, QuantKey]:
        """Derive the (weight_quant_key, activation_quant_key) pair an mxfp4
        quant config of this shape corresponds to (W4A16 when activations
        are unquantized, W4A4 when they are also mxfp4)."""
        return kMxfp4Static, (kMxfp4Dynamic if self.quant_dtype == "mxfp4" else None)

    @property
    def is_w4a16_mxfp4(self) -> bool:
        """True for the asymmetric W4A16 mxfp4 case (mxfp4 weight, unquantized
        activation), as opposed to the symmetric W4A4 case (both mxfp4)."""
        return self.quant_dtype is None and self.weight_dtype == "mxfp4"

    @property
    def mxfp4_backend(self) -> "Mxfp4MoeBackend":
        """The AITER mxfp4 kernel backend this config's (weight, activation)
        dtype pair maps to."""
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

        assert self.weight_dtype == "mxfp4"
        return (
            Mxfp4MoeBackend.AITER
            if self.is_w4a16_mxfp4
            else Mxfp4MoeBackend.AITER_MXFP4_MXFP4
        )

    def fe_supports_quant_scheme(self) -> bool:
        """Check if the fused experts class supports this quant config.
        See https://github.com/ROCm/aiter/issues/2419 for AITER gaps."""
        if self.quant_config is None or self.weight_dtype is None:
            return True
        if self.weight_dtype == "mxfp4":
            w_key, a_key = self.mxfp4_quant_key_pair()
        elif is_fp8(self.quant_dtype):
            w_key, a_key = self.fp8_quant_key_pair()
        else:
            return True
        fe_cls = self.fused_experts_type
        if hasattr(fe_cls, "_supports_quant_scheme"):
            try:
                return fe_cls._supports_quant_scheme(w_key, a_key)
            except NotImplementedError:
                pass
        return True

    def is_fp8_block_quantized(self):
        return is_fp8(self.quant_dtype) and self.quant_block_shape is not None

    def is_batched_prepare_finalize(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return mk.FusedMoEActivationFormat.BatchedExperts == info.activation_format

    def is_batched_fused_experts(self):
        info = expert_info(self.fused_experts_type)
        return mk.FusedMoEActivationFormat.BatchedExperts == info.activation_format

    def is_standard_fused_experts(self):
        info = expert_info(self.fused_experts_type)
        return mk.FusedMoEActivationFormat.Standard == info.activation_format

    def fe_supported_types(self):
        info = expert_info(self.fused_experts_type)
        return info.supported_dtypes

    def pf_supported_types(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.supported_dtypes

    def is_block_quant_supported(self):
        info = expert_info(self.fused_experts_type)
        return info.blocked_quantization_support

    def supports_apply_weight_on_input(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.supports_apply_weight_on_input

    def needs_deep_gemm(self):
        info = expert_info(self.fused_experts_type)
        return info.needs_deep_gemm

    def needs_deep_ep(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return (
            info.backend == "deepep_high_throughput"
            or info.backend == "deepep_low_latency"
        )

    def needs_deep_ep_v2(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend == "deepep_v2"

    def needs_aiter(self):
        info = expert_info(self.fused_experts_type)
        return info.needs_aiter

    def needs_mori(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend in ("mori_high_throughput", "mori_low_latency")

    def all2all_backend(self):
        info = prepare_finalize_info(self.prepare_finalize_type)
        return info.backend

    def is_valid(self) -> tuple[bool, str | None]:
        # Check prepare-finalize and fused-experts compatibility
        if self.is_batched_prepare_finalize():
            if not self.is_batched_fused_experts():
                return False, "Mismatched format."
        else:
            if not self.is_standard_fused_experts():
                return False, "Mismatched format."

        # Check quantization sanity
        if (
            int(self.is_per_act_token_quant)
            + int(self.is_per_tensor_act_quant)
            + int(self.quant_block_shape is not None)
        ) > 1:
            # invalid quant config
            return False, f"Bad quant_config {self.quant_config}."

        # check type support
        if self.weight_dtype is None:
            if (
                self.dtype not in self.pf_supported_types()
                or self.dtype not in self.fe_supported_types()
            ):
                return False, (
                    f"Unsupported type {self.dtype} not in "
                    f"{self.pf_supported_types()} and "
                    f"{self.fe_supported_types()}."
                )
        else:
            dtypes_to_check = {self.quant_dtype, self.weight_dtype} - {None}
            for dtype in dtypes_to_check:
                if (
                    dtype not in self.pf_supported_types()
                    or dtype not in self.fe_supported_types()
                ):
                    return False, (
                        f"Unsupported quant type {dtype} "
                        f"not in {self.pf_supported_types()} and "
                        f"{self.fe_supported_types()}."
                    )

        # Check quant scheme compatibility with fused experts class
        if not self.fe_supports_quant_scheme():
            return False, (
                f"FE {self.fused_experts_type.__name__} does not support "
                f"quant scheme (per_out_ch={self.is_per_out_ch_quant}, "
                f"per_act_token={self.is_per_act_token_quant}, "
                f"block={self.quant_block_shape})"
            )

        # Check activation support; NotImplementedError means no opinion.
        try:
            if not self.fused_experts_type._supports_activation(self.activation):
                return False, (
                    f"FE {self.fused_experts_type.__name__} does not support "
                    f"activation {self.activation}"
                )
        except NotImplementedError:
            pass

        # Check block quantization support
        is_block_quantized = self.quant_block_shape is not None
        if is_block_quantized and self.quant_dtype is None:
            return False, "No block quantization support."

        if is_block_quantized and not self.is_block_quant_supported():
            return False, "Mismatched block quantization support."

        # deep_gemm only works with block-quantized
        if self.needs_deep_gemm() and not is_block_quantized:
            return False, "Needs DeepGEMM but not block quantized."

        # Check dependencies (turn into asserts?)
        if self.needs_deep_ep() and not has_deep_ep():
            return False, "Needs DeepEP, but DeepEP not available."
        if self.needs_deep_ep_v2() and not has_deep_ep_v2():
            return False, "Needs DeepEP v2, but DeepEP v2 not available."
        if self.needs_deep_gemm() and not has_deep_gemm():
            return (
                False,
                "Needs DeepGEMM, but the current vLLM environment does not provide it.",
            )
        if self.needs_aiter() and not has_aiter():  # noqa: SIM103
            return False, "Needs Aiter, but Aiter not available."
        if self.needs_mori() and not has_mori():  # noqa: SIM103
            return False, "Needs MoRI, but MoRI not available."
        if self.needs_mori() and not rocm_aiter_ops.is_fused_moe_enabled():
            return False, (
                "Mori requires AITER's fused-moe backend to be enabled "
                "(VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1)."
            )

        try:
            if not self.fused_experts_type._supports_current_device():
                return (
                    False,
                    f"{self.fused_experts_type} not supported on the current device.",
                )
        except NotImplementedError:
            pass

        return True, None


@dataclass
class WeightTensors:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    w1_gs: torch.Tensor | None = None
    w2_gs: torch.Tensor | None = None

    def describe(self):
        s = ""
        s += "== Weight Tensors: \n"
        s += f" - {_describe_tensor(self.w1, 'w1')} \n"
        s += f" - {_describe_tensor(self.w2, 'w2')} \n"
        s += f" - {_describe_tensor(self.w1_scale, 'w1_scale')} \n"
        s += f" - {_describe_tensor(self.w2_scale, 'w2_scale')} \n"
        s += f" - {_describe_tensor(self.w1_gs, 'w1_gs')} \n"
        s += f" - {_describe_tensor(self.w2_gs, 'w2_gs')} \n"
        return s

    def is_quantized(self) -> bool:
        # or w1_scale is not None?
        return (
            is_fp8(self.w1.dtype)
            or self.w1.dtype == torch.uint8
            or self.w1.dtype == torch.int8
        )

    def to_current_device(self):
        device = torch.accelerator.current_device_index()
        self.w1 = self.w1.to(device=device)
        self.w2 = self.w2.to(device=device)

        if self.w1_scale is not None:
            self.w1_scale = self.w1_scale.to(device=device)
        if self.w2_scale is not None:
            self.w2_scale = self.w2_scale.to(device=device)

        if self.w1_gs is not None:
            self.w1_gs = self.w1_gs.to(device=device)
        if self.w2_gs is not None:
            self.w2_gs = self.w2_gs.to(device=device)

    def slice_weights(self, rank: int, num_local_experts: int) -> "WeightTensors":
        s = rank * num_local_experts
        e = s + num_local_experts
        w1 = self.w1[s:e, :, :]
        w2 = self.w2[s:e, :, :]
        w1_scale = self.w1_scale[s:e, :, :] if self.w1_scale is not None else None
        w2_scale = self.w2_scale[s:e, :, :] if self.w2_scale is not None else None
        w1_gs = self.w1_gs[s:e] if self.w1_gs is not None else None
        w2_gs = self.w2_gs[s:e] if self.w2_gs is not None else None

        return WeightTensors(w1, w2, w1_scale, w2_scale, w1_gs, w2_gs)

    @staticmethod
    def make(config: Config) -> "WeightTensors":
        (_, w1, w1_scale, w1_gs), (_, w2, w2_scale, w2_gs) = make_test_weights(
            e=config.E,
            n=config.N,
            k=config.K,
            in_dtype=config.dtype,
            quant_dtype=config.weight_dtype,
            block_shape=config.quant_block_shape,
            # or config.is_per_out_ch_quant
            per_out_ch_quant=config.is_per_act_token_quant,
        )
        return WeightTensors(
            w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale, w1_gs=w1_gs, w2_gs=w2_gs
        )


@dataclass
class RankTensors:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: torch.Tensor | None

    def describe(self):
        s = ""
        s += "== Rank Tensors: \n"
        s += f" - {_describe_tensor(self.hidden_states, 'HS')} \n"
        s += f" - {_describe_tensor(self.hidden_states_scale, 'HS_scale')} \n"
        s += f" - {_describe_tensor(self.topk_weights, 'topk_weights')} \n"
        s += f" - {_describe_tensor(self.topk_ids, 'topk_ids')} \n"
        s += f" - {_describe_tensor(self.expert_map, 'expert_map')} \n"
        return s

    @staticmethod
    def make_hidden_states(
        config: Config,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return hidden_states."""
        m, k, dtype = (config.M, config.K, config.dtype)
        device = torch.accelerator.current_device_index()
        a = torch.randn((m, k), device=device, dtype=dtype) / 15.0

        if config.quant_dtype is None:
            return a, None

        if config.quant_dtype == "mxfp4":
            # Quantize and dequantize using the real mxfp4 kernels so the
            # dequantized input is already a fixed point of this
            # quantization (same rationale as the FP8 branches below). Must
            # be checked before the FP8 per-tensor/per-token branches, since
            # those conditions also evaluate true for this config.
            from triton_kernels.numerics_details.mxfp import (
                downcast_to_mxfp,
                upcast_from_mxfp,
            )

            a_q, a_scales = downcast_to_mxfp(a, torch.uint8, axis=-1)
            return upcast_from_mxfp(a_q, a_scales, dtype, axis=-1), None

        # We dequant and use that as hidden_states so the tests are stable.
        # quantizing and dequantizing yield slightly different results
        # depending on the hardware. Here we, quantize and dequantize
        # first - so further quantize and dequantize will yield the same
        # values.
        if config.is_per_tensor_act_quant:
            a_q, a_scales = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=False)
            return a_q.float().mul(a_scales).to(dtype), a_scales

        if config.is_per_act_token_quant:
            a_q, a_scales = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
            return a_q.float().mul(a_scales).to(dtype), None

        assert config.quant_block_shape is not None
        block_k = config.quant_block_shape[1]
        a_q, a_scales = per_token_cast_to_fp8(a, block_size=block_k)
        return a_q.float().view((-1, block_k)).mul(a_scales.view(-1, 1)).view(m, k).to(
            dtype
        ), None

    @staticmethod
    def make(config: Config, pgi: ProcessGroupInfo):
        dtype = config.dtype
        topk, m, _ = (config.topk, config.M, config.K)
        hidden_states, hidden_states_scale = RankTensors.make_hidden_states(config)

        num_local_experts, global_num_experts = (config.num_local_experts, config.E)
        score = torch.randn((m, global_num_experts), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk, False)

        # distribute topk_ids evenly
        device = torch.accelerator.current_device_index()
        for mi in range(m):
            topk_ids[mi] = torch.randperm(config.E)[:topk]
        topk_ids = topk_ids.to(device=device)

        expert_map = None
        if config.world_size > 1:
            expert_map = torch.full(
                (global_num_experts,), fill_value=-1, dtype=torch.int32
            )
            s = pgi.rank * num_local_experts
            e = s + num_local_experts
            expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
            expert_map = expert_map.to(device=device, dtype=torch.int32)

        return RankTensors(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )


def reference_moe_impl(
    config: Config, weights: WeightTensors, rank_tensors: RankTensors
) -> torch.Tensor:
    if config.quant_dtype == "nvfp4":
        quant_blocksize = 16
        dtype = config.dtype

        w1_q = weights.w1
        w1_blockscale = weights.w1_scale
        w1_gs = weights.w1_gs

        w2_q = weights.w2
        w2_blockscale = weights.w2_scale
        w2_gs = weights.w2_gs

        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            / torch.amax(rank_tensors.hidden_states.flatten(), dim=-1)
        ).to(torch.float32)

        assert w1_gs is not None
        assert w2_gs is not None
        assert w1_blockscale is not None
        assert w2_blockscale is not None

        assert w1_blockscale.shape[1] % 128 == 0
        assert w1_blockscale.shape[2] % 4 == 0
        assert w2_blockscale.shape[1] % 128 == 0
        assert w2_blockscale.shape[2] % 4 == 0

        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(
            rank_tensors.hidden_states, a_global_scale
        )

        a = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=dtype,
            device=a_fp4.device,
            block_size=quant_blocksize,
        )

        e = w1_q.shape[0]
        n = w1_q.shape[1] // 2
        k = w2_q.shape[1]

        w1 = torch.zeros((e, 2 * n, k), device="cuda", dtype=dtype)
        w2 = torch.zeros((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                w1_blockscale[idx],
                w1_gs[idx],
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                w2_blockscale[idx],
                w2_gs[idx],
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )
        a_scale = None
        w1_scale = None
        w2_scale = None
        quant_dtype = None
        per_act_token_quant = False
        block_shape = None
    elif config.weight_dtype == "mxfp4":
        from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

        dtype = config.dtype
        a = rank_tensors.hidden_states
        w1_q, w1_scale_q = weights.w1, weights.w1_scale
        if config.is_w4a16_mxfp4:
            assert config.activation == MoEActivation.SWIGLUOAI
            # SwigluOAIAndMul expects gate/up interleaved row order
            # (x[..., ::2], x[..., 1::2]), not the contiguous [gate; up]
            # halves that `weights.w1` is generated in. Mirror the same
            # reorder `_maybe_convert_weights_for_experts()` applies before
            # the real MK-path AITER kernel call.
            w1_q = _interleave_gate_up_rows(w1_q)
            w1_scale_q = _interleave_gate_up_rows(w1_scale_q)
        w1 = upcast_from_mxfp(w1_q, w1_scale_q, dtype, axis=-1)
        w2 = upcast_from_mxfp(weights.w2, weights.w2_scale, dtype, axis=-1)
        a_scale = None
        w1_scale = None
        w2_scale = None
        quant_dtype = None
        per_act_token_quant = False
        block_shape = None
    else:
        a = rank_tensors.hidden_states
        a_scale = rank_tensors.hidden_states_scale
        w1 = weights.w1
        w1_scale = weights.w1_scale
        w2 = weights.w2
        w2_scale = weights.w2_scale
        quant_dtype = config.quant_dtype
        per_act_token_quant = config.is_per_act_token_quant
        block_shape = config.quant_block_shape

    return torch_experts(
        a=a,
        w1=w1,
        w2=w2,
        topk_weight=rank_tensors.topk_weights,
        topk_ids=rank_tensors.topk_ids,
        global_num_experts=config.E,
        expert_map=None,
        activation=config.activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
        apply_router_weights_on_input=config.topk == 1
        and config.supports_apply_weight_on_input(),
    )


def _make_gscale(num_experts: int) -> torch.Tensor:
    return torch.ones(
        (num_experts,),
        device=torch.accelerator.current_device_index(),
        dtype=torch.float32,
    )


def make_modular_kernel(
    config: Config,
    vllm_config: VllmConfig,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEKernel:
    # make moe config
    moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        pcp_size_=get_pcp_group().world_size,
        dp_size_=get_dp_group().world_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )

    moe = FusedMoEConfig(
        num_experts=config.E,
        experts_per_token=config.topk,
        hidden_dim=config.K,
        intermediate_size=config.N,
        num_local_experts=config.num_local_experts,
        num_logical_experts=config.E,
        moe_parallel_config=moe_parallel_config,
        in_dtype=config.dtype,
        max_num_tokens=next_power_of_2(config.M),
        activation=config.activation,
        device=vllm_config.device_config.device,
        routing_method=RoutingMethodType.DeepSeekV3,
        hidden_dim_unpadded=config.hidden_dim_unpadded,
        intermediate_size_per_partition_unpadded=(
            config.intermediate_size_per_partition_unpadded
        ),
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe,
        quant_config=quant_config,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    fused_experts = make_fused_experts(
        config.fused_experts_type,
        moe,
        quant_config,
        prepare_finalize.num_dispatchers(),
        config.N,
    )

    modular_kernel = mk.FusedMoEKernel(
        prepare_finalize=prepare_finalize,
        fused_experts=fused_experts,
    )

    return modular_kernel


def _shuffle_weights_for_aiter(rank_weights: WeightTensors) -> WeightTensors:
    """Pre-shuffle weights so AITER selects its prebuilt `preshuffle_on` module.

    Production shuffles in `process_weights_after_loading`; this harness builds its
    tensors directly, so without this every call asks for a `preshuffle_off` module
    that is not prebuilt, and JIT-compiles a kernel the image already ships.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    w1, w2 = rocm_aiter_ops.shuffle_weights(rank_weights.w1, rank_weights.w2)
    w1.is_shuffled = True
    w2.is_shuffled = True
    return replace(rank_weights, w1=w1, w2=w2)


def _maybe_convert_weights_for_experts(
    config: Config,
    rank_weights: WeightTensors,
) -> WeightTensors:
    """Convert weights to expert-specific format (e.g., TrtLLM BlockMajorK)."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        convert_to_fp8_moe_kernel_format,
    )

    fe_type = config.fused_experts_type
    fe_name = getattr(fe_type, "__name__", "")

    if fe_name == "AiterExperts" and config.weight_dtype == "mxfp4":
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        )

        # The converter mutates its w13/w2 weight and scale args in place, and
        # rank_weights is a view into the shared master WeightTensors, so
        # everything passed in must be a fresh tensor.
        if config.is_w4a16_mxfp4:
            w1 = _interleave_gate_up_rows(rank_weights.w1)
            w1_scale = _interleave_gate_up_rows(rank_weights.w1_scale)
        else:
            w1 = rank_weights.w1.clone()
            w1_scale = rank_weights.w1_scale.clone()
        w2 = rank_weights.w2.clone()
        w2_scale = rank_weights.w2_scale.clone()

        w1, w2, w1_scale, w2_scale, _, _ = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=config.mxfp4_backend,
                layer=torch.nn.Module(),
                w13_weight=w1,
                w2_weight=w2,
                w13_weight_scale=w1_scale,
                w2_weight_scale=w2_scale,
            )
        )
        return WeightTensors(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale)

    # AITER's prebuilt modules carry a gfx950 instance table even on gfx942 and reject
    # intermediate sizes that are not a multiple of 256, but only for the fp8 per-tensor
    # and per-token schemes. Those keep JIT-compiling until
    # https://github.com/ROCm/aiter/issues/5766 is fixed.
    if fe_name == "AiterExperts" and (
        config.quant_dtype is None or config.quant_block_shape is not None
    ):
        return _shuffle_weights_for_aiter(rank_weights)

    backend: Fp8MoeBackend | None = None
    if fe_name == "TrtLlmFp8ExpertsModular":
        backend = Fp8MoeBackend.FLASHINFER_TRTLLM
    elif fe_name == "FlashInferExperts":
        backend = Fp8MoeBackend.FLASHINFER_CUTLASS

    if backend is None or not rank_weights.is_quantized():
        return rank_weights

    mock_layer = SimpleNamespace(
        weight_block_size=config.quant_block_shape,
        moe_config=SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=config.N,
        ),
        activation=SimpleNamespace(is_gated=True),
    )

    w1, w2, w1_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=backend,
        layer=mock_layer,
        w13=rank_weights.w1,
        w2=rank_weights.w2,
        w13_scale=rank_weights.w1_scale,
        w2_scale=rank_weights.w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )

    return WeightTensors(
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_gs=rank_weights.w1_gs,
        w2_gs=rank_weights.w2_gs,
    )


def run_modular_kernel(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    config: Config,
    weights: WeightTensors,
    rank_tensors: RankTensors,
) -> torch.Tensor:
    assert isinstance(config.Ms, int)
    assert isinstance(config.topks, int)

    # weights for rank
    rank_weights = weights.slice_weights(pgi.rank, config.num_local_experts)
    rank_weights = _maybe_convert_weights_for_experts(config, rank_weights)

    if config.quant_dtype == "nvfp4":
        gscale = _make_gscale(config.num_local_experts)
    else:
        gscale = None

    if config.weight_dtype == "mxfp4":
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            make_mxfp4_moe_quant_config,
        )

        quant_config = make_mxfp4_moe_quant_config(
            config.mxfp4_backend,
            w1_scale=rank_weights.w1_scale,
            w2_scale=rank_weights.w2_scale,
        )
    else:
        quant_config = FusedMoEQuantConfig.make(
            config.quant_dtype,
            w1_scale=rank_weights.w1_scale,
            w2_scale=rank_weights.w2_scale,
            a1_scale=rank_tensors.hidden_states_scale,
            g1_alphas=(1 / rank_weights.w1_gs)
            if rank_weights.w1_gs is not None
            else None,
            g2_alphas=(1 / rank_weights.w2_gs)
            if rank_weights.w2_gs is not None
            else None,
            a1_gscale=gscale,
            a2_gscale=gscale,
            block_shape=config.quant_block_shape,
            per_act_token_quant=config.is_per_act_token_quant,
            per_out_ch_quant=config.is_per_out_ch_quant,
        )

    mk = make_modular_kernel(config, vllm_config, quant_config)

    # impls might update the tensor in place
    hidden_states = rank_tensors.hidden_states.clone()

    topk_ids = rank_tensors.topk_ids.to(mk.prepare_finalize.topk_indices_dtype())

    mk_kwargs = {
        "hidden_states": hidden_states,
        "w1": rank_weights.w1,
        "w2": rank_weights.w2,
        "topk_weights": rank_tensors.topk_weights,
        "topk_ids": topk_ids,
        "activation": config.activation,
        "expert_map": rank_tensors.expert_map,
        "global_num_experts": config.E,
        "apply_router_weight_on_input": config.topk == 1
        and config.supports_apply_weight_on_input(),
    }

    num_tokens = rank_tensors.hidden_states.shape[0]
    num_tokens_across_dp = torch.tensor(
        [num_tokens] * config.world_size, device="cuda", dtype=torch.int
    )

    torch.distributed.barrier()

    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        out = mk.apply(**mk_kwargs)

    return out
