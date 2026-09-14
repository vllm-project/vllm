# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from inspect import signature

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def make_mega_moe_expert_params_mapping(
    num_experts: int,
    ckpt_gate_proj_name: str = "w1",
    ckpt_down_proj_name: str = "w2",
    ckpt_up_proj_name: str = "w3",
) -> list[tuple[str, str, int, str]]:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ]
    ]


class DeepGemmMegaMoEExperts(nn.Module):
    """DeepGEMM expert-parallel MegaMoE for FP4 or BF16 expert weights."""

    _symm_buffer_cache: dict[
        tuple[int, int, int, int, int, int, int, int, str], object
    ] = {}

    @staticmethod
    def source_is_mxfp4(
        quant_config: QuantizationConfig | None,
        layer: nn.Module | None = None,
        prefix: str | None = None,
    ) -> bool:
        if quant_config is None or quant_config.get_name() != "compressed-tensors":
            return False
        source_format = getattr(quant_config, "quant_format", None)
        if layer is not None and prefix is not None:
            get_scheme_dict = getattr(quant_config, "get_scheme_dict", None)
            expert_prefix = (
                prefix if prefix.endswith(".experts") else f"{prefix}.experts"
            )
            if should_ignore_layer(
                expert_prefix,
                ignore=getattr(quant_config, "ignore", ()),
                fused_mapping=getattr(quant_config, "packed_modules_mapping", {}),
            ):
                return False
            if get_scheme_dict is not None:
                scheme_dict = get_scheme_dict(layer, expert_prefix)
                if scheme_dict is not None:
                    source_format = scheme_dict.get("format") or source_format
        elif source_format is None:
            config = getattr(quant_config, "config", None) or {}
            source_format = config.get("format")
        if source_format != "mxfp4-pack-quantized":
            raise NotImplementedError(
                "DeepGEMM MegaMoE compressed-tensors integration supports "
                "MXFP4 expert checkpoints."
            )
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        source_mxfp4: bool = False,
        prefix: str = "",
        num_logical_experts: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        if mma_type not in ("fp8xfp4", "bf16xbf16"):
            raise ValueError(f"Unsupported DeepGEMM MegaMoE MMA type: {mma_type}")
        if source_mxfp4 and mma_type != "fp8xfp4":
            raise ValueError("MXFP4 source weights require FP8xFP4 MegaMoE.")
        self.mma_type = mma_type
        self.source_mxfp4 = source_mxfp4
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )

        self.eplb_state = EplbLayerState()

        weight_attrs = {"weight_loader": self.weight_loader}
        source_is_packed = source_mxfp4
        if source_is_packed or mma_type == "fp8xfp4":
            weight_dtype = torch.uint8
        else:
            weight_dtype = torch.bfloat16
        uses_packed_storage = mma_type == "fp8xfp4"
        packed_hidden_size = hidden_size // 2 if uses_packed_storage else hidden_size
        packed_intermediate_size = (
            intermediate_size // 2 if uses_packed_storage else intermediate_size
        )
        w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                packed_hidden_size,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w13_weight, weight_attrs)
        if source_is_packed:
            self.register_parameter("w13_weight_packed", w13_weight)
            self.w13_weight = None
        else:
            self.register_parameter("w13_weight", w13_weight)
            self.w13_weight_packed = None

        if mma_type == "fp8xfp4":
            self.w13_weight_scale = nn.Parameter(
                torch.zeros(
                    num_local_experts,
                    2 * intermediate_size,
                    hidden_size // 32,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.w13_weight_scale, weight_attrs)
            self.w13_weight_scale.quant_method = "block"
        else:
            self.w13_weight_scale = None

        w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                packed_intermediate_size,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight, weight_attrs)
        if source_is_packed:
            self.register_parameter("w2_weight_packed", w2_weight)
            self.w2_weight = None
        else:
            self.register_parameter("w2_weight", w2_weight)
            self.w2_weight_packed = None

        if mma_type == "fp8xfp4":
            self.w2_weight_scale = nn.Parameter(
                torch.zeros(
                    num_local_experts,
                    hidden_size,
                    intermediate_size // 32,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.w2_weight_scale, weight_attrs)
            self.w2_weight_scale.quant_method = "block"
        else:
            self.w2_weight_scale = None

        self._transformed_l1_weights: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_l2_weights: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_shared_l1_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_shared_l2_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Return local (per-rank) slot offsets where logical expert
        `expert_id` should land on this rank.
        """
        physical_ids: list[int] = []
        for p in range(self.experts_start_idx, self.experts_end_idx):
            if p % self.num_logical_experts == expert_id:
                physical_ids.append(p - self.experts_start_idx)
        return physical_ids

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if not local_expert_ids:
            return False if return_success else None

        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_size = expert_data.shape[0] // 2
                shard_offset = 0 if shard_id == "w1" else shard_size
                expert_data = expert_data.narrow(0, shard_offset, shard_size)
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.numel() == 1 and loaded_weight.numel() == 1:
                loaded_weight = loaded_weight.reshape_as(expert_data)
            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"DeepGEMM MegaMoE expert weight shape mismatch for "
                    f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                    f"vs checkpoint {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True

        if return_success:
            return loaded_any
        return None

    @staticmethod
    def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
        return (sf.to(torch.int32) << 23).view(torch.float32)

    def _check_runtime_supported(self) -> None:
        loader_weight = self.w13_weight_packed if self.source_mxfp4 else self.w13_weight
        assert loader_weight is not None
        device = loader_weight.device
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    @staticmethod
    def _deep_gemm_supports_shared_experts(deep_gemm) -> bool:
        """Check the Python API before touching a symmetric-memory group.

        This also gives users of an older precompiled vLLM wheel a safe serial
        fallback instead of failing halfway through multi-rank buffer setup.
        """
        try:
            buffer_params = signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters
            kernel_params = signature(deep_gemm.fp8_fp4_mega_moe).parameters
        except (TypeError, ValueError):
            return False
        return (
            hasattr(deep_gemm, "get_block_m_for_mega_moe")
            and hasattr(deep_gemm, "transform_weights_for_mega_moe")
            and "num_shared_experts" in buffer_params
            and "shared_l1_weights" in kernel_params
            and "shared_l2_weights" in kernel_params
        )

    def _finalize_shared_expert_weights(
        self, deep_gemm, shared_experts: nn.Module
    ) -> None:
        gate_up = shared_experts.gate_up_proj
        down = shared_experts.down_proj
        gate_up_weight = gate_up.weight.data
        gate_up_scale = (
            gate_up.weight_scale
            if hasattr(gate_up, "weight_scale")
            else gate_up.weight_scale_inv
        ).data
        down_weight = down.weight.data
        down_scale = (
            down.weight_scale
            if hasattr(down, "weight_scale")
            else down.weight_scale_inv
        ).data

        # MegaMoE's shared FP8 MMA consumes a 1x32 scale for every weight row,
        # while the checkpoint uses coarser block-FP8 scales (usually
        # 128x128). Build a dedicated, numerically equivalent scale view before
        # the generic linear post-load hook replaces the raw checkpoint scales
        # with its 128x128 DeepGEMM layout.
        checkpoint_scale_dtypes = (torch.float8_e8m0fnu, torch.uint8)
        if (
            gate_up_scale.dtype in checkpoint_scale_dtypes
            and down_scale.dtype in checkpoint_scale_dtypes
        ):
            gate_up_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                gate_up,
                gate_up_scale,
                gate_up_weight.shape[0],
                gate_up_weight.shape[1],
            )
            down_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                down,
                down_scale,
                down_weight.shape[0],
                down_weight.shape[1],
            )

        if gate_up_scale is None or down_scale is None:
            self.num_shared_experts = 0
            return

        shared_intermediate_size = self.intermediate_size * self.num_shared_experts
        expected_gate_up_shape = (
            2 * shared_intermediate_size,
            self.hidden_size,
        )
        expected_down_shape = (self.hidden_size, shared_intermediate_size)
        if (
            gate_up_weight.dtype != torch.float8_e4m3fn
            or down_weight.dtype != torch.float8_e4m3fn
            or gate_up_scale.dtype != torch.int32
            or down_scale.dtype != torch.int32
            or tuple(gate_up_weight.shape) != expected_gate_up_shape
            or tuple(down_weight.shape) != expected_down_shape
        ):
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: expected "
                "replicated block-FP8 weights with gate_up=%s, down=%s, and "
                "DeepGEMM int32 scales; got gate_up=%s/%s/%s and down=%s/%s/%s.",
                self.prefix,
                expected_gate_up_shape,
                expected_down_shape,
                tuple(gate_up_weight.shape),
                gate_up_weight.dtype,
                gate_up_scale.dtype,
                tuple(down_weight.shape),
                down_weight.dtype,
                down_scale.dtype,
            )
            self.num_shared_experts = 0
            return

        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
            (gate_up_weight, gate_up_scale),
            (down_weight, down_scale),
        )
        # L1 interleaving allocates a full copy. Re-home the loader Parameter on
        # that storage so the original 2*intermediate*hidden FP8 tensor can be
        # released instead of adding roughly 0.7 GiB per rank on DSV4-Flash.
        # The generic linear post-load hook may still repack the serial scales,
        # but this shared MLP is never called after native fusion is enabled.
        gate_up.weight.data = transformed_l1[0]
        self._transformed_shared_l1_weights = (
            gate_up.weight.data,
            transformed_l1[1],
        )
        self._transformed_shared_l2_weights = transformed_l2

    def _prepare_shared_expert_scale(
        self,
        deep_gemm,
        linear: nn.Module,
        scale: torch.Tensor,
        mn: int,
        k: int,
    ) -> torch.Tensor | None:
        block_size = getattr(linear, "weight_block_size", None)
        if block_size is None or len(block_size) != 2:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "shared FP8 weight block size is unavailable.",
                self.prefix,
            )
            return None

        block_m, block_k = block_size
        expected_shape = (
            (mn + block_m - 1) // block_m,
            (k + block_k - 1) // block_k,
        )
        if block_k % 32 != 0 or tuple(scale.shape) != expected_shape:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "cannot convert shared scale shape %s with block size %s "
                "to MegaMoE's 1x32 layout for weight (%d, %d).",
                self.prefix,
                tuple(scale.shape),
                tuple(block_size),
                mn,
                k,
            )
            return None

        scale_fp32 = self._ue8m0_uint8_to_float(scale.view(torch.uint8))
        scale_1x32 = (
            scale_fp32.repeat_interleave(block_m, dim=0)
            .repeat_interleave(block_k // 32, dim=1)[:mn, : k // 32]
            .contiguous()
        )
        # The grouped API is used with a singleton dimension to request the
        # MN-major, TMA-aligned packed-UE8M0 strides, then squeezed back to the
        # 2D layout required for a shared expert.
        return deep_gemm.transform_sf_into_required_layout(
            scale_1x32.unsqueeze(0),
            mn,
            k,
            (1, 32),
            1,
        ).squeeze(0)

    def finalize_weights(self, shared_experts: nn.Module | None = None) -> None:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        if self._transformed_l1_weights is None:
            self._check_runtime_supported()
            if self.mma_type == "bf16xbf16":
                assert self.w13_weight is not None
                assert self.w2_weight is not None
                self._transformed_l1_weights, self._transformed_l2_weights = (
                    deep_gemm.transform_weights_for_mega_moe(
                        self.w13_weight.data,
                        self.w2_weight.data,
                    )
                )
            else:
                w13_weight = (
                    self.w13_weight_packed if self.source_mxfp4 else self.w13_weight
                )
                w2_weight = (
                    self.w2_weight_packed if self.source_mxfp4 else self.w2_weight
                )
                assert w13_weight is not None
                assert w2_weight is not None
                assert self.w13_weight_scale is not None
                assert self.w2_weight_scale is not None
                # MegaMoE's 1x32 activation scales need 16-byte TMA rows, so
                # pad each gate/up half and the down projection to a multiple of 512.
                padded_size = (self.intermediate_size + 511) // 512 * 512
                padding = padded_size - self.intermediate_size
                if padding:
                    for param in (w13_weight, self.w13_weight_scale):
                        gate_up = param.data.unflatten(1, (2, self.intermediate_size))
                        param.data = torch.nn.functional.pad(
                            gate_up, (0, 0, 0, padding)
                        ).flatten(1, 2)
                    w2_weight.data = torch.nn.functional.pad(
                        w2_weight.data, (0, padding // 2)
                    )
                    self.w2_weight_scale.data = torch.nn.functional.pad(
                        self.w2_weight_scale.data, (0, padding // 32)
                    )
                    self.intermediate_size = padded_size
                w13_scale = deep_gemm.transform_sf_into_required_layout(
                    self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
                    2 * self.intermediate_size,
                    self.hidden_size,
                    (1, 32),
                    self.num_local_experts,
                )
                w2_scale = deep_gemm.transform_sf_into_required_layout(
                    self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
                    self.hidden_size,
                    self.intermediate_size,
                    (1, 32),
                    self.num_local_experts,
                )
                self._transformed_l1_weights, self._transformed_l2_weights = (
                    deep_gemm.transform_weights_for_mega_moe(
                        (
                            w13_weight.data.view(torch.int8).contiguous(),
                            w13_scale,
                        ),
                        (w2_weight.data.view(torch.int8).contiguous(), w2_scale),
                    )
                )
            # Drop the original loader-side parameters: the MegaMoE kernels only
            # consume the transformed views above. transform_weights_for_mega_moe
            # allocates a fresh tensor for the L1 weight (see
            # _interleave_l1_weights) and fresh SF tensors for L1/L2; the L2
            # weight is the only tensor that aliases the original storage, and
            # _transformed_l2_weights still holds it, so the storage stays live
            # after we drop the Parameter.
            self.w13_weight = None
            self.w13_weight_packed = None
            self.w13_weight_scale = None
            self.w2_weight = None
            self.w2_weight_packed = None
            self.w2_weight_scale = None

        if shared_experts is None or self.num_shared_experts == 0:
            return
        if self._transformed_shared_l1_weights is not None:
            return
        if not self._deep_gemm_supports_shared_experts(deep_gemm):
            logger.warning_once(
                "Disabling native MegaMoE shared-expert fusion because the "
                "installed DeepGEMM Python API is older than the vLLM "
                "source. Rebuild the vendored _deep_gemm_C extension to enable it.",
            )
            self.num_shared_experts = 0
            return
        self._finalize_shared_expert_weights(deep_gemm, shared_experts)

    @property
    def has_fused_shared_experts(self) -> bool:
        return self._transformed_shared_l1_weights is not None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        group = get_ep_group().device_group
        assert self._transformed_l1_weights is not None
        l1_weight = self._transformed_l1_weights
        device = (l1_weight[0] if isinstance(l1_weight, tuple) else l1_weight).device
        assert device.index is not None
        key = (
            id(group),
            device.index,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts if self.has_fused_shared_experts else 0,
            self.mma_type,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            with torch.accelerator.device_index(device.index):
                symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                    group,
                    self.num_experts,
                    self.max_num_tokens,
                    self.top_k,
                    self.hidden_size,
                    self.intermediate_size,
                    num_shared_experts=(
                        self.num_shared_experts if self.has_fused_shared_experts else 0
                    ),
                    mma_type=self.mma_type,
                )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None

        def _to_eplb_view(name: str, t: torch.Tensor) -> torch.Tensor:
            """Return a (num_local_experts, -1) view with contiguous memory layout."""
            assert t.shape[0] == self.num_local_experts
            if t.is_contiguous():
                return t.view(self.num_local_experts, -1)
            elif t.dim() == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1]:
                # scales have shape (E, M, N) with memory layout (E, N, M)
                back = torch.transpose(t, 1, 2)
                assert back.is_contiguous()
                return back.view(self.num_local_experts, -1)

            raise AssertionError(
                f"MegaMoE EPLB {name}: non-contiguous expert tensor with "
                f"unexpected layout shape={tuple(t.shape)} "
                f"stride={tuple(t.stride())} dtype={t.dtype}"
            )

        if self.mma_type == "bf16xbf16":
            assert isinstance(self._transformed_l1_weights, torch.Tensor)
            assert isinstance(self._transformed_l2_weights, torch.Tensor)
            return [
                _to_eplb_view("l1_weight", self._transformed_l1_weights),
                _to_eplb_view("l2_weight", self._transformed_l2_weights),
            ]

        assert isinstance(self._transformed_l1_weights, tuple)
        assert isinstance(self._transformed_l2_weights, tuple)
        return [
            _to_eplb_view("l1_packed", self._transformed_l1_weights[0]),
            _to_eplb_view("l1_scale", self._transformed_l1_weights[1]),
            _to_eplb_view("l2_weight", self._transformed_l2_weights[0]),
            _to_eplb_view("l2_scale", self._transformed_l2_weights[1]),
        ]

    def update_expert_map(self) -> None:
        pass

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepGEMM MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # EPLB: map logical expert IDs to physical replicas and record load.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        shared_x_sf = None
        shared_block_m = None
        if self.has_fused_shared_experts:
            shared_x_sf = symm_buffer.shared_l1_acts_sf
            shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                get_ep_group().world_size,
                self.num_experts,
                symm_buffer.num_max_tokens_per_rank,
                num_tokens,
                self.top_k,
                "fp8xfp4",
            )

        if self.mma_type == "bf16xbf16":
            symm_buffer.x[:num_tokens].copy_(hidden_states)
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
                topk_weights = torch.where(is_padding.unsqueeze(1), 0.0, topk_weights)
            symm_buffer.topk_idx[:num_tokens].copy_(topk_ids)
            symm_buffer.topk_weights[:num_tokens].copy_(topk_weights)
        else:
            prepare_megamoe_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens],
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
                is_padding=is_padding,
                shared_x_sf=shared_x_sf,
                shared_block_m=shared_block_m,
            )

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        kernel = (
            deep_gemm.bf16_mega_moe
            if self.mma_type == "bf16xbf16"
            else deep_gemm.fp8_fp4_mega_moe
        )
        if self.has_fused_shared_experts:
            kernel(
                y,
                self._transformed_l1_weights,
                self._transformed_l2_weights,
                symm_buffer,
                shared_l1_weights=self._transformed_shared_l1_weights,
                shared_l2_weights=self._transformed_shared_l2_weights,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        else:
            kernel(
                y,
                self._transformed_l1_weights,
                self._transformed_l2_weights,
                symm_buffer,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        return y


DeepGemmMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
