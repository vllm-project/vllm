# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x routed experts on DeepGEMM BF16 and NVFP4 MegaMoE kernels.

The ModelOpt NVFP4 checkpoint maps onto the kernel without requantization.
"""

import functools
import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.common.ops.sequence_parallel import is_sequence_parallel_active
from vllm.models.muse_spark.nvidia.ops.prepare_megamoe_bf16 import (
    prepare_megamoe_bf16_inputs,
)
from vllm.models.muse_spark.nvidia.ops.prepare_megamoe_nvfp4 import (
    prepare_megamoe_nvfp4_inputs,
)
from vllm.utils.deep_gemm import _import_deep_gemm

if TYPE_CHECKING:
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xRMSNorm

logger = init_logger(__name__)


@functools.cache
def _get_deep_gemm():
    module = _import_deep_gemm()
    if (
        module is None
        or not hasattr(module, "fp4_fp4_mega_moe")
        or not hasattr(module, "bf16_mega_moe")
        or "post_expert_norm_gamma"
        not in inspect.signature(module.bf16_mega_moe).parameters
        or "weight_layout" not in inspect.signature(module.fp4_fp4_mega_moe).parameters
    ):
        raise RuntimeError(
            "Muse Spark MegaMoE requires the pinned DeepGEMM build "
            "with BF16 post-expert normalization and TRT-LLM NVFP4 weight layout. "
            "Install it with tools/install_deepgemm.sh or rebuild the vLLM wheel."
        )
    return module


def get_trtllm_token_threshold(vllm_config: VllmConfig) -> int:
    parallel = vllm_config.parallel_config
    default_threshold = (
        16
        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        and isinstance(vllm_config.quant_config, ModelOptNvFp4Config)
        and parallel.enable_expert_parallel
        and parallel.tensor_parallel_size in (4, 8)
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        else 0
    )
    additional_config = vllm_config.additional_config
    assert isinstance(additional_config, dict)
    threshold = additional_config.get(
        "muse_spark_trtllm_token_threshold", default_threshold
    )
    if type(threshold) is not int or not 0 <= threshold <= 256:
        raise ValueError(
            "Muse Spark TRT-LLM token threshold must be an integer in [0, 256]"
        )
    return threshold


def _shard_padding_mask(is_padding: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """This rank's chunk of a full-length padding mask; the round-up tail pads."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    pad = num_tokens * tp_size - is_padding.shape[0]
    if pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, pad), value=True)
    return is_padding.narrow(0, tp_rank * num_tokens, num_tokens)


class MuseSparkMegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple, tuple] = {}

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
        post_expert_norm: "MuseSpark1xRMSNorm",
        num_max_tokens_per_rank: int,
        is_sequence_parallel: bool = False,
        is_nvfp4: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = num_max_tokens_per_rank
        self.is_sequence_parallel = is_sequence_parallel
        self.post_expert_norm = post_expert_norm
        self.is_nvfp4 = is_nvfp4
        self.trtllm_token_threshold = get_trtllm_token_threshold(vllm_config)
        self.sp_min_tokens = self.trtllm_token_threshold + 1
        self.weight_layout = (
            "trtllm" if is_nvfp4 and self.trtllm_token_threshold else "native"
        )
        if hidden_size % 256 != 0 or intermediate_size % 256 != 0:
            raise ValueError(
                "MuseSpark1x MegaMoE requires hidden and intermediate "
                "sizes to be multiples of 256 for its shared BF16/NVFP4 "
                f"buffer layouts; got {hidden_size}/{intermediate_size}."
            )

        attrs = {"weight_loader": self.weight_loader}

        def param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
            p = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, attrs)
            return p

        if is_nvfp4:
            e4m3 = torch.float8_e4m3fn
            self.w13_weight = param(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            )
            self.w13_weight_scale = param(
                num_local_experts, 2 * intermediate_size, hidden_size // 16, dtype=e4m3
            )
            self.w13_weight_scale_2 = param(num_local_experts, 2, dtype=torch.float32)
            self.w2_weight = param(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            )
            self.w2_weight_scale = param(
                num_local_experts, hidden_size, intermediate_size // 16, dtype=e4m3
            )
            self.w2_weight_scale_2 = param(num_local_experts, dtype=torch.float32)
            self.w2_input_scale = param(num_local_experts, dtype=torch.float32)
            self.w13_input_scale = param(num_local_experts, 2, dtype=torch.float32)

        else:
            self.w13_weight = param(
                num_local_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.bfloat16,
            )
            self.w2_weight = param(
                num_local_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
            )

        self._transformed_l1: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_l2: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._norm_gamma: torch.Tensor | None = None

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        **kwargs,
    ) -> None:
        shard = loaded_weight[self.experts_start_idx : self.experts_end_idx]
        if shard.shape != param.data.shape:
            raise ValueError(
                f"{self.prefix}: expert weight {weight_name!r} shard "
                f"{tuple(shard.shape)} does not fit parameter "
                f"{tuple(param.data.shape)}"
            )
        param.data.copy_(shard.to(param.data.dtype))

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, tensor in weights:
            param = getattr(self, name, None)
            if param is None:
                raise AttributeError(
                    f"{self.prefix}: unexpected expert weight {name!r}"
                )
            self.weight_loader(param, tensor, name)
            yield name

    def finalize_weights(self) -> None:
        """Repack checkpoint tensors into the layout the kernel reads."""
        if self._transformed_l1 is not None:
            return
        deep_gemm = _get_deep_gemm()

        if self.weight_layout == "trtllm":
            self._prepare_trtllm_weights()
        elif self.is_nvfp4:

            def sf_to_kernel_layout(sf: torch.Tensor) -> torch.Tensor:
                # Four E4M3 block scales per int32, then MN-major for the TMA loads.
                packed = sf.data.view(torch.uint8).contiguous().view(torch.int32)
                return packed.transpose(-1, -2).contiguous().transpose(-1, -2)

            self._transformed_l1, self._transformed_l2 = (
                deep_gemm.transform_weights_for_mega_moe(
                    (
                        self.w13_weight.data.view(torch.int8),
                        sf_to_kernel_layout(self.w13_weight_scale),
                    ),
                    (
                        self.w2_weight.data.view(torch.int8),
                        sf_to_kernel_layout(self.w2_weight_scale),
                    ),
                )
            )
            self._drop_retransformed_weights()
        else:
            self._transformed_l1, self._transformed_l2 = (
                deep_gemm.transform_weights_for_mega_moe(
                    self.w13_weight.data, self.w2_weight.data
                )
            )
            self.w13_weight = None
        # The kernel applies gamma verbatim; fold in the zero-centered gain.
        norm = self.post_expert_norm
        gamma = norm.weight.data.float()
        if norm.zero_centered_gamma:
            gamma = gamma + norm.gain_center
        self._norm_gamma = gamma.contiguous()

    def _prepare_trtllm_weights(self) -> None:
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            prepare_static_weights_for_trtllm_fp4_moe,
            reorder_w1w3_to_w3w1,
        )

        w13, sf13 = reorder_w1w3_to_w3w1(
            self.w13_weight.data, self.w13_weight_scale.data
        )
        w13, sf13, w2, sf2 = prepare_static_weights_for_trtllm_fp4_moe(
            w13,
            self.w2_weight.data,
            sf13,
            self.w2_weight_scale.data,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            True,
        )
        self.w13_weight.data = w13
        self.w13_weight_scale.data = sf13
        self.w2_weight.data = w2
        self.w2_weight_scale.data = sf2
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        # Views describe logical shapes; DeepGEMM reads the physical TRT layout.
        self._transformed_l1 = (
            w13.view(torch.int8),
            sf13.view(torch.int32).view(e, 2 * i, h // 64),
        )
        self._transformed_l2 = (
            w2.view(torch.int8),
            sf2.view(torch.int32).view(e, h, i // 64),
        )

    def _drop_retransformed_weights(self) -> None:
        """Release the checkpoint tensors the kernel no longer reads."""
        assert isinstance(self._transformed_l2, tuple)
        kept = self._transformed_l2[0].untyped_storage().data_ptr()
        if kept != self.w2_weight.untyped_storage().data_ptr():
            raise AssertionError(
                f"{self.prefix}: MegaMoE no longer returns the L2 weight "
                "aliased to w2_weight, so the drop list here is stale."
            )
        for name in ("w13_weight", "w13_weight_scale", "w2_weight_scale"):
            param = getattr(self, name)
            if param.untyped_storage().data_ptr() == kept:
                raise AssertionError(
                    f"{self.prefix}: {name} is aliased by the transformed "
                    "weights and must not be dropped."
                )
            setattr(self, name, None)

    def get_symm_buffer(self):
        """Return this dtype's view of the EP group's shared allocation.

        Layers must consume the buffer sequentially; producers on auxiliary
        streams must wait for the preceding layer before writing either view.
        """
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        buffers = self._symm_buffer_cache.get(key)
        if buffers is None:
            deep_gemm = _get_deep_gemm()
            # BF16 needs the larger layout, so it owns the shared storage.
            bf16 = deep_gemm.SymmBuffer(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="bf16xbf16",
            )
            nvfp4 = deep_gemm.SymmBuffer(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp4xfp4",
                base=bf16,
            )
            buffers = (bf16, nvfp4)
            self._symm_buffer_cache[key] = buffers
        return buffers[1 if self.is_nvfp4 else 0]

    def get_padding_mask(self, num_tokens: int) -> torch.Tensor | None:
        if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
            return None
        is_padding = get_forward_context().is_padding
        if is_padding is None:
            return None
        if is_sequence_parallel_active(
            self.is_sequence_parallel, is_padding.shape[0], self.sp_min_tokens
        ):
            return _shard_padding_mask(is_padding, num_tokens)
        return is_padding[:num_tokens]

    def input_buffer(self, num_tokens: int):
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"MuseSpark1x MegaMoE got {num_tokens} tokens, but the "
                f"symmetric buffer was sized for {self.max_num_tokens}."
            )
        assert self._transformed_l1 is not None, "finalize_weights() not called"
        return self.get_symm_buffer(), self.get_padding_mask(num_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        buffer, is_padding = self.input_buffer(num_tokens)

        if self.is_nvfp4:
            prepare_megamoe_nvfp4_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                buffer.x[:num_tokens],
                buffer.x_sf[:num_tokens].view(torch.uint8),
                buffer.topk_idx[:num_tokens],
                buffer.topk_weights[:num_tokens],
                is_padding=is_padding,
            )

        else:
            prepare_megamoe_bf16_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                buffer.x[:num_tokens],
                buffer.topk_idx[:num_tokens],
                buffer.topk_weights[:num_tokens],
                is_padding=is_padding,
            )

        return self.forward_prepared(hidden_states, buffer)

    def forward_prepared(self, hidden_states: torch.Tensor, buffer) -> torch.Tensor:
        """Consume input views after the producing streams have joined."""
        deep_gemm = _get_deep_gemm()

        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        if not self.is_nvfp4:
            deep_gemm.bf16_mega_moe(
                y,
                self._transformed_l1,
                self._transformed_l2,
                buffer,
                post_expert_norm_gamma=self._norm_gamma,
                post_expert_norm_eps=self.post_expert_norm.variance_epsilon,
            )
            return y
        deep_gemm.fp4_fp4_mega_moe(
            y=y,
            l1_weights=self._transformed_l1,
            l2_weights=self._transformed_l2,
            sym_buffer=buffer,
            l1_alphas=self.w13_weight_scale_2.data,
            l2_alphas=self.w2_weight_scale_2.data,
            a2_scales=self.w2_input_scale.data,
            post_expert_norm_gamma=self._norm_gamma,
            post_expert_norm_eps=self.post_expert_norm.variance_epsilon,
            weight_layout=self.weight_layout,
        )
        return y


MuseSparkMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
