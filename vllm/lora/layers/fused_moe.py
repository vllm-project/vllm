# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)

from .utils import _get_lora_device


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer
        self._ep_check()
        # Use the MoE-aware TP rank/size: when EP is active, FusedMoE collapses
        # moe_parallel_config.tp_size to 1 (experts are sharded across the
        # TP group instead).
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(base_layer)
        # For non-gated MoE (is_act_and_mul=False), only 1 slice is needed
        # since there's only up_proj (w1), not gate_proj + up_proj (w1 + w3)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1

        self.base_layer.ensure_moe_quant_config_init()
        if getattr(self.base_layer.quant_method, "supports_internal_mk", False):
            moe_kernel = self.base_layer.quant_method.moe_kernel
        else:
            prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
            moe_kernel = FusedMoEKernel(
                prepare_finalize,
                self.base_layer.quant_method.select_gemm_impl(
                    prepare_finalize, self.base_layer
                ),
            )
        assert moe_kernel.supports_lora(), (
            f"{type(moe_kernel.fused_experts).__name__} does not support LoRA. "
            "For unquantized MoE, set moe_backend='triton' or moe_backend='auto' "
            "(auto selects Triton automatically when LoRA is enabled). "
            "For quantized MoE, mix LoRAExpertsMixin into the experts class "
            "and consume self._lora_context in apply()."
        )
        self._moe_kernel = moe_kernel
        self.base_layer._replace_quant_method(
            FusedMoEModularMethod(self.base_layer.quant_method, moe_kernel)
        )

    def _build_lora_context(self):
        return MoELoRAContext(
            w13_lora_a_stacked=self.w13_lora_a_stacked,
            w13_lora_b_stacked=self.w13_lora_b_stacked,
            w2_lora_a_stacked=self.w2_lora_a_stacked,
            w2_lora_b_stacked=self.w2_lora_b_stacked,
            adapter_enabled=self.adapter_enabled,
            max_loras=self.max_loras,
            top_k=self.base_layer.top_k,
            w13_num_slices=self._w13_slices,
            fully_sharded=self.fully_sharded,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            local_num_experts=self.base_layer.local_num_experts,
            punica_wrapper=self.punica_wrapper,
            use_tuned_config=bool(envs.VLLM_TUNED_CONFIG_FOLDER),
        )

    def _create_lora_a_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
    ):
        self.w13_lora_a_stacked: tuple[torch.Tensor, ...] = tuple(
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    lora_config.max_lora_rank
                    if not self.fully_sharded
                    else divide(lora_config.max_lora_rank, self.tp_size),
                    self.base_layer.hidden_size,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self._w13_slices)
        )
        self.w2_lora_a_stacked: tuple[torch.Tensor, ...] = (
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    lora_config.max_lora_rank,
                    self.base_layer.intermediate_size_per_partition,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

    def _create_lora_b_weights(self, max_loras: int, lora_config: LoRAConfig):
        self.w13_lora_b_stacked: tuple[torch.Tensor, ...] = tuple(
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.intermediate_size_per_partition,
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self._w13_slices)
        )
        self.w2_lora_b_stacked: tuple[torch.Tensor, ...] = (
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.hidden_size
                    if not self.fully_sharded
                    else divide(self.base_layer.hidden_size, self.tp_size),
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

    def _ep_check(self):
        if self.base_layer.use_ep:
            moe_config = self.base_layer.moe_config
            all2all_backend = moe_config.moe_parallel_config.all2all_backend
            assert all2all_backend == "allgather_reducescatter", (
                "Fused MoE LoRA with EP currently only supports "
                f"all2all_backend='allgather_reducescatter', got '{all2all_backend}'."
            )
            assert not moe_config.moe_parallel_config.is_sequence_parallel

    def _verify_ep_fs(self, lora_config: LoRAConfig):
        # EP and fully_sharded LoRA both partition along the same TP group —
        # EP on the expert dim, fully_sharded on the LoRA rank dim — with
        # mutually contradictory assumptions about which rank holds which
        # expert's rank-shard.
        assert not (self.base_layer.use_ep and lora_config.fully_sharded_loras), (
            "Fused MoE LoRA does not support enable_expert_parallel=True "
            "together with fully_sharded_loras=True. Disable one of them."
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""

        self._verify_ep_fs(lora_config)
        self.max_loras = lora_config.max_loras
        self.fully_sharded = lora_config.fully_sharded_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)
        # They will be used by 'LoRALayerWeights.create_dummy_lora_weights'
        # to create a dummy LoRA weights.
        # TODO Optimize this section
        self.lora_a_stacked = []
        self.lora_b_stacked = []
        for lora_id in range(max_loras):
            for experts_id in range(self.base_layer.local_num_experts):
                # For gated MoE: gate_proj (w1), down_proj (w2), up_proj (w3)
                # For non-gated MoE: up_proj (w1), down_proj (w2)
                self.lora_a_stacked.append(
                    self.w13_lora_a_stacked[0][lora_id][experts_id]
                )
                self.lora_a_stacked.append(
                    self.w2_lora_a_stacked[0][lora_id][experts_id]
                )

                self.lora_b_stacked.append(
                    self.w13_lora_b_stacked[0][lora_id][experts_id]
                )
                self.lora_b_stacked.append(
                    self.w2_lora_b_stacked[0][lora_id][experts_id]
                )

                # Only add w3 (up_proj) for gated MoE (_w13_slices == 2)
                if self._w13_slices == 2:
                    self.lora_a_stacked.append(
                        self.w13_lora_a_stacked[1][lora_id][experts_id]
                    )
                    self.lora_b_stacked.append(
                        self.w13_lora_b_stacked[1][lora_id][experts_id]
                    )

    def _slice_w13_a(self, w13_lora_a: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1 or not self.fully_sharded:
            return w13_lora_a

        # w13_lora_a shape (num_experts,rank,input_size)
        current_lora_rank = w13_lora_a.shape[1]
        assert current_lora_rank % self.tp_size == 0
        # Based on S-LoRA, we slice W13/W1/W3 A along the rank dim.
        shard_size = self.w13_lora_a_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        return w13_lora_a[:, start_idx:end_idx, :]

    def _slice_w13_b(self, w13_lora_b: torch.Tensor):
        if self.tp_size == 1:
            return w13_lora_b

        # w13_lora_b shape (num_experts,output_size,rank)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w13_lora_b[:, start_idx:end_idx, :]

    def _slice_w2_a(self, w2_lora_a: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1:
            return w2_lora_a
        # w2_lora_a shape (num_experts,rank,input_size)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w2_lora_a[:, :, start_idx:end_idx]

    def _slice_w2_b(self, w2_lora_b: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1 or not self.fully_sharded:
            return w2_lora_b
        # Based on S-LoRA, we slice W2 B along the hidden_size dim.
        # w2_lora_b shape (num_experts,output_size,rank)
        shard_size = self.w2_lora_b_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w2_lora_b[:, start_idx:end_idx, :]

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        for pos in range(self._w13_slices):
            self.w13_lora_a_stacked[pos][index] = 0
            self.w13_lora_b_stacked[pos][index] = 0

        self.w2_lora_a_stacked[0][index] = 0
        self.w2_lora_b_stacked[0][index] = 0
        self.adapter_enabled[index] = 0

    #

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        # Make mypy happy
        assert isinstance(lora_a, list)
        assert isinstance(lora_b, list)

        self.reset_lora(index)
        self.adapter_enabled[index] = 1

        num_experts = self.w13_lora_a_stacked[0].shape[1]

        w1_lora_a, w2_lora_a, w3_lora_a = lora_a
        w1_lora_b, w2_lora_b, w3_lora_b = lora_b

        # EP slicing is done once at add time in
        # LoRAModelManager._slice_moe_lora_ep, so by here the cached
        # tensors already match the local-expert dim of the stacked buffers.
        assert (
            num_experts
            == w1_lora_a.shape[0]
            == w2_lora_a.shape[0]
            == w3_lora_a.shape[0]
        )

        slliced_w1_lora_a = self._slice_w13_a(w1_lora_a)
        slliced_w1_lora_b = self._slice_w13_b(w1_lora_b)

        sliced_w2_lora_a = self._slice_w2_a(w2_lora_a)
        sliced_w2_lora_b = self._slice_w2_b(w2_lora_b)

        self.w13_lora_a_stacked[0][
            index, :, : slliced_w1_lora_a.shape[1], : slliced_w1_lora_a.shape[2]
        ].copy_(slliced_w1_lora_a, non_blocking=True)

        self.w13_lora_b_stacked[0][
            index, :, : slliced_w1_lora_b.shape[1], : slliced_w1_lora_b.shape[2]
        ].copy_(slliced_w1_lora_b, non_blocking=True)

        # Only copy w3 (up_proj) for gated MoE (_w13_slices == 2)
        if self._w13_slices == 2:
            slliced_w3_lora_a = self._slice_w13_a(w3_lora_a)
            slliced_w3_lora_b = self._slice_w13_b(w3_lora_b)

            self.w13_lora_a_stacked[1][
                index, :, : slliced_w3_lora_a.shape[1], : slliced_w3_lora_a.shape[2]
            ].copy_(slliced_w3_lora_a, non_blocking=True)

            self.w13_lora_b_stacked[1][
                index, :, : slliced_w3_lora_b.shape[1], : slliced_w3_lora_b.shape[2]
            ].copy_(slliced_w3_lora_b, non_blocking=True)

        self.w2_lora_a_stacked[0][
            index, :, : sliced_w2_lora_a.shape[1], : sliced_w2_lora_a.shape[2]
        ].copy_(sliced_w2_lora_a, non_blocking=True)

        self.w2_lora_b_stacked[0][
            index, :, : sliced_w2_lora_b.shape[1], : sliced_w2_lora_b.shape[2]
        ].copy_(sliced_w2_lora_b, non_blocking=True)

    def set_mapping(self, punica_wrapper):
        super().set_mapping(punica_wrapper)
        lora_context = self._build_lora_context()
        self._moe_kernel.fused_experts.set_lora_context(lora_context)
        prepare_finalize = self._moe_kernel.prepare_finalize
        if hasattr(prepare_finalize, "set_lora_context"):
            prepare_finalize.set_lora_context(lora_context)

    def forward(self, *args, **kwargs):
        return self.base_layer.forward(*args, **kwargs)

    @property
    def quant_method(self):
        return self.base_layer.quant_method

    @property
    def runner(self):
        return self.base_layer.runner

    @property
    def is_internal_router(self) -> bool:
        return self.base_layer.is_internal_router

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""

        # source_layer is FusedMoE
        moe_cls = maybe_get_oot_by_class(FusedMoE)
        return isinstance(source_layer, moe_cls) and len(packed_modules_list) == 2


class FusedMoE3DWithLoRA(FusedMoEWithLoRA):
    def __init__(self, base_layer):
        super().__init__(base_layer)
        self._w13_slices = 1

    def _create_lora_b_weights(self, max_loras, lora_config):
        self.w13_lora_b_stacked: tuple[torch.Tensor] = tuple(
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.intermediate_size_per_partition * 2,
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self._w13_slices)
        )
        self.w2_lora_b_stacked: tuple[torch.Tensor] = (
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.hidden_size
                    if not self.fully_sharded
                    else divide(self.base_layer.hidden_size, self.tp_size),
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""

        assert isinstance(model_config, PretrainedConfig)
        self._verify_ep_fs(lora_config)
        self._base_model = model_config.architectures[0]
        self.max_loras = lora_config.max_loras
        self.fully_sharded = lora_config.fully_sharded_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)

    def _slice_w13_b(self, w13_lora_b: torch.Tensor):
        if self.tp_size == 1:
            return w13_lora_b

        # w13_lora_b shape (num_experts,output_size,rank)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        # HACK: Currently, only GPT-OSS is in interleaved order
        if self._base_model == "GptOssForCausalLM":
            # For models like GPT-OSS, the weights of w1 (gate_proj) and w3 (up_proj)
            # in the interleaved order, and corresponding LoRA need to be processed.
            w1_lora_b = w13_lora_b[:, ::2, :]
            w3_lora_b = w13_lora_b[:, 1::2, :]
            sliced_w1_lora_b = w1_lora_b[:, start_idx:end_idx, :]
            sliced_w3_lora_b = w3_lora_b[:, start_idx:end_idx, :]

            return torch.stack([sliced_w1_lora_b, sliced_w3_lora_b], dim=2).flatten(
                1, 2
            )
        else:
            slice_size = w13_lora_b.shape[1] // 2
            w1_lora_b = w13_lora_b[:, :slice_size, :]
            w3_lora_b = w13_lora_b[:, slice_size:, :]
            sliced_w1_lora_b = w1_lora_b[:, start_idx:end_idx, :]
            sliced_w3_lora_b = w3_lora_b[:, start_idx:end_idx, :]

            return torch.cat([sliced_w1_lora_b, sliced_w3_lora_b], dim=1)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        # Make mypy happy
        assert isinstance(lora_a, list)
        assert isinstance(lora_b, list)
        assert len(lora_a) == len(lora_b) == 2

        self.reset_lora(index)
        self.adapter_enabled[index] = 1

        w13_lora_a, w2_lora_a = lora_a
        w13_lora_b, w2_lora_b = lora_b

        sliced_w13_lora_a = self._slice_w13_a(w13_lora_a)
        sliced_w13_lora_b = self._slice_w13_b(w13_lora_b)

        sliced_w2_lora_a = self._slice_w2_a(w2_lora_a)
        sliced_w2_lora_b = self._slice_w2_b(w2_lora_b)

        self.w13_lora_a_stacked[0][
            index, :, : sliced_w13_lora_a.shape[1], : sliced_w13_lora_a.shape[2]
        ].copy_(sliced_w13_lora_a, non_blocking=True)
        self.w2_lora_a_stacked[0][
            index, :, : sliced_w2_lora_a.shape[1], : sliced_w2_lora_a.shape[2]
        ].copy_(sliced_w2_lora_a, non_blocking=True)

        self.w13_lora_b_stacked[0][
            index, :, : sliced_w13_lora_b.shape[1], : sliced_w13_lora_b.shape[2]
        ].copy_(sliced_w13_lora_b, non_blocking=True)
        self.w2_lora_b_stacked[0][
            index, :, : sliced_w2_lora_b.shape[1], : sliced_w2_lora_b.shape[2]
        ].copy_(sliced_w2_lora_b, non_blocking=True)

    @property
    def w13_input_size(self):
        """
        Full size
        """
        return self.w13_lora_a_stacked[0].shape[-1]

    @property
    def w13_output_size(self):
        """
        Full size
        """
        return self.w13_lora_b_stacked[0].shape[-2] * self.tp_size

    @property
    def w2_input_size(self):
        """
        Full size
        """
        return self.w2_lora_a_stacked[0].shape[-1] * self.tp_size

    @property
    def w2_output_size(self):
        """
        Full size
        """
        return self.base_layer.hidden_size

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        # source_layer is FusedMoE
        moe_cls = maybe_get_oot_by_class(FusedMoE)
        return isinstance(source_layer, moe_cls) and len(packed_modules_list) == 1
