# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XPU Qwen3.5 implementation adapted from the NVIDIA eager model."""

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, ClassVar, cast

import torch
from torch import nn
from transformers import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)

from vllm.config import MultiModalConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsFp8
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    mrope_positions_factor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

if TYPE_CHECKING:
    from vllm_xpu_kernels.moe_shared_fused_interface import XpuMoESharedFusedDecode

logger = init_logger(__name__)


class _XpuMoESharedFusion:
    """Prepare native weights once; refresh derived copies without recapture."""

    def __init__(self, block: "Qwen3_5SparseMoeBlock"):
        self.plan: XpuMoESharedFusedDecode | None = None
        self._weights = self._get_weights(block)
        self.dtype: torch.dtype | None = None
        self.hidden_dim = self.top_k = 0
        if self._weights is not None:
            config = cast(MoERunner, block.experts).moe_config
            self.dtype = config.in_dtype
            self.hidden_dim = config.hidden_dim
            self.top_k = config.experts_per_token

    @staticmethod
    def _get_weights(
        block: "Qwen3_5SparseMoeBlock",
    ) -> tuple[torch.Tensor, ...] | None:
        runner, shared = block.experts, block.shared_expert
        if not isinstance(runner, MoERunner) or shared is None:
            return None
        config = runner.moe_config
        parallel = config.moe_parallel_config
        if (
            block.is_sequence_parallel
            or block.replicate_shared_expert
            or block.is_fused_shared_expert_enabled
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner.enable_dbo
            or isinstance(runner.router, ZeroExpertRouter)
            or runner.activation != MoEActivation.SILU
            or runner.routed_scaling_factor != 1.0
            or runner.routed_experts.apply_router_weight_on_input
            or config.has_bias
            or config.is_lora_enabled
            or parallel.enable_eplb
            or parallel.use_ep
            or parallel.ep_size != 1
            or runner.expert_map is not None
            or parallel.dp_size != 1
            or parallel.pcp_size != 1
            or parallel.is_sequence_parallel
            or not isinstance(shared.act_fn, SiluAndMul)
        ):
            return None
        if any(
            layer.bias is not None
            for layer in (
                shared.gate_up_proj,
                shared.down_proj,
                block.shared_expert_gate,
            )
        ):
            return None
        kernel = runner.routed_experts.quant_method.moe_kernel
        experts = kernel.fused_experts if kernel is not None else None
        if not isinstance(experts, XPUExpertsFp8):
            return None
        gu, down, gate = (
            shared.gate_up_proj,
            shared.down_proj,
            block.shared_expert_gate,
        )
        s13, s2 = experts.w1_scale, experts.w2_scale
        shared_s13 = getattr(gu, "weight_scale", None)
        shared_s2 = getattr(down, "weight_scale", None)
        if s13 is None or s2 is None or shared_s13 is None or shared_s2 is None:
            return None
        w13, w2 = runner.routed_experts.w13_weight, runner.routed_experts.w2_weight
        if not (
            w13.dtype
            == w2.dtype
            == gu.weight.dtype
            == down.weight.dtype
            == torch.float8_e4m3fn
            and s13.shape == s2.shape == (w13.shape[0],)
            and shared_s13.numel() == shared_s2.numel() == 1
            and gate.weight.dtype == runner.moe_config.in_dtype
            and gu.weight.shape == w13.shape[1:]
            and down.weight.shape == w2.shape[1:]
            and w2.shape[0] == w13.shape[0]
            and w13.shape[2] == 2 * w2.shape[1]
            and w2.shape[2] == w13.shape[1]
            and 0 < runner.moe_config.experts_per_token <= w13.shape[0] <= 512
            and gate.weight.numel() == w13.shape[1]
        ):
            return None
        matrices = (w13, w2, gu.weight, down.weight)
        scales = (s13, s2, shared_s13, shared_s2)
        weights = (*matrices, *scales, gate.weight)
        if not all(t.device == w13.device for t in weights):
            return None
        # The provider copies transposed shared linears once, before capture.
        if any(t.is_contiguous() and t.data_ptr() % 64 for t in matrices):
            return None
        if (gate.weight.is_contiguous() and gate.weight.data_ptr() % 32) or any(
            t.dtype != torch.float32 for t in scales
        ):
            return None
        return weights

    def get_plan(self, x: torch.Tensor) -> "XpuMoESharedFusedDecode | None":
        if (
            self._weights is None
            or x.device.type != "xpu"
            or x.ndim != 2
            or not 1 <= x.shape[0] <= 8
            or x.shape[1] != self.hidden_dim
            or x.dtype != self.dtype
            or not x.is_contiguous()
            or x.data_ptr() % 64
        ):
            return None
        try:
            from vllm_xpu_kernels.moe_shared_fused_interface import (
                XpuMoESharedFusedDecode,
                supports,
            )
        except ImportError:
            return None
        if not supports(
            x.dtype,
            self._weights[0].dtype,
            x.shape[0],
            self.top_k,
            self.hidden_dim,
            self._weights[1].shape[1],
        ):
            return None
        if self.plan is None:
            w13, w2, shared_w13, shared_w2, s13, s2, shared_s13, shared_s2, gate = (
                self._weights
            )
            self.plan = XpuMoESharedFusedDecode(
                w13=w13,
                w13_scales=s13,
                w2=w2,
                w2_scales=s2,
                shared_w13=shared_w13,
                shared_w13_scale=shared_s13,
                shared_w2=shared_w2,
                shared_w2_scale=shared_s2,
                shared_gate=gate,
                top_k=self.top_k,
            )
        return self.plan

    @torch.no_grad()
    def refresh(self, block: "Qwen3_5SparseMoeBlock") -> None:
        weights = self._get_weights(block)
        plan = self.plan
        assert plan is not None
        if weights is None:
            raise RuntimeError(
                "Reload changed fused MoE support; recapture is required"
            )
        targets = (
            plan.w13,
            plan.w2,
            plan.shared_w13,
            plan.shared_w2,
            plan.w13_scales,
            plan.w2_scales,
            plan.shared_w13_scale,
            plan.shared_w2_scale,
            plan.shared_gate,
        )
        for index, (source, target) in enumerate(zip(weights, targets)):
            if (
                source.numel() != target.numel()
                or source.dtype != target.dtype
                or source.device != target.device
                or (index < 4 and source.shape != target.shape)
            ):
                raise RuntimeError(
                    "Reload changed fused MoE weight metadata; recapture is required"
                )
        for source, target in zip(weights, targets):
            if source.data_ptr() != target.data_ptr():
                target.copy_(source.reshape(target.shape))
        self._weights = weights


class Qwen3_5SparseMoeBlock(Qwen3NextSparseMoeBlock):
    _fusion: _XpuMoESharedFusion | None = None

    @property
    def _xpu_moe_shared_fused_plan(self) -> "XpuMoESharedFusedDecode | None":
        return self._fusion.plan if self._fusion is not None else None

    def refresh_fused_weights(self) -> None:
        """Called after weight processing, before any captured graph replays."""
        if self._fusion is not None and self._fusion.plan is not None:
            self._fusion.refresh(self)
        else:
            self._fusion = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if already_sequence_parallel:
            return super().forward(hidden_states, already_sequence_parallel)
        if self._fusion is None:
            self._fusion = _XpuMoESharedFusion(self)
        plan = self._fusion.get_plan(hidden_states)
        if plan is None:
            return super().forward(hidden_states, already_sequence_parallel)
        runner = cast(MoERunner, self.experts)
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = runner.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=runner.routed_experts.quant_method.topk_indices_dtype,
            input_ids=None,
        )
        output = plan.forward(
            torch.empty_like(hidden_states),
            hidden_states,
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.contiguous(),
        )
        # Both routed and shared outputs are rank-local partials.
        if self.tp_size > 1 and not runner.moe_config.skip_final_all_reduce:
            output = tensor_model_parallel_all_reduce(output)
        return output


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
        self.use_fused_allreduce_norm = False
        # Whether this layer's MLP/MoE leaves an un-reduced partial, i.e.
        # whether the *next* norm (or the final norm) must do the all-reduce.
        self.mlp_output_is_partial = self.use_fused_allreduce_norm
        reduce_results = not (
            self.use_attn_reduce_scatter_for_moe or self.use_fused_allreduce_norm
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=reduce_results,
            )
            self.linear_attn._fused_gdn_gated_rms_norm = None
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=reduce_results,
                mrope_positions_factor=mrope_positions_factor(vllm_config),
            )
            self.self_attn.use_fused_output_gate = False
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3_5SparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.use_fused_allreduce_norm,
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.use_fused_allreduce_norm,
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


class Qwen3_5Model(nn.Module, EagleModelMixin):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        # Mirror of the decoder layers' flag (uniform across layers): when
        # True, every layer returns its MLP/MoE output as an un-reduced TP
        # partial, so the final norm and eagle aux captures must all-reduce.
        self.mlp_output_is_partial = getattr(
            self.layers[self.start_layer], "mlp_output_is_partial", False
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def use_sequence_parallel(self) -> bool:
        return self.layers[self.start_layer].use_attn_reduce_scatter_for_moe

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if residual is None and layer_idx in self.aux_hidden_state_layers:
            # The layer-0 capture aliases the embeddings, which become the
            # first residual and are mutated in place by the next layer's
            # fused-add norm; capture by value.
            hidden_states = hidden_states.clone()
        if (
            residual is not None
            and self.mlp_output_is_partial
            and layer_idx in self.aux_hidden_state_layers
        ):
            # hidden_states is an un-reduced TP partial; the aux capture needs
            # the complete hidden state.
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return super()._maybe_add_hidden_state(
            aux_hidden_states, layer_idx, hidden_states, residual
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        if self.use_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Maps PEFT embed/lm_head LoRA targets onto vLLM embedding wrappers.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Some community text-only checkpoints keep the extraneous
    # `model.language_model.` prefix inherited from the VL training stack.
    # Strip it so both prefixed and clean checkpoints load correctly.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    supports_multimodal_pruning = True

    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_prefix={"mtp.": None})
    )

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = cast(
            MultiModalConfig, vllm_config.model_config.multimodal_config
        )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        With mrope enabled (default for Qwen3VL-class models), ``positions``
        has shape `(3, seq_len)`, otherwise `(seq_len,)`.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    language_model: Qwen3_5MoeForCausalLM
    num_local_physical_experts: int

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: ClassVar[bool] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = cast(
            MultiModalConfig, vllm_config.model_config.multimodal_config
        )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
