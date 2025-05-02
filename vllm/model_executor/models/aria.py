# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Mapping, Sequence
from typing import List, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from transformers import AriaConfig, AriaTextConfig, BatchFeature
from transformers.models.aria.modeling_aria import AriaCrossAttention
from transformers.models.aria.processing_aria import AriaProcessor

from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

# yapf: disable
from .idefics2_vision_model import Idefics2VisionConfig
from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics3VisionTransformer)
# yapf: enable
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsQuant
from .llama import LlamaDecoderLayer, LlamaMLP, LlamaModel
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    is_pp_missing_parameter, maybe_prefix,
                    merge_multimodal_embeddings)


class AriaImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    pixel_mask: Optional[torch.Tensor]
    """
    Shape:
        pixel_values: `(batch_size * num_images, num_channels, height, width)`
        pixel_mask: `(batch_size * num_images, height, width)`
    """


class AriaVisionTransformer(Idefics3VisionTransformer, SupportsQuant):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # Unlike Idefics3VisionTransformer which uses LayerNorm after the
        # final layer, Aria omits this normalization, so we replace it with an
        # Identity layer
        self.post_layernorm = nn.Identity()

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            # NOTE: post_layernorm is not used in Aria
            if "post_layernorm" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class AriaProjectorMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.linear_in = ColumnParallelLinear(in_features,
                                              hidden_features,
                                              bias=False)
        self.linear_out = RowParallelLinear(hidden_features,
                                            output_dim,
                                            bias=False)
        self.act = get_act_fn("gelu_new")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_out(hidden_states)
        return hidden_states


class AriaProjector(nn.Module):
    """
    A projection module with one cross attention layer and one FFN layer, which
    projects ViT's outputs into MoE's inputs.

    Args:
        patch_to_query_dict (dict): Maps patch numbers to their corresponding
        query numbers,
            e.g., {1225: 128, 4900: 256}. This allows for different query sizes
            based on image resolution.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        kv_dim (int): Dimension of key and value.
        ff_dim (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
        norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.

    Outputs:
        A tensor with the shape of (batch_size, query_number, output_dim)
    """

    def __init__(self, config: AriaConfig) -> None:
        super().__init__()

        self.patch_to_query_dict = config.projector_patch_to_query_dict
        self.in_features = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_attention_heads
        self.kv_dim = config.vision_config.hidden_size
        self.hidden_features = config.text_config.hidden_size
        self.output_dim = config.text_config.hidden_size

        self.query = nn.Parameter(
            torch.empty(config.max_value_projector_patch_to_query_dict,
                        self.in_features))

        self.cross_attn = AriaCrossAttention(config)

        self.layer_norm = nn.LayerNorm(self.in_features)
        self.feed_forward = AriaProjectorMLP(self.in_features,
                                             self.hidden_features,
                                             self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_patches = x.shape[0], x.shape[1]

        if num_patches not in self.patch_to_query_dict:
            raise KeyError(f"Number of patches {num_patches} not found in "
                           "patch_to_query_dict amongst possible values "
                           f"{self.patch_to_query_dict.keys()}.")

        query_num = self.patch_to_query_dict[num_patches]

        queries = self.query[:query_num].unsqueeze(0).repeat(batch_size, 1, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)

        out = self.feed_forward(self.layer_norm(attention_out))

        return out


class AriaFusedMoE(FusedMoE):

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      shard_id: str) -> None:
        # Override the weight_loader to handle the expert weights in the Aria
        # model, which are already packed with experts, and merge the gate and
        # up weights for each expert.
        # Note: Loading expert weights with quantization is not supported
        tp_rank = get_tensor_model_parallel_rank()
        if shard_id == 'w13':
            # the shape of loaded_weight is
            # (num_experts, hidden_size, 2 * moe_intermediate_size)
            if self.tp_size > 1:
                up, gate = loaded_weight.chunk(2, dim=-1)
                up_current_rank = up.chunk(self.tp_size, dim=-1)[tp_rank]
                gate_current_rank = gate.chunk(self.tp_size, dim=-1)[tp_rank]
                up_and_gate = torch.cat([up_current_rank, gate_current_rank],
                                        dim=-1).transpose(1, 2)
                param.data.copy_(up_and_gate)
            else:
                param.data.copy_(loaded_weight.transpose(1, 2))
        elif shard_id == 'w2':
            # the shape of loaded_weight is
            # (num_experts, moe_intermediate_size, hidden_size)
            if self.tp_size > 1:
                down_current_rank = loaded_weight.chunk(self.tp_size,
                                                        dim=1)[tp_rank]
                param.data.copy_(down_current_rank.transpose(1, 2))
            else:
                param.data.copy_(loaded_weight.transpose(1, 2))


class AriaTextMoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer for the AriaMoE model.

    This layer implements the MoE mechanism, which routes input tokens to
    different experts based on a routing algorithm, processes them through the
    experts, and then combines the outputs.
    """

    def __init__(
        self,
        config: AriaTextConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.router_weight = nn.Parameter(
            torch.empty(
                (self.config.moe_num_experts, self.config.hidden_size)))

        self.experts = AriaFusedMoE(
            num_experts=config.moe_num_experts,
            top_k=config.moe_topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = LlamaMLP(
            config.hidden_size,
            config.intermediate_size * config.moe_num_shared_experts,
            "silu",
            quant_config=quant_config,
            bias=config.mlp_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size,
            sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.
        """

        router_output = torch.nn.functional.linear(hidden_states,
                                                   self.router_weight)

        hidden_states_copy = hidden_states.clone()
        # NOTE: hidden_states will be modified inplace by `FusedMoE`
        sparse_expert_output = self.experts(hidden_states, router_output)
        shared_expert_output = self.shared_experts(hidden_states_copy)

        return sparse_expert_output + shared_expert_output


class AriaTextDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the AriaMoE model which modifies the standard
    `LlamaDecoderLayer` by replacing the traditional MLP with a Mixture of
    Experts (MoE) Layer.
    """

    def __init__(
        self,
        config: AriaTextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix)
        self.mlp = AriaTextMoELayer(config,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.mlp")


class AriaTextModel(LlamaModel, SupportsQuant):
    """
    Custom LlamaModel for the AriaMoE model which modifies the standard
    LlamaModel by replacing the `LlamaDecoderLayer` with `MoEDecoderLayer`.
    """
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts.w13_weight": ["experts.fc1.weight"],
        "experts.w2_weight": ["experts.fc2.weight"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=AriaTextDecoderLayer)

    # Adapted from LlamaModel.load_weights with the modification of adding
    # the expert weights mapping to `stacked_params_mapping`
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            ("experts.w13_weight", "experts.fc1.weight", 'w13'),
            ("experts.w2_weight", "experts.fc2.weight", 'w2'),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class AriaProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(AriaConfig)

    def get_vision_config(self):
        return self.get_hf_config().vision_config

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(AriaProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        return max(hf_config.projector_patch_to_query_dict.values())


class AriaDummyInputsBuilder(BaseDummyInputsBuilder[AriaProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token: str = processor.tokenizer.image_token  # type: ignore

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        vision_config = self.info.get_vision_config()

        max_image_size = vision_config.image_size
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=max_image_size,
                                   height=max_image_size,
                                   num_images=num_images)
        }


class AriaMultiModalProcessor(BaseMultiModalProcessor[AriaProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_mask=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        num_image_tokens = self.info.get_num_image_tokens()

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=[image_token_id] * num_image_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(AriaMultiModalProcessor,
                                        info=AriaProcessingInfo,
                                        dummy_inputs=AriaDummyInputsBuilder)
class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language
    model to perform tasks that involve both image and text inputs.
    """
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model": "language_model",
            "language_model.lm_head": "lm_head",
        },
        orig_to_new_suffix={
            "router.weight": "router_weight",
        },
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vision_tower = AriaVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_tower",
        )
        self.multi_modal_projector = AriaProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AriaTextModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model.model"),
        )
        self.pad_token_id = (self.config.pad_token_id
                             if self.config.pad_token_id is not None else -1)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size,
            quant_config=quant_config,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.vocab_size, logit_scale)

    def _validate_image_sizes(
            self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        if not all(img.shape == images[0].shape for img in images):
            raise ValueError("All images must be the same size")
        return images

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[AriaImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_mask = kwargs.pop("pixel_mask", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        pixel_values = self._validate_image_sizes(pixel_values)
        pixel_values = flatten_bn(pixel_values, concat=True)

        if pixel_mask is not None:
            if not isinstance(pixel_mask, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel mask. "
                                 f"Got type: {type(pixel_mask)}")

            pixel_mask = flatten_bn(pixel_mask, concat=True)

        return AriaImagePixelInputs(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

    def _create_patch_attention_mask(
            self, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pixel_mask is None:
            return None

        patches_subgrid = pixel_mask.unfold(
            dimension=1,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        ).unfold(
            dimension=2,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        )
        return (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    def _process_image_input(
        self, image_input: AriaImagePixelInputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.vision_tower is not None

        pixel_values = image_input['pixel_values']
        pixel_mask = image_input['pixel_mask']

        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)

        image_outputs = self.vision_tower(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        image_attn_mask = None
        if patch_attention_mask is not None:
            flattened_mask = patch_attention_mask.flatten(1)
            image_attn_mask = torch.logical_not(flattened_mask)

        return self.multi_modal_projector(image_outputs, image_attn_mask)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        multimodal_embeddings = self._process_image_input(image_input)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
