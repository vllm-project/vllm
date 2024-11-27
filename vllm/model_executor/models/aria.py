import math
from typing import Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.inputs import INPUT_REGISTRY, token_inputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.sampler import (Sampler, SamplerOutput,
                                                SamplingMetadata)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.idefics2_vision_model import (
    Idefics2VisionTransformer)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (LlamaDecoderLayer, LlamaMLP,
                                              LlamaModel)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              is_pp_missing_parameter,
                                              maybe_prefix,
                                              merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.aria import (AriaMoELMConfig,
                                                  AriaVisionConfig)

from .utils import flatten_bn


class AriaImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    pixel_mask: Optional[torch.Tensor]
    """
    Shape: 
        pixel_values: `(batch_size * num_images, num_channels, height, width)`
        pixel_mask: `(batch_size * num_images, height, width)`
    """


class AriaVisionTransformer(Idefics2VisionTransformer):
    """
    AriaVisionTransformer is a modified version of Idefics2VisionTransformer
    that replaces the post-layernorm with an identity layer.
    """

    def __init__(
        self,
        config: AriaVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.post_layernorm = nn.Identity()


class AriaVisionModel(nn.Module):
    config_class = AriaVisionConfig

    def __init__(
        self,
        config: AriaVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.vision_model = AriaVisionTransformer(
            config,
            quant_config,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)

        vit_oup = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        image_atts = self._create_image_attention_mask(patch_attention_mask)

        return vit_oup, image_atts

    def _create_patch_attention_mask(self, pixel_mask):
        if pixel_mask is None:
            return None

        patches_subgrid = pixel_mask.unfold(
            dimension=1,
            size=self.vision_model.config.patch_size,
            step=self.vision_model.config.patch_size,
        ).unfold(
            dimension=2,
            size=self.vision_model.config.patch_size,
            step=self.vision_model.config.patch_size,
        )
        return (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    def _create_image_attention_mask(self, patch_attention_mask):
        if patch_attention_mask is None:
            return None

        flattened_mask = patch_attention_mask.flatten(1)
        return torch.logical_not(flattened_mask)


class FFN(nn.Module):

    def __init__(self, embed_dim, ff_dim, output_dim):
        super().__init__()
        self.linear_in = ColumnParallelLinear(embed_dim, ff_dim, bias=False)
        self.linear_out = RowParallelLinear(ff_dim, output_dim, bias=False)
        self.act = get_act_fn("gelu_new")

    def forward(self, hidden_states):
        hidden_states, _ = self.linear_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_out(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):

    def __init__(self, kv_dim, embed_dim, num_heads, drop_out_rate=0):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, embed_dim, bias=False)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_out_rate)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)

    def forward(self, x, hidden_states, attn_mask=None, add_residual=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        query = self.q_proj(normed_hidden_states).permute(1, 0, 2)

        x = self.ln_kv(x)
        key = self.k_proj(x).permute(1, 0, 2)
        value = self.v_proj(x).permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(query,
                                             key,
                                             value,
                                             attn_mask=attn_mask)

        attn_output = attn_output.permute(1, 0, 2)

        if add_residual:
            attn_output = hidden_states + self.dropout(
                self.linear(attn_output))
        else:
            attn_output = self.dropout(self.linear(attn_output))

        return attn_output


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

    def __init__(
        self,
        patch_to_query_dict,
        embed_dim,
        num_heads,
        kv_dim,
        ff_dim,
        output_dim,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.patch_to_query_dict = patch_to_query_dict
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(
            torch.zeros(max(patch_to_query_dict.values()), self.embed_dim))

        trunc_normal_(self.query, std=0.02)

        self.cross_attn = CrossAttention(kv_dim, embed_dim, num_heads)

        self.ln_ffn = norm_layer(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, output_dim)

    def forward(self, x, attn_mask=None):
        bs = x.shape[0]
        queries = self.query.unsqueeze(0).repeat(bs, 1, 1)

        query_num = self.patch_to_query_dict.get(x.shape[1], None)
        assert (query_num is not None
                ), f"Query number for {x.shape[1]} patches is not provided"

        queries = queries[:, :query_num, :]

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)

        out = self.ffn(self.ln_ffn(attention_out))

        return out


class AriaFusedMoE(FusedMoE):

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      shard_id: str) -> Set[str]:
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


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer for the AriaMoE model.

    This layer implements the MoE mechanism, which routes input tokens to
    different experts based on a routing algorithm, processes them through the
    experts, and then combines the outputs.
    """

    def __init__(
        self,
        config: AriaMoELMConfig,
        quant_config: Optional[QuantizationConfig],
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
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            reduce_results=True,
        )
        self.shared_experts = LlamaMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.moe_num_shared_experts,
            "silu",
            quant_config=quant_config,
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

        shared_expert_output = self.shared_experts(hidden_states)
        sparse_expert_output = self.experts(hidden_states, router_output)

        return sparse_expert_output + shared_expert_output


class MoEDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the AriaMoE model which modifies the standard
    `LlamaDecoderLayer` by replacing the traditional MLP with a Mixture of
    Experts (MoE) Layer.
    """

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix)
        self.mlp = MoELayer(config, quant_config=quant_config)


class AriaMoELMModel(LlamaModel):
    """
    Custom LlamaModel for the AriaMoE model which modifies the standard
    LlamaModel by replacing the `LlamaDecoderLayer` with `MoEDecoderLayer`.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=MoEDecoderLayer)

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
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
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


def build_mm_projector(config):
    return AriaProjector(
        patch_to_query_dict=config.projector_patch_to_query_dict,
        embed_dim=config.vision_config.hidden_size,
        num_heads=config.vision_config.num_attention_heads,
        kv_dim=config.vision_config.hidden_size,
        ff_dim=config.text_config.hidden_size,
        output_dim=config.text_config.hidden_size,
    )


def get_max_multimodal_tokens(ctx):
    return max(ctx.model_config.hf_config.image_size2tokens.values())


def input_mapper_for_aria(ctx, data):
    return MultiModalInputs(data)


def input_processor(ctx, llm_inputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    # if it is pure text input, use it as is
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    image_processor = cached_get_image_processor(
        model_config.model, trust_remote_code=model_config.trust_remote_code)
    hf_config = model_config.hf_config

    # prepare image tokens, the max_image_size is used to determine the number
    # of patch_size for every image
    max_image_size = multi_modal_data.pop("max_image_size", 980)
    _split_image = multi_modal_data.pop("split_image", False)

    assert isinstance(max_image_size,
                      (int, float)), "max_image_size should be float or int"
    images = (multi_modal_data["image"] if isinstance(
        multi_modal_data["image"], list) else [multi_modal_data["image"]])

    image_inputs = image_processor.preprocess(images,
                                              max_image_size=max_image_size,
                                              split_image=_split_image,
                                              return_tensors="pt").data
    image_inputs['pixel_values'] = image_inputs['pixel_values'].to(
        ctx.model_config.dtype)
    num_crops = image_inputs.pop("num_crops")

    prompt_token_ids = llm_inputs["prompt_token_ids"]
    if num_crops.sum().item() > 0:
        _, prompt_token_ids, _ = repeat_and_pad_placeholder_tokens(
            tokenizer,
            None,
            prompt_token_ids,
            placeholder_token_id=hf_config.image_token_index,
            repeat_count=num_crops,
        )

    repeat_count = [hf_config.image_size2tokens[max_image_size]
                    ] * sum(num_crops).item()
    new_prompt, new_token_ids, _ = repeat_and_pad_placeholder_tokens(
        tokenizer,
        None,
        prompt_token_ids,
        placeholder_token_id=hf_config.image_token_index,
        repeat_count=repeat_count,
    )

    return token_inputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data={"image": image_inputs},
    )


@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_multimodal_tokens)
@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_aria)
@INPUT_REGISTRY.register_input_processor(input_processor)
class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language
    model to perform tasks that involve both image and text inputs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # prepare the image_size to tokens mapping for the image preprocess, see
        # input_processor
        config.image_size2tokens = {
            int(math.sqrt(k) * config.vision_config.patch_size): v
            for k, v in config.projector_patch_to_query_dict.items()
        }
        self.config = config
        self.vision_tower = AriaVisionModel(config.vision_config)
        self.multi_modal_projector = build_mm_projector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AriaMoELMModel(
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
        self.sampler = Sampler()

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
            pixel_mask = flatten_bn(pixel_mask, concat=True)

        return AriaImagePixelInputs(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

    def _process_image_input(
        self, image_input: AriaImagePixelInputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.vision_tower is not None

        pixel_values = image_input['pixel_values']
        pixel_mask = image_input['pixel_mask']

        image_feature, image_attn_mask = self.vision_tower(
            pixel_values, pixel_mask=pixel_mask)
        return self.multi_modal_projector(image_feature, image_attn_mask)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        multimodal_embeddings = self._process_image_input(image_input)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
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
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
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
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "language_model.model": "language_model",
                "language_model.lm_head": "lm_head",
            },
            orig_to_new_suffix={
                "router.weight": "router_weight",
            },
        )

        loader = AutoWeightsLoader(self)
        loader.load_weights(weights, mapper=hf_to_vllm_mapper)
