# Adapted from
# https://github.com/THUDM/CogAgent
"""Inference-only CogAgent model compatible with THUDM weights."""
from argparse import Namespace
from array import array
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple,
                    TypedDict,Union)

import torch
from PIL import Image
from torch import nn
from torch.nn import LayerNorm

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4_vision_encoder import EVA2CLIPModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ModalityData, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, BatchFeature,
                                        MultiModalFieldConfig, ProcessorMixin,
                                        PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors,
                           SequenceData)
from vllm.transformers_utils.configs import ChatGLMConfig

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


def calculate_image_placeholder(vision_config):
    return (vision_config["image_size"] // vision_config["patch_size"] // 2)**2


def mm_input_mapper_for_glmv(
    ctx: InputContext,
    data: ModalityData[object],
) -> Dict:
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)
    if tokenizer is None:
        raise RuntimeError("No HuggingFace processor is available "
                           "to process the image object")
    try:
        raw_batch_data = tokenizer.apply_chat_template(
            conversation=[{
                "role": "user",
                "image": data
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True).data
    except Exception:
        logger.error("Failed to process image (%s)", data)
        raise
    pixel_values = raw_batch_data['images']

    return MultiModalKwargs({'pixel_values': pixel_values})


def merge_glm_vision_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    vision_embeddings: torch.Tensor,
    boi_token_id: int,
    eoi_token_id: int,
) -> torch.Tensor:

    boi_positions = (input_ids == boi_token_id).nonzero(as_tuple=True)[0]
    eoi_positions = (input_ids == eoi_token_id).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for boi_pos, eoi_pos in zip(boi_positions, eoi_positions):
        assert boi_pos < eoi_pos
        mask[boi_pos:eoi_pos + 1] = True
    inputs_embeds[mask] = vision_embeddings.view(-1,
                                                 vision_embeddings.shape[-1])
    return inputs_embeds


class GLMImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""


def get_max_glmv_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(ChatGLMConfig)

    vision_config = getattr(hf_config, 'vision_config', None)
    if vision_config is None:
        return 1
    elif isinstance(vision_config, dict):
        return calculate_image_placeholder(vision_config)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def dummy_data_for_glmv(ctx: InputContext, seq_len: int,
                        mm_counts: Mapping[str, int]) -> DummyData:
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = getattr(hf_config, 'vision_config', None)

    if vision_config is None:
        token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [0] * seq_len)
        seq_data = SequenceData(token_ids)
        return DummyData(seq_data, None)
    elif isinstance(vision_config, dict):
        image_size = vision_config["image_size"]
        image_placeholder_length = calculate_image_placeholder(vision_config)
        token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [hf_config.boi_token_id] +
                          [0] * image_placeholder_length +
                          [hf_config.eoi_token_id])
        token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                           [0] * (seq_len - image_placeholder_length - 2))
        seq_data = SequenceData(token_ids)

        mm_data = {
            "image": Image.new("RGB", (image_size, image_size), color=0)
        }

        return DummyData(seq_data, mm_data)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def find_all_positions(input_ids: List[int], target: int) -> List[int]:
    return [index for index, value in enumerate(input_ids) if value == target]


def input_processor_for_glmv(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = getattr(hf_config, 'vision_config', None)

    if vision_config is None:
        return inputs
    elif isinstance(vision_config, dict):
        image_placeholder_length = calculate_image_placeholder(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    input_ids = inputs["prompt_token_ids"]

    tokenizer = cached_get_tokenizer(
        ctx.model_config.model,
        trust_remote_code=ctx.model_config.trust_remote_code)

    try:
        raw_batch_data = tokenizer.apply_chat_template(
            conversation=[{
                "role": "user",
                "image": multi_modal_data["image"],
                "content": inputs['prompt'],
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).data
    except Exception:
        logger.error("Failed to process content (%s)", inputs['prompt'])
        raise
    input_ids = raw_batch_data['input_ids'][0].tolist()

    boi_token_id = hf_config.boi_token_id
    eoi_token_id = hf_config.eoi_token_id
    boi_positions = find_all_positions(input_ids, boi_token_id)
    eoi_positions = find_all_positions(input_ids, eoi_token_id)

    assert len(boi_positions) == len(eoi_positions)

    new_input_ids = []
    final_processed_position = 0

    for boi_position, eoi_position in zip(boi_positions, eoi_positions):
        assert boi_position < eoi_position
        new_input_ids.extend(input_ids[final_processed_position:boi_position +
                                       1])
        new_input_ids.extend([input_ids[boi_position + 1]] *
                             image_placeholder_length)
        final_processed_position = eoi_position

    new_input_ids.extend(input_ids[final_processed_position:])

    prompt = inputs.get("prompt")
    if prompt is None:
        prompt = tokenizer.decode(new_input_ids)

    return token_inputs(
        prompt_token_ids=new_input_ids,
        prompt=prompt,
        multi_modal_data=multi_modal_data,
    )


class GLM4VProcessingInfo(BaseProcessingInfo):
    pass

    def __init__(self, ctx):
        super().__init__(ctx)
        self._pre_calculate()

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:

        return {"image": self.image_token_num}

    def _pre_calculate(self):
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        self.image_token_num = (vision_config["image_size"] //
                             vision_config["patch_size"] // 2)**2
        self.image_szie = vision_config["image_size"]

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin],
    ) -> int:
        return self.image_token_num

    def get_image_size(self) -> ImageSize:

        return ImageSize(height=self.image_szie, width=self.image_szie)


class GLM4VDummyInputsBuilder(BaseDummyInputsBuilder[GLM4VProcessingInfo]):
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size()

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }
        text="<|endoftext|>"
        return ProcessorInputs(
            prompt_text=text,
            mm_data=mm_data,
        )


class GLM4VMultiModalProcessor(BaseMultiModalProcessor[GLM4VProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            images=MultiModalFieldConfig.batched("image"),

        )

    
    
    def _apply_hf_processor_main(
        self,
        prompt: Union[str, list[int]],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_replacement: bool,
    ) -> tuple[list[int], MultiModalKwargs]:
        """
        Apply the HF processor on the prompt text and multi-modal data.

        Note:
            If :code:`enable_hf_prompt_replacement=False`, the prompt should
            correspond to the multi-modal items.
        """
        if isinstance(prompt, str):
            if enable_hf_prompt_replacement:
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                )

            prompt_ids = self._apply_hf_processor_text_only(prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)

        mm_missing_kwargs = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return prompt_ids, mm_missing_kwargs

    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            tokenizer = self.info.get_tokenizer()
            prefix = "<|begin_of_image|><|endoftext|><|end_of_image|>"
            prompt_ids = tokenizer.encode(prompt + prefix)

            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        try:
            
            tokenizer = self.info.get_tokenizer()
            img=mm_data.get("images",None)[0]
            raw_batch_data = tokenizer.apply_chat_template(
                conversation=[
                    {
                        "role": "user",
                        "image":img,
                        "content": prompt,
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).data
        except Exception:
            logger.error("Failed to process content (%s)", prompt)

        return BatchFeature(
            dict(
                input_ids=raw_batch_data["input_ids"],
                images=[raw_batch_data["images"][0]] if mm_data else None,
            )
        )

        # return token_inputs(
        #     prompt_token_ids=new_input_ids,
        #     prompt=prompt,
        #     multi_modal_data=mm_data,
        # )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        tokenizer=self.info.get_tokenizer()
        image_token_str = "<|endoftext|>"
        image_token_id = tokenizer.convert_tokens_to_ids(image_token_str)
        def get_replacement(item_idx: int):
            num_image_tokens=self.info.get_num_image_tokens(image_height=1120,image_width=1120,processor=None)
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]
            


class GLMAttention(nn.Module):

    def __init__(
        self,
        config: ChatGLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        # NOTE: THUDM/cogagent-9b-20241220 uses original_rope=False,
        # which is equivalent to is_neox_style=True
        is_neox_style = not config.original_rope
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim // 2,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size,
                                               eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = GLMAttention(config,
                                           cache_config,
                                           quant_config,
                                           prefix=f"{prefix}.self_attention")
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config: ChatGLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_layers,
            lambda prefix: GLMBlock(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i - self.start_layer],
                attn_metadata=attn_metadata,
            )
        # Final layer norm.
        if get_pp_group().is_last_rank and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.embedding = VocabParallelEmbedding(config.padded_vocab_size,
                                                config.hidden_size,
                                                quant_config=quant_config)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config,
                                      cache_config,
                                      quant_config,
                                      prefix=f"{prefix}.encoder")

        self.output_layer = ParallelLMHead(config.padded_vocab_size,
                                           config.hidden_size,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.output_layer")

        vision_config_flag = getattr(config, 'vision_config', None)
        if vision_config_flag is not None:
            self.vision_config = Namespace(**config.vision_config)
            self.vision = EVA2CLIPModel(self.config,
                                        quant_config,
                                        prefix=f"{prefix}.vision")
        else:
            self.vision = None

        self.make_empty_intermediate_tensors = (
            self.encoder.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> GLMImagePixelInputs:

        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is not None and self.vision is not None:
            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.ndim > 2:
                    pixel_values = torch.concat(list(pixel_values))
            elif isinstance(pixel_values, list):
                return torch.concat(pixel_values)
            else:
                raise TypeError("""pixel_values must be a torch.Tensor
                    or a list of torch.Tensor
                    """)
        return GLMImagePixelInputs(pixel_values=pixel_values)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input["pixel_values"] is None:
            return None
        pixel_values = image_input["pixel_values"].to(
            dtype=self.config.torch_dtype)
        vision_embeddings = self.vision(pixel_values)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_glm_vision_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                vision_embeddings=multimodal_embeddings,
                boi_token_id=self.config.boi_token_id,
                eoi_token_id=self.config.eoi_token_id)
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
    ) -> torch.Tensor:

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        if intermediate_tensors is None and inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None
        else:
            inputs_embeds = intermediate_tensors["hidden_states"]

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("linear_proj.merged_proj", "linear_proj.gate_proj", 0),
            ("linear_proj.merged_proj", "linear_proj.dense_h_to_4h", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
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
                if "rotary_pos_emb.inv_freq" in name:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class ChatGLMBaseModel(nn.Module, SupportsLoRA, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".word_embeddings": ""}, )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config

        self.quant_config = quant_config
        self.max_position_embeddings = getattr(config, "max_sequence_length",
                                               8192)
        self.transformer = ChatGLMModel(vllm_config=vllm_config,
                                        prefix=maybe_prefix(
                                            prefix, "transformer"))
        if self.config.tie_word_embeddings:
            self.transformer.output_layer.weight = (
                self.transformer.embedding.weight)
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(config.padded_vocab_size)
        self.sampler = get_sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors,
                                         **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class ChatGLM(ChatGLMBaseModel):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"]
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]

    embedding_modules = {}
    embedding_padding_modules = []


class ChatGLMV(ChatGLMBaseModel, SupportsMultiModal):

    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"],
        "merged_proj": ["gate_proj", "dense_h_to_4h"]
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        # vision
        "fc1",
        "fc2",
        "merged_proj",
        "linear_proj"
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="transformer.encoder",
            connector="transformer.vision.linear_proj",
            tower_model="transformer.vision.transformer")


# @MULTIMODAL_REGISTRY.register_image_input_mapper(mm_input_mapper_for_glmv)
# @MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_glmv_image_tokens)
# @INPUT_REGISTRY.register_dummy_data(dummy_data_for_glmv)
# @INPUT_REGISTRY.register_input_processor(input_processor_for_glmv)


@MULTIMODAL_REGISTRY.register_processor(GLM4VMultiModalProcessor,
                                        info=GLM4VProcessingInfo,
                                        dummy_inputs=GLM4VDummyInputsBuilder)
class ChatGLMForCausalLM(ChatGLMBaseModel, SupportsLoRA, SupportsPP,
                         SupportsMultiModal):
    # Ensure that the LoRA support check passes when the class is not
    # initialized, but set all these attributes to empty.
    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    def __new__(
        cls,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        config = vllm_config.model_config.hf_config
        # Initialize VL
        if hasattr(config, "vision_config"):
            return ChatGLMV(vllm_config=vllm_config, prefix=prefix)
        # Initialize LLM
        else:
            return ChatGLM(vllm_config=vllm_config, prefix=prefix)
