from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import nn

# TODO(xwjiang): We should port CLIPVisionModel's code over to not depend on
# transformers' impl.
from transformers import CLIPVisionModel

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.config import VisionLanguageConfig, LoRAConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]

if TYPE_CHECKING:
    from transformers import LlavaConfig  # pylint: disable=ungrouped-imports
    from vllm.model_executor.layers import LinearMethodBase  # pylint: disable=ungrouped-imports

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


# TODO(xwjiang): Run benchmark and decide if TP.
class LlavaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def _merge_vision_embeddings(input_ids: torch.Tensor,
                             inputs_embeds: torch.Tensor,
                             vision_embeddings: torch.Tensor,
                             image_token_id: int):
    """In place merges in vision_embeddings with inputs_embeds."""
    mask = (input_ids == image_token_id)
    inputs_embeds[mask] = vision_embeddings.view(-1,
                                                 vision_embeddings.shape[-1])


# TODO(xwjiang): Add model correctness test. This model should produce the same
# output as huggingface.
class LlavaForConditionalGeneration(nn.Module):
    supports_lora = False

    def __init__(self,
                 config: "LlavaConfig",
                 vision_language_config: VisionLanguageConfig,
                 linear_method: Optional["LinearMethodBase"] = None,
                 lora_config: Optional["LoRAConfig"] = None) -> None:
        # TODO(xwjiang): Remove when adding LoRA support.
        assert not lora_config
        super().__init__()
        self.config = config

        hf_text_config = config.text_config
        hf_vision_config = config.vision_config
        self.vision_language_config = vision_language_config

        assert self.vision_language_config

        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_tower = CLIPVisionModel(hf_vision_config)
        else:
            self.vision_tower = None

        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=hf_vision_config.hidden_size,
            text_hidden_size=hf_text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)

        self.linear_method = linear_method
        self.language_model = LlamaModel(hf_text_config,
                                         linear_method,
                                         lora_config=lora_config)
        self.unpadded_vocab_size = hf_text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            hf_text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size)
        self.sampler = Sampler(self.unpadded_vocab_size,
                               self.language_model.org_vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                input_metadata: InputMetadata,
                image_input: Optional[torch.Tensor] = None) -> SamplerOutput:
        """Run forward pass for Llava.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.
        Concretely, consider a text prompt:
        "<image>\nUSER: What's the content of the image?\nASSISTANT:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted image has a size of 576 (24 * 24) along the context
        length dimension.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        There will be 576 `32000` in the `input_ids`.
        (32000 is the token id for `<image>`.)

        This way, the `positions` and `input_metadata` are consistent
        with the `input_ids`.

        The design is chosen so that all data massaging happens inside
        Llava model.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            image_input: A batch of image inputs (either of type PIXEL_VALUES
                or IMAGE_FEATURES).
                For PIXEL_VALUES, expecting [1, 3, 336, 336].
                For IMAGE_FEATURES, expecting [1, 576, 1024].
        """
        if image_input is not None:
            # This should be ensured by ray-llm.
            # In case vllm entrypoint is used, this is a user facing error.
            if list(image_input.shape[1:]) != list(
                    self.vision_language_config.image_input_shape[1:]):
                raise ValueError(
                    f"The expected image tensor shape is batch dimension "
                    f"plus "
                    f"{self.vision_language_config.image_input_shape[1:]}."
                    f" You supplied {image_input.shape[1:]}. "
                    f"If you are using vLLM's entrypoint, make sure your "
                    f"supplied image input is consistent with "
                    f"image_input_shape in engine args.")
            if self.vision_tower is not None:
                # TODO(xwjiang): Port minimal CLIPVisionModel over.
                image_outputs = self.vision_tower(image_input,
                                                  output_hidden_states=True)
                image_features = image_outputs.hidden_states[
                    self.config.vision_feature_layer]
                if self.config.vision_feature_select_strategy == "default":
                    image_features = image_features[:, 1:]
            else:
                image_features = image_input
            vision_embeddings = self.multi_modal_projector(image_features)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            _merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.vision_language_config.image_token_id)

            hidden_states = self.language_model(None,
                                                positions,
                                                kv_caches,
                                                input_metadata,
                                                inputs_embeds=inputs_embeds)
        else:
            hidden_states = self.language_model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )

        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            normal_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    normal_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    normal_weight_loading = True
            if normal_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
