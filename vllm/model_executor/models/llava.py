"""Inference-only LLaVA model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlavaConfig, AutoModel
from transformers.activations import ACT2FN

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.sequence import SamplerOutput
from vllm.logger import init_logger
import numpy as np

logger = init_logger(__name__)
KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlavaMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.linear_method = linear_method

        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.language_model = LlamaForCausalLM(config.text_config,
                                               linear_method)
        self.multi_modal_projector = LlavaMultiModalProjector(config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def prepare_promt(self,
                      input_ids: List[int],
                      pixel_values: torch.Tensor = None):
        input_ids = np.asarray(input_ids)
        assert len(
            input_ids.shape
        ) == 1, f"input_ids should be 1D array, got {input_ids.shape}"

        # Create a mask to know where image tokens are
        image_token_mask = input_ids == self.config.image_token_index
        non_image_indices = np.where(
            input_ids != self.config.image_token_index)

        # check the number of image tokens and images
        num_image_tokens = image_token_mask.sum()
        num_images = 0 if pixel_values is None else pixel_values.shape[0]
        assert num_images == num_image_tokens, f" The input provided to the model are wrong. The number of image tokens ({num_image_tokens}) is not equal to the number of images ({num_images}) provided."

        # expand each image token to image_hidden_dim
        if pixel_values is not None:
            # get image features
            pixel_values = pixel_values.to('cuda')
            image_outputs = self.vision_tower(pixel_values,
                                              output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.

            selected_image_feature = image_outputs.hidden_states[
                self.config.vision_feature_layer]
            if self.config.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.config.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )
            image_features = self.multi_modal_projector(
                selected_image_feature).cpu()
            nb_images, image_hidden_dim, embed_dim = image_features.shape

            # Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            new_token_positions = np.cumsum(
                (image_token_mask * (image_hidden_dim - 1) + 1), -1) - 1
            text_to_overwrite = new_token_positions[non_image_indices]

            final_input_ids = np.ones(
                (num_images * (image_hidden_dim - 1)) + len(input_ids),
                dtype=input_ids.dtype) * self.config.image_token_index
            final_input_ids[text_to_overwrite] = input_ids[non_image_indices]

            input_ids = final_input_ids
            image_features = image_features.contiguous().reshape(-1, embed_dim)
        else:
            image_features = None
        return input_ids, image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        image_features: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            # Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # TODO change the vision_tower to parallel version or pre-compute the image features somewhere else
            #   currently, if put the vision_tower here will cause duplicated process.

            # repace the embedding of image tokens with the image features.
            if image_features is not None and input_ids.shape[1] != 1:
                image_token_mask = input_ids == self.config.image_token_index
                # image_features is a list of tensor, len(image_features) == batch_size
                #   each tensor is a concatenate of image features, there shapes are not the same,
                #   and may be None if the prompt have no image tokens.
                #   shape: [image_num * image_hidden_dim, embed_dim], image_hidden_dim: feature tokens per image
                for i, features in enumerate(image_features):
                    if features is not None:  # the prompt have a image
                        inputs_embeds[i][image_token_mask[i]] = features.to(
                            inputs_embeds)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                pass

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            input_metadata,
                                            inputs_embeds=inputs_embeds)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        return self.language_model.sample(hidden_states, sampling_metadata)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        unused_keys = []

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            if name.startswith(
                    "model.language_model"):  # load language model weights
                name = name[6:]  # remove "model." prefix
                if "rotary_emb.inv_freq" in name:
                    continue
                if ("rotary_emb.cos_cached" in name
                        or "rotary_emb.sin_cached" in name):
                    # Models trained using ColossalAI may include these tensors in
                    # the checkpoint. Skip them.
                    continue
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if params_dict.get(name, None) is None:
                        unused_keys.append(name)
                    else:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            elif name.startswith("model.vision_tower") or name.startswith(
                    'model.multi_modal_projector'
            ):  # load vision model weights
                name = name[6:]  # remove "model." prefix
                if params_dict.get(name, None) is None:
                    unused_keys.append(name)
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            else:
                # duplicate keys with out 'model.' prefix
                pass

        if len(unused_keys) > 0:
            unused_keys.sort()
            logger.warning(
                f"These keys found in checkpoint but not used in model! {unused_keys}"
            )
