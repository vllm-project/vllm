# coding=utf-8
# Adapted from
# https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py
# Copyright 2023 The Omlab team.
# Copyright 2022 Haotian Liu and vLLM and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Haotian Liu's llava.
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
import os
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import LlamaConfig, CLIPVisionModel, CLIPImageProcessor, AutoConfig, \
    AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from vllm.model_executor import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.models.llama import LlamaModel, KVCache
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.parallel_utils.tensor_parallel import ColumnParallelLinear
from vllm.model_executor.weight_utils import hf_model_weights_iterator, load_tensor_parallel_weights

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



CLIP_MODEL_MAP={}
class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(nn.Module):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super().__init__()
        CLIP_MODEL_MAP.update({"openai/clip-vit-large-patch14": f"{os.path.abspath(config._name_or_path)}/clip-vit-large-patch14"})
        self.llama_model = LlamaModel(config)
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            if CLIP_MODEL_MAP[config.mm_vision_tower]:
                clip_model=CLIP_MODEL_MAP[config.mm_vision_tower]
            else:
                clip_model=config.mm_vision_tower
            self.vision_tower = [CLIPVisionModel.from_pretrained(clip_model).cuda()]
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model,torch_dtype=torch.float16)


        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(self.config_class.mm_vision_tower,
                                                             torch_dtype=torch.float16)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(self,
                input_ids: None,
                image_datas: List,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                input_metadata: InputMetadata,
                cache_events: Optional[List[torch.cuda.Event]],

                ) -> Union[Tuple, BaseModelOutputWithPast]:


        batch_images = image_datas
        inputs_embeds = self.llama_model.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and batch_images:
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                batch_image_tensors = []
                for images in batch_images:
                    if images is None:
                        batch_image_tensors.append(images)
                    elif type(images) is list:
                        # variable length images
                        image_features = []

                        for image in images:
                            image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                            select_hidden_state_layer = getattr(self.llama_model.config, "mm_vision_select_layer", -1)
                            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                            image_feature = select_hidden_state[:, 1:]
                            image_feature = self.mm_projector(image_feature)[0]
                            image_features.append(image_feature)
                        batch_image_tensors.append(image_features)
                    else:
                        image_forward_outs = vision_tower(images, output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.llama_model.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                        image_features = select_hidden_state[:, 1:]
                        image_features = self.mm_projector(image_features)[0]
                        batch_image_tensors.append(image_features)

            # updata input embed
            inputs_embeds = self.updata_input_embed(input_ids, inputs_embeds, batch_image_tensors,vision_tower)


        hidden_states = self.llama_model(input_ids, positions, kv_caches,
                                         input_metadata, cache_events, inputs_embeds=inputs_embeds)
        return hidden_states

    def updata_input_embed(self, input_ids, inputs_embeds, batch_image_tensors,vision_tower):
        image_tensors_list = [image_tensors[0] for image_tensors in batch_image_tensors]


        use_im_start_end = vision_tower.config.use_im_start_end
        if use_im_start_end:
            image_start_id = vision_tower.config.im_start_token
            image_end_id = vision_tower.config.im_end_token
        image_patch_id = vision_tower.config.im_patch_token
        if image_tensors_list:
            number_patch = image_tensors_list[0][0]
        else:
            return inputs_embeds

        def get_image_embed_index(input_ids, use_im_start_end):
            if use_im_start_end:
                img_embed_start_indexs = [_id.item()+1  for _id in torch.where(input_ids == image_start_id)[0]]
                img_embed_end_indexs = [_id.item() for _id in torch.where(image_end_id == input_ids)[0]]
            else:
                img_patch_indexs = [_id.item() for _id in torch.where(image_end_id == image_patch_id)[0]]
                img_embed_start_indexs=[]
                img_embed_end_indexs=[]
                for i in range(int(len(img_patch_indexs)/number_patch)):
                    img_embed_start_indexs.append(img_patch_indexs[i*number_patch])
                    img_embed_end_indexs.append(img_patch_indexs[(i+1)*number_patch]+1)
            return img_embed_start_indexs, img_embed_end_indexs

        img_embed_start_indexs, img_embed_end_indexs=get_image_embed_index(input_ids, use_im_start_end)

        def insert_image_embed(i_img):
            inputs_embeds[img_embed_start_indexs[i_img]:img_embed_end_indexs[i_img],:] = batch_image_tensors[i_img][0]

        for i_img in range(len(img_embed_start_indexs)):
            insert_image_embed(i_img)

        return inputs_embeds


    def get_batch_inputs(self, input_ids, inputs_embeds, positions):
        device = input_ids.device
        input_ids_list = list(input_ids.cpu().numpy())
        inputs_embeds_list = list(inputs_embeds.cpu().numpy())
        batch_input_ids, batch_inputs_embeds = [], []
        temp_ids, temp_embed = [], []
        for input_id, inputs_embed, position in zip(input_ids_list, inputs_embeds_list, positions):
            if position == 0:
                if temp_ids and temp_embed:
                    batch_input_ids.append(torch.tensor(temp_ids, device=device))
                    batch_inputs_embeds.append(torch.tensor(temp_embed, device=device))
                temp_ids, temp_embed = [], []
            temp_ids.append(input_id)
            temp_embed.append(inputs_embed)
        if temp_ids and temp_embed:
            batch_input_ids.append(torch.tensor(temp_ids, device=device))
            batch_inputs_embeds.append(torch.tensor(temp_embed, device=device))
        return batch_input_ids, batch_inputs_embeds


class LlavaLlamaForCausalLM(nn.Module):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlavaLlamaModel(config)

        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.Tensor,
            image_datas: List,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        hidden_states = self.model(input_ids, image_datas, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight",
        "gate_proj.weight", "up_proj.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "mm_projector" in name or "vision_tower" in name:
                pass
            else:
                name = name.replace("model", "model.llama_model")
            if "rotary_emb.inv_freq" in name:
                continue

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                    ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                                shard_size * tensor_model_parallel_rank:shard_size *
                                                                        (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                                                (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                                shard_size * tensor_model_parallel_rank:shard_size *
                                                                        (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                                                (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self,tokenizer):
        mm_use_im_start_end = getattr(self.model.llama_model.config, "mm_use_im_start_end", None)
        assert mm_use_im_start_end is not None, "please 'use_im_start_end' in llama config."
        # device
        vision_config = self.get_model().vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)


        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
