import os
import logging
from packaging import version
from importlib import import_module
from typing import Iterable, List, Set, Tuple, Union, Optional, Dict
import PIL

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

import transformers
from torch.nn.functional import softmax, gumbel_softmax, pad

from vllm.transformers_utils.configs.ovis import ConversationFormatter,OvisConfig
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import cached_get_image_processor
from vllm.sequence import IntermediateTensors

from .siglip import SiglipVisionModel
from .interfaces import SupportsMultiModal,SupportsPP  
from .utils import (AutoWeightsLoader, maybe_prefix,init_vllm_registered_model)

IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]

def get_max_ovis_image_tokens(ctx:InputContext,max_partitions:int=9):
    hf_config = ctx.get_hf_config(OvisConfig)
    visual_tokenizer_config = hf_config.visual_tokenizer_config
    
    hidden_stride = visual_tokenizer_config.hidden_stride
    vocab_size = visual_tokenizer_config.vocab_size
    
    max_grid_size = max_partitions * max_partitions
    max_tokens_per_grid = hidden_stride * hidden_stride * max_grid_size
    
    reserved_tokens = len(IMAGE_INDICATOR_IDS)
    
    usable_vocab_size = vocab_size - reserved_tokens
    max_image_tokens = min(max_tokens_per_grid,usable_vocab_size)
    return max_image_tokens

class SiglipVisualTokenizer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, *inputs, **kwargs):
        super().__init__()
        quant_config = vllm_config.quant_config
        config = vllm_config.model_config.hf_config.visual_tokenizer_config
        self.config = config
        self.image_processor = cached_get_image_processor(kwargs['image_processor_name_or_path'])
        self.backbone = SiglipVisionModel(config.backbone_config._name_or_path,
                                              quant_config,
                                              prefix="vision_backbone")
        head_dim = self.config.vocab_size - len(IMAGE_INDICATOR_IDS)  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                self.backbone.config.hidden_size * self.config.hidden_stride * self.config.hidden_stride, head_dim,
                bias=False
            ),
            torch.nn.LayerNorm(head_dim)
        )

    def get_backbone(self):
        return self.backbone

    def get_image_processor(self):
        return self.image_processor

    def mock_input(self):
        height, width = self.get_image_size()
        return torch.zeros(1, 3, height, width), self.construct_image_placeholders((1, 1))

    def get_head(self):
        return self.head
    
    def get_image_size(self):
        height = self.image_processor.size["height"]
        width = self.image_processor.size["width"]
        return height, width

    @staticmethod
    def construct_image_placeholders(grid):
        image_placeholders = [IMAGE_INDICATOR_IDS[0], IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS[1]]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(IMAGE_ATOM_ID)
                    if c < grid[1] - 1:
                        image_placeholders.append(IMAGE_INDICATOR_IDS[2])
                if r < grid[0] - 1:
                    image_placeholders.append(IMAGE_INDICATOR_IDS[3])
        image_placeholders.append(IMAGE_INDICATOR_IDS[4])
        return image_placeholders

    def preprocess_image(self, image: PIL.Image.Image, max_partition=9, covering_threshold=0.9, convert_to_rgb=True):
        def _preprocess(img: PIL.Image.Image, side):
            # first resize and preprocess
            w, h = img.size
            if w == h:
                new_width = new_height = side
            elif w > h:
                new_width = side
                new_height = int(h / w * new_width)
            else:
                new_height = side
                new_width = int(w / h * new_height)
            new_size = dict(height=new_height, width=new_width)
            pixel_values = self.image_processor.preprocess(img, size=new_size, return_tensors='pt')['pixel_values']

            # then pad to square
            square_values = torch.zeros([1, 3, side, side], dtype=pixel_values.dtype, device=pixel_values.device)
            new_height, new_width = pixel_values.shape[2:]
            if new_height == new_width:
                square_values[:, :, :, :] = pixel_values
            elif new_height > new_width:
                from_index = (side - new_width) // 2
                square_values[:, :, :, from_index:from_index + new_width] = pixel_values
            else:
                from_index = (side - new_height) // 2
                square_values[:, :, from_index:from_index + new_height, :] = pixel_values

            return square_values

        def _partition(img, grid):
            w, h = img.size
            row_height = h // grid[0]
            col_width = w // grid[1]

            partition = []
            for row in range(grid[0]):
                for col in range(grid[1]):
                    left = col * col_width
                    upper = row * row_height
                    right = w if col == grid[1] - 1 else (col + 1) * col_width
                    lower = h if row == grid[0] - 1 else (row + 1) * row_height
                    partition.append((left, upper, right, lower))

            return partition

        def _covering_area(left, upper, right, lower, side):
            w = right - left
            h = lower - upper
            w, h = max(w, h), min(w, h)
            if w > side:
                h = h / w * side
                w = side
            return w * h

        def _get_best_grid(img, side):
            img_area = img.size[0] * img.size[1]

            candidate_grids = []
            for i in range(1, max_partition + 1):
                for j in range(1, max_partition + 1):
                    if i * j <= max_partition:
                        candidate_grids.append((i, j))

            all_grids = []
            good_grids = []
            for grid in candidate_grids:
                partition = _partition(img, grid)
                covering_ratio = sum([_covering_area(*p, side) for p in partition]) / img_area
                assert covering_ratio <= 1.0
                all_grids.append((grid, covering_ratio))
                if covering_ratio > covering_threshold:
                    good_grids.append((grid, covering_ratio))

            if len(good_grids) > 0:
                # pick the good partition with minimum #sub_images and break the tie using covering_ratio
                return sorted(good_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][0]
            else:
                # pick the partition with maximum covering_ratio and break the tie using #sub_images
                return sorted(all_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]

        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')

        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError('get_image_size() returns non-square size')
        side = sides[0]
        grid = _get_best_grid(image, side)
        partition = _partition(image, grid)
        crops = [image.crop(p) for p in partition]
        if len(crops) > 1:
            crops.insert(0, image)
        pixel_values = torch.cat([_preprocess(crop, side) for crop in crops], dim=0)
        image_placeholders = self.construct_image_placeholders(grid)
        return pixel_values, image_placeholders

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        output = self.backbone(pixel_values,interpolate_pos_encoding=True)
        features = output.hidden_states[-1]
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=3, this leads to a token length reduction: 729 -> 81 for siglip
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride - (sqrt_l % self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride, self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride, self.config.hidden_stride, d)
            features = features.permute(0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1, self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens

class VisualEmbedding(torch.nn.Embedding):
    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_ovis_image_tokens)
class OvisForConditionalGeneration(nn.Module,SupportsMultiModal,SupportsPP):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix,"language_model")
        )
        self.text_tokenizer = get_tokenizer(self.config.name_or_path)
        self.visual_tokenizer = SiglipVisualTokenizer(self.config,
                                                      image_processor_name_or_path=self.config.name_or_path)
        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size,
            device=self.visual_tokenizer.device,
            dtype=self.visual_tokenizer.dtype
        )

    def get_text_tokenizer(self):
        return self.text_tokenizer

    def get_visual_tokenizer(self):
        return self.visual_tokenizer

    def get_conversation_formatter(self) -> ConversationFormatter:
        if getattr(self, 'conversation_formatter', None) is None:
            self.conversation_formatter = getattr(import_module(".configuration_ovis", __package__),
                                                  self.config.conversation_formatter_class)(self.text_tokenizer)
        return self.conversation_formatter

    def merge_multimodal(
        self,
        text_input_ids: torch.Tensor,
        text_attention_masks: torch.Tensor,
        text_labels: Optional[torch.Tensor],
        pixel_values: List[Optional[torch.Tensor]],
        left_padding: bool = False
    ):
        input_device = text_input_ids.device
        visual_vocab_size = self.get_visual_tokenizer().config.vocab_size
        visual_indicator_embeds = self.get_vte()(
            torch.tensor(
                list(range(visual_vocab_size - 5, visual_vocab_size)),
                dtype=torch.long,
                device=self.get_visual_tokenizer().device
            )
        ).to(device=input_device)

        num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
        if sum(num_images) > 0:
            visual_tokens = self.visual_tokenizer(torch.cat([x for x in pixel_values if x is not None], dim=0))
            visual_embeds = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                        split_size_or_sections=num_images, dim=0)
            visual_input_ids = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                           split_size_or_sections=num_images, dim=0)
            visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                             visual_input_ids]
        else:
            # just placeholders
            visual_embeds = [None] * len(num_images)
            visual_input_ids = [None] * len(num_images)
            visual_labels = [None] * len(num_images)
        if text_labels is None:
            text_labels = torch.full(text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device)

        input_embeds = []
        attention_masks = []
        labels = []
        for text_input_id, text_label, text_attention_mask, visual_embed, visual_input_id, visual_label in zip(
                text_input_ids, text_labels, text_attention_masks, visual_embeds, visual_input_ids, visual_labels
        ):
            placeholder_token_mask = torch.lt(text_input_id, 0)
            text_embed = self.get_wte()(torch.masked_fill(text_input_id, placeholder_token_mask, 0))
            for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
                text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
            image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
            if len(image_atom_positions) > 0:
                input_embed_parts = []
                attention_mask_parts = []
                label_parts = []
                prev_image_atom_position = -1
                for index, image_atom_position in enumerate(image_atom_positions):
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:image_atom_position, :])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:image_atom_position])
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:image_atom_position])
                    input_embed_parts.append(visual_embed[index])
                    attention_mask_parts.append(
                        torch.ones_like(visual_label[index], dtype=torch.bool))
                    label_parts.append(visual_label[index])
                    prev_image_atom_position = image_atom_position
                if prev_image_atom_position + 1 < text_input_id.shape[0]:
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:, :])
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:])
                input_embed = torch.cat(input_embed_parts, dim=0)
                attention_mask = torch.cat(attention_mask_parts, dim=0)
                label = torch.cat(label_parts, dim=0)
            else:
                input_embed = text_embed
                attention_mask = text_attention_mask
                label = text_label
            input_embeds.append(input_embed)
            attention_masks.append(attention_mask)
            labels.append(label)

        batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0, left_padding=left_padding)
        batch_attention_mask = self.pad_truncate_sequence(attention_masks, batch_first=True, padding_value=False, left_padding=left_padding)
        batch_labels = self.pad_truncate_sequence(labels, batch_first=True, padding_value=IGNORE_ID, left_padding=left_padding)

        return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask

    def pad_truncate_sequence(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0, left_padding: bool = False) -> torch.Tensor:
        if left_padding == False:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
            return pad_sequence[:,:self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in sequences],batch_first=True, padding_value=padding_value).flip(dims=[1])
            return pad_sequence[:,-self.config.multimodal_max_length:]

    def preprocess_inputs(
        self,
        text_or_conversations: Union[List[Dict], str],
        images: Optional[List[PIL.Image.Image]],
        max_partition=9,
        generation_preface='',
        return_labels=False,
        propagate_exception=True
    ):
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{
                "from": "human",
                "value": text_or_conversations
            }]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        else:
            raise ValueError(f'Invalid type of `text_or_conversations`, expected `List[Dict]` or `str`,'
                             f' but got {type(text_or_conversations)}')

        # format conversations
        prompt, raw_input_ids, raw_labels = self.get_conversation_formatter().format(
            conversations, generation_preface=generation_preface)

        # place image placeholders
        input_ids = []
        labels = []
        pixel_values = []
        invalidate_label = False
        image_token_indices = [i for i, v in enumerate(raw_input_ids) if v == IMAGE_TOKEN_ID]
        last_image_token_index = -1
        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])
            labels.extend(raw_labels[head:tail])
            try:
                image = images[i]
                raw_pixel_values, image_placeholders = self.visual_tokenizer.preprocess_image(
                    image, max_partition=max_partition)
            except Exception as e:
                if propagate_exception:
                    raise e
                logging.exception(e)
                invalidate_label = True
                raw_pixel_values, image_placeholders = self.visual_tokenizer.mock_input()
            input_ids.extend(image_placeholders)
            labels.extend([IGNORE_ID] * len(image_placeholders))
            pixel_values.append(raw_pixel_values)
        input_ids.extend(raw_input_ids[last_image_token_index + 1:])
        labels.extend(raw_labels[last_image_token_index + 1:])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_ID] * len(labels) if invalidate_label else labels, dtype=torch.long)
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None

        if return_labels:
            return prompt, input_ids, pixel_values, labels
        else:
            return prompt, input_ids, pixel_values

    def forward(
        self,
        inputs : torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs
    ) -> Union[SamplerOutput, IntermediateTensors]:
        input_ids, inputs_embeds, _, _ = self.merge_multimodal(
            text_input_ids=inputs,
            text_attention_masks=kwargs.pop('attention_mask'),
            text_labels=None,
            pixel_values=kwargs.pop('pixel_values'),
            left_padding=True
        )
        
        hidden_states = self.llm(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,   
            inputs_embeds=inputs_embeds
        )
        
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.llm.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.llm.sample(logits, sampling_metadata)
    
    def load_weights(self,weights:Iterable[Tuple[str,torch.Tensor]])->Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
        
        