# SPDX-License-Identifier: Apache-2.0

# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/ovis/modeling_ovis.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Ovis2 model."""
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Dict, Union)
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from torch.nn import init

from transformers import PretrainedConfig, AutoConfig, AutoModel
from transformers import BatchFeature, AutoTokenizer
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.aimv2 import Aimv2VisualTokenizer
from vllm.model_executor.models.utils import maybe_prefix, flatten_bn, AutoWeightsLoader, init_vllm_registered_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs, NestedTensors,
                                    )
from vllm.multimodal.parse import (ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from collections import defaultdict

# Cannot find the following number from hf config.
IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]
MAX_SEGMENTS = 30  # default value in the ovis2 modeling

NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT = 256

# ----------------------------------------------------------------------
#                           Ovis2 Configuration
# ----------------------------------------------------------------------
class Ovis2Config(PretrainedConfig):
    model_type = "ovis2" # swithched to this to have compatible image token

    def __init__(
        self,
        llm_config: Optional[Union[PretrainedConfig, dict]] = None,
        visual_tokenizer_config: Optional[Union[PretrainedConfig, dict]] = None,
        multimodal_max_length=8192,
        hidden_size=None,
        conversation_formatter_class=None,
        llm_attn_implementation=None,
        disable_tie_weight=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PretrainedConfig, dict)), \
                f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config['model_type']
                llm_config.pop('model_type')
                llm_config = AutoConfig.for_model(model_type, **llm_config)
        self.llm_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(visual_tokenizer_config, (PretrainedConfig, dict)), \
                f"expect `visual_tokenizer_config` to be instance of PretrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            if not isinstance(visual_tokenizer_config, PretrainedConfig):
                model_type = visual_tokenizer_config['model_type']
                visual_tokenizer_config.pop('model_type')
                visual_tokenizer_config = AutoConfig.for_model(model_type, **visual_tokenizer_config)
        self.visual_tokenizer_config = visual_tokenizer_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
        #added to work with vllm
        self.num_hidden_layers = llm_config.num_hidden_layers
        self.vocab_size = llm_config.vocab_size
        self.num_attention_heads = llm_config.num_attention_heads if llm_config else 0




class Ovis2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            'max_partition':9,
            'covering_threshold':0.9,
            'convert_to_rgb':True,
        'return_tensors':'pt'},
    }



class Ovis2Processor(ProcessorMixin):
    r"""
    Constructs a Ovis2 processor which wraps a Ovis2 image processor and a Qwen2 tokenizer into a single processor.
    [`Ovis2Processor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Ovis2Processor.__call__`] and [`~Ovis2Processor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[Ovis2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

            Args:
                images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                    The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                    tensor. Both channels-first and channels-last formats are supported.
                text (`str`, `List[str]`, `List[List[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                    (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                    `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                    The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                    tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
                return_tensors (`str` or [`~utils.TensorType`], *optional*):
                    If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.

            Returns:
                [`BatchFeature`]: A [`BatchFeature`] with the following fields:

                - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
                - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
                  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
                  `None`).
                - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
                - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
                - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
                - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
                - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Ovis2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process all images first
        image_features = {}
        if images is not None:
            processed_images = []
            image_placeholders_list = []
            grids = []

            # Process each image
            for image in images if isinstance(images, list) else [images]:
                pixel_values, image_placeholders, grid = self.preprocess_image(
                    image=image, **output_kwargs["images_kwargs"]
                )
                processed_images.append(pixel_values)
                image_placeholders_list.append(image_placeholders)
                grids.append(grid)

            # assign all processed images
            if processed_images:
                image_features["image_placeholders"] = image_placeholders_list

        # Process text input
        if text is not None:

            if not isinstance(text, list):
                text = [text]

            tokenized_batched_text = self.tokenizer.batch_encode_plus(
                text,
                **output_kwargs["text_kwargs"]
            )
            image_token_id = self.tokenizer(self.tokenizer.extra_special_tokens['image_token'])['input_ids'][0]
            replaced_ids_list = []
            replaced_attn_mask_list = []
            idx = 0
            for ids_tensor, attn_mask in zip(tokenized_batched_text['input_ids'],
                                             tokenized_batched_text['attention_mask']):
                if image_token_id in ids_tensor and "image_placeholders" in image_features:
                    if idx < len(image_features["image_placeholders"]):
                        # Converts in list for ease of use
                        ids_list = ids_tensor.tolist()
                        attn_list = attn_mask.tolist()
                        placeholder_ids = image_features["image_placeholders"][idx]

                        new_ids = []
                        new_attn = []

                        # replace placeholders
                        for i, token_id in enumerate(ids_list):
                            if token_id == image_token_id:
                                new_ids.extend(placeholder_ids)
                                new_attn.extend([1] * len(placeholder_ids))
                            else:
                                new_ids.append(token_id)
                                new_attn.append(attn_list[i])

                        # Converts back to tensors
                        ids_tensor = torch.tensor(new_ids, dtype=torch.long)
                        attn_mask = torch.tensor(new_attn, dtype=torch.long)
                        idx += 1
                    else:
                        raise RuntimeError(
                            'Mismatch between the images you provided and the number of placeholder present in the text')

                replaced_ids_list.append(ids_tensor)
                replaced_attn_mask_list.append(attn_mask)

            if replaced_ids_list:
                replaced_and_tokenized_ids = torch.stack(replaced_ids_list)
                replaced_and_tokenized_attn_mask = torch.stack(replaced_attn_mask_list)
            else:
                replaced_and_tokenized_ids = torch.tensor([], dtype=torch.long)
                replaced_and_tokenized_attn_mask = torch.tensor([], dtype=torch.long)

            # Create the output with text features
            output = BatchFeature(
                data={
                    "input_ids": replaced_and_tokenized_ids,
                    "attention_mask": replaced_and_tokenized_attn_mask,
                }
            )

            # Add image features if present
            if image_features:
                output["pixel_values"] = processed_images
                output['grids'] = grids

            return output


        # If only images were provided
        return BatchFeature(data=image_features)


    def get_image_size(self):
        height = self.image_processor.crop_size["height"]
        width = self.image_processor.crop_size["width"]
        return height, width

    def get_token_value(self, tok):
            return self.tokenizer(self.tokenizer.extra_special_tokens[tok])["input_ids"][0]

    def construct_image_placeholders(self, grid):

        image_placeholders = [self.get_token_value('image_start'),
                              self.get_token_value('image_atom'),
                              self.get_token_value('image_prefix')]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(self.get_token_value('image_atom') )
                    if c < grid[1] - 1:
                        image_placeholders.append(self.get_token_value('image_col_sep'))
                if r < grid[0] - 1:
                    image_placeholders.append(self.get_token_value('image_row_sep'))
        image_placeholders.append(self.get_token_value('image_end'))
        return image_placeholders

    def preprocess_image(self, image: PIL.Image.Image, max_partition, covering_threshold, convert_to_rgb, return_tensors):
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
            pixel_values = self.image_processor.preprocess(img, size=new_size, return_tensors=return_tensors)['pixel_values']

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
        return pixel_values, image_placeholders, grid

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]


class Ovis2ImagePatchInputs(TypedDict):
    type: Literal["image_patches"]
    flat_data: torch.Tensor
    """
    Shape: 
    `(batch_size * num_patches, patch_size_x * patch_size_y * num_channels)`
    """

    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    This is used to restore the first two dimensions of `flat_data`.
    """


class VisualEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class Ovis2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Ovis2Config)

    def get_hf_processor(self,
                         **kwargs):
        return self.ctx.get_hf_processor(Ovis2Processor)

    def get_image_processor(self) -> Ovis2Processor:
        return self.get_hf_processor().image_processor  # type: ignore

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {# 32k is model token limit at the moment
            "image": self.get_hf_config().multimodal_max_length // (MAX_SEGMENTS *
                                                                    NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT)}

    def get_mm_max_tokens_per_item(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image":
                (mm_counts['image'] * MAX_SEGMENTS * 256) + 11
        }  # 6 image pos token

    def get_image_size(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(width=image_processor.size['shortest_edge'] * 9 * 2,
                         height=image_processor.size['shortest_edge'] * 9 * 2)


class Ovis2DummyInputsBuilder(BaseDummyInputsBuilder[Ovis2ProcessingInfo]):

    def get_dummy_processor_inputs(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int]
    ) -> ProcessorInputs:
        target_width, target_height = \
            self.info.get_image_size()
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
                self._get_dummy_images(width=target_width,
                                       height=target_height,
                                       num_images=num_images),
        }

        return ProcessorInputs(
            prompt_text='''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Describe the image.<|im_end|>
<|im_start|>assistant''',
            mm_data=mm_data,

        )


class Ovis2MultiModalProcessor(BaseMultiModalProcessor[Ovis2ProcessingInfo]):

    def _get_token_value(self, tok):
        return self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens[tok])["input_ids"]

    def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            #    # Avoid warning from HF logger for text-only input
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            # prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids) nope
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        return processed_outputs

    def _apply_hf_processor_tokens_only(
            self,
            prompt_tokens: list[int],
    ) -> list[int]:

        return prompt_tokens

    def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            grids=MultiModalFieldConfig.batched("image")
        )

    def _get_prompt_replacements(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:

        def get_replacement_tokens_ovis(grid):
            """
            Calculates the placeholder for the sequence, starting from the grid

            Args:
                grid: the grid tuple for the image
            Returns:
                list: Placeholder sequence for the image with padding
            """
            hf_processor = self.info.get_hf_processor()
            # Get the base placeholder tokens
            placeholder_tokens = hf_processor.construct_image_placeholders(grid)
            image_atom_token_id = \
                self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens['image_atom'])['input_ids'][0]

            # Extract the padding token ID from tokenizer
            image_padding_token_id = \
                self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens['image_pad'])['input_ids'][0]

            # Create a new list with padding tokens inserted
            padded_placeholder_tokens = []
            for token in placeholder_tokens:
                padded_placeholder_tokens.append(token)
                if token == image_atom_token_id:
                    # Add 255 padding tokens after each image atom token
                    padded_placeholder_tokens.extend([image_padding_token_id] * 255)

            return padded_placeholder_tokens

        return [
            PromptReplacement(
                modality="image",
                target=self.info.get_tokenizer()(
                    self.info.get_tokenizer()
                    .extra_special_tokens['image_token']
                )['input_ids'],
                replacement=get_replacement_tokens_ovis(grid),
            )
            for grid in out_mm_kwargs["grids"]]


@MULTIMODAL_REGISTRY.register_processor(Ovis2MultiModalProcessor,
                                        info=Ovis2ProcessingInfo,
                                        dummy_inputs=Ovis2DummyInputsBuilder)
class Ovis2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.padding_idx = config.pad_token_id
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(config.llm_config),
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.visual_tokenizer = Aimv2VisualTokenizer(
            config=config.visual_tokenizer_config,
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer",
            image_processor_name_or_path=config.visual_tokenizer_config.backbone_config.name_or_path,
        ).to(self.config.torch_dtype)

        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size,
            device='cuda',
            dtype=self.visual_tokenizer.dtype
        )

        # we'll instantiate a tokenizer and keep just the external mapping
        tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        self.extra_token_mapping = {
            k: tokenizer(v)['input_ids'][0] for k, v in tokenizer.extra_special_tokens.items()
        }

        self.extra_token_mapping_for_substitution = {
            k: tokenizer(v)['input_ids'][0] for k, v in tokenizer.extra_special_tokens.items() if k in
                                                                                                  {'image_atom',
                                                                                                   'image_pad'}
        }

        self.visual_indicators_embeds_dict = None
        # VocabParallelEmbedding( if enabled leads to numerical diff
        #    self.config.visual_tokenizer_config.vocab_size,
        #    self.config.hidden_size,
        #    params_dtype=self.visual_tokenizer.dtype,
        #    quant_config=quant_config,
        #    prefix=f"{prefix}.vte"
        # )

        # self.make_empty_intermediate_tensors = (
        #    self.language_model.make_empty_intermediate_tensors) ?

    def _init_embed_representation(self):
        if not self.visual_indicators_embeds_dict:
            # we precalcualte the embeddings for the image tokens
            visual_vocab_size = self.visual_tokenizer.config.vocab_size
            visual_indicator_embeds = self.vte(
                torch.tensor(
                    list(range(visual_vocab_size - 5, visual_vocab_size)),
                    dtype=torch.long,
                    device=self.vte.device
                )
            )

            self.visual_indicators_embeds_dict = {
                'image_start': visual_indicator_embeds[0],
                'image_prefix': visual_indicator_embeds[1],
                'image_col_sep': visual_indicator_embeds[2],
                'image_row_sep': visual_indicator_embeds[3],
                'image_end': visual_indicator_embeds[4],
            }

    @property
    def sampler(self):
        return self.llm.sampler

    def merge_multimodal(
            self,
            text_input_ids: Union[List[torch.Tensor], torch.Tensor],
            pixel_values: Optional[Union[List[torch.Tensor], torch.Tensor, object]],
            left_padding: bool = True  # must be true during inference
    ):  # todo check when different sized  inputs are batched
        # todo the tokenizer do not uses /n
        # we need to decompose the pixel_value_tensor
        # vllm batches it fi it is ccompatible otherwise it will pass it as  list
        self._init_embed_representation()
        if pixel_values is not None and not isinstance(pixel_values, list):
            if pixel_values.dim() == 6:
                # if is [tensor_batch, 1, num_segments, ch, w, h] we need -> [tensor_batch, num_segments, ch, w, h]
                pixel_values = pixel_values.squeeze(1)
                pixel_values = [pixel_value.to(self.config.torch_dtype) for pixel_value in pixel_values]
            else:
                pixel_values = [pixel_values]

        # When inference, sample can include only text with `None` pixel_value
        num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
        if sum(num_images) > 0:
            visual_tokens = self.visual_tokenizer(
                torch.cat(
                    [x for x in pixel_values if x is not None],
                    dim=0).to(self.visual_tokenizer.dtype)
            )

            visual_embeds = self.vte(visual_tokens)  # 1:1 numeric eq.


        else:
            # just placeholders
            visual_embeds = [None] * len(num_images)

        input_embeds = []

        for text_input_id, visual_embed in zip(text_input_ids, visual_embeds):

            placeholder_token_mask = torch.zeros_like(text_input_id, dtype=torch.bool)
            for value in self.extra_token_mapping_for_substitution.values():
                placeholder_token_mask |= torch.eq(text_input_id, value)

            text_embed = torch.zeros((text_input_id.shape[0], self.llm.model.norm.hidden_size),
                                     device=text_input_id.device, dtype=self.visual_tokenizer.dtype)
            text_embed[~placeholder_token_mask] = self.llm.model.embed_tokens(
                text_input_id[~placeholder_token_mask])  # 1:1

            for key, indicator_id in self.extra_token_mapping.items():
                if key in self.visual_indicators_embeds_dict:
                    text_embed[text_input_id == indicator_id] = self.visual_indicators_embeds_dict[key].to(
                        text_embed.device)
            # image_atom_positions = torch.where(torch.eq(text_input_id, self.extra_token_mapping['image_atom']))[0].tolist()
            # if len(image_atom_positions) > 0:
            # if not is_testing:
            #    input_embed_parts = []
            #    prev_image_atom_position = -1
            #    for index, image_atom_position in enumerate(image_atom_positions):
            #        input_embed_parts.append(
            #            text_embed[prev_image_atom_position + 1:image_atom_position, :])
            #
            #        input_embed_parts.append(visual_embeds[index])
            #
            #        prev_image_atom_position = image_atom_position
            #    if prev_image_atom_position + 1 < text_input_id.shape[0]:
            #        input_embed_parts.append(
            #            text_embed[prev_image_atom_position + 1:, :])
            #
            #    input_embed = torch.cat(input_embed_parts, dim=0)
            # else:

            # here we have already preallocated the multimodal tokens (in the testing phase) se the logic should be different
            # we should check consider that each atom token should replace 256 text tokens embeddings

            # It just needs this unified verison, since if no  images aare present it should just skip this
            text_embed[placeholder_token_mask] = visual_embeds.view(-1, text_embed.shape[-1])

            # else:
            #    input_embed = text_embed

            input_embeds.append(text_embed)

        batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0,
                                                        left_padding=left_padding)

        return batch_input_embeds

    def pad_truncate_sequence(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0,
                              left_padding: bool = False) -> torch.Tensor:
        if not left_padding:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first,
                                                           padding_value=padding_value)
            return pad_sequence[:, :self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in sequences], batch_first=True,
                                                           padding_value=padding_value).flip(dims=[1])
            return pad_sequence[:, -self.config.multimodal_max_length:]

    def get_tensor_formatted(self, input: Union[torch.Tensor, List]) -> List[torch.Tensor]:
        '''
        if thhe input is list check if its input arte 1d if so usueeze() them in 0
        if it is a tensor it needs to be splittend in a list
        :param input:
        :return:
        '''
        if isinstance(input, list):
            output_list = []
            for element in input:
                if element.dim() == 1:
                    output_list.append(element.unsqueeze(0))
                else:
                    output_list.append(element)
            return output_list
        else:
            return [tensor for tensor in input] if input.dim() > 1 else [input]

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
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None and 'pixel_values' in kwargs:  # vllm batches the input or make it a list but does not have a attn mask
            inputs_embeds = self.merge_multimodal(text_input_ids=self.get_tensor_formatted(input_ids),
                                                  pixel_values=kwargs['pixel_values'], )
            # is_testing = kv_caches[0].numel() == 0) valid approach but probably not needed
            # input_ids = None
        # up until here we have a inputs_embeds 100% numerical identity between the OG HF Transformers implementation and ours
        hidden_states = self.llm(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.llm.logits_processor(
            self.llm.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.llm.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
    torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)