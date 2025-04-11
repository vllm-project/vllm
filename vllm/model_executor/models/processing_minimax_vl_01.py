"""
Processor class for MiniMaxVL01.
"""

from typing import List, Union, Mapping, TypeVar

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin#, _validate_images_text_input_order
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from .image_processer import CustomBatchFeature
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

logger = logging.get_logger(__name__)

import os
import numpy as np

LEGACY_PROCESSING = int(os.getenv('LEGACY_PROCESSING', 1))

class MiniMaxVL01ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }

def get_hw_multiple_of(image_size, multiple, max_size=None):
    w, h = image_size
    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)
    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0
        if new_w > max_w or new_h > max_h:
            # ratio = min(max_w / new_w, max_h / new_h)
            # new_w = int(new_w * ratio)
            # new_h = int(new_h * ratio)
            new_w = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)

            new_w = new_w if new_w % multiple == 0 else new_w + (multiple - new_w % multiple)
            new_h = new_h if new_h % multiple == 0 else new_h + (multiple - new_h % multiple)
        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h
    return new_w, new_h

def split_special_tokens(text, special_tokens):
    # 使用正则表达式匹配所有特殊标记及其前后内容
    import re
    pattern = '|'.join(map(re.escape, special_tokens))
    return re.split(f'({pattern})', text)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit 


def get_w_h_num(resolution, best_resolution):
    original_width, original_height = resolution
    current_width, current_height = best_resolution

    current_height = int(current_height)
    current_width = int(current_width)
    original_height = int(original_height)
    original_width = int(original_width)

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        w_num = current_width
        h_num = current_height - 2*padding
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * current_height) // original_height
        
        padding = (current_width - new_width) // 2
        w_num = current_width - 2*padding
        h_num = current_height

    return (w_num, h_num)
    
def get_num_token(img_h, img_w, grid_pinpoints, patch_size):
    #patch_size = 14
    #grid_pinpoints = eval("[(336, 336), (336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016)]")
    best_resolution = select_best_resolution((img_w,img_h), grid_pinpoints)
    resized_w, resized_h = best_resolution
    w_num, h_num = get_w_h_num((img_w, img_h), (resized_w// patch_size, resized_h// patch_size))
    total_token = int((w_num+1) * h_num) + (336//patch_size)**2
    return total_token


class MiniMaxVL01Processor(ProcessorMixin):
    r"""
    Constructs a MiniMaxVL01 processor which wraps a MiniMaxVL01 image processor and a MiniMaxVL01 tokenizer into a single processor.
    [`MiniMaxVL01Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MiniMaxVL01Processor.__call__`] and [`~MiniMaxVL01Processor.decode`] for more information.
    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        
        # 确保 tokenizer 是 PreTrainedTokenizerBase 类型
        if hasattr(tokenizer, 'get_hf_tokenizer'):
            tokenizer = tokenizer.get_hf_tokenizer()
        elif hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
            
        # 确保 image_processor 是 ImageProcessor 类型
        if hasattr(image_processor, 'get_hf_processor'):
            image_processor = image_processor.get_hf_processor()
            
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.patch_size = image_processor.patch_size
        self.grid_pinpoints = image_processor.image_grid_pinpoints
        self.max_size = image_processor.size
        self.process_image_mode = image_processor.process_image_mode

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # check if images and text inputs are reversed for BC
        #images, text = _validate_images_text_input_order(images, text)
        output_kwargs = self._merge_kwargs(
            MiniMaxVL01ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            if self.process_image_mode == 'anyres':
                if LEGACY_PROCESSING:# 推理时不提前替换image token
                    pixel_values = image_inputs["pixel_values"]
                    image_sizes = image_inputs["image_sizes"]
                    # height, width = get_image_size(to_numpy_array(pixel_values[0]))
                    # num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
                    # if self.vision_feature_select_strategy == "default":
                    #     num_image_tokens -= 1
                    all_image_tokens = []
                    for pixel_value, image_size in zip(pixel_values, image_sizes):
                        height, width = image_size
                        num_image_tokens = get_num_token(height, width, self.grid_pinpoints, self.patch_size)
                        # if self.vision_feature_select_strategy == "default":
                        #     num_image_tokens -= 1
                        all_image_tokens.append(num_image_tokens)
                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                final_text += _sample * all_image_tokens[image_index]
                                image_index += 1
                            else:
                                final_text += _sample
                        #sample = sample.replace(self.image_token, self.image_token * all_image_tokens)
                        prompt_strings.append(final_text)
            elif self.process_image_mode == 'resize':
                pixel_values = image_inputs["pixel_values"]
                all_image_tokens = []
                for pixel_value in pixel_values:
                    height, width = get_image_size(to_numpy_array(pixel_value))
                    all_image_tokens.append(int(height*width/self.patch_size**2))
                
                prompt_strings = []
                image_index = 0
                for sample in text:
                    split_text = split_special_tokens(sample, [self.image_token])
                    final_text = ''
                    for i, _sample in enumerate(split_text):
                        if _sample == self.image_token:
                            final_text += _sample * all_image_tokens[image_index]
                            image_index += 1
                        else:
                            final_text += _sample
                    #sample = sample.replace(self.image_token, self.image_token * all_image_tokens)
                    prompt_strings.append(final_text)
            else:
                
                if self.patch_size is not None:
                    # Replace the image token with the expanded image token sequence
                    pixel_values = image_inputs["pixel_values"]
                    # height, width = get_image_size(to_numpy_array(pixel_values[0]))
                    # num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
                    # if self.vision_feature_select_strategy == "default":
                    #     num_image_tokens -= 1
                    all_image_tokens = []
                    for pixel_value in pixel_values:
                        height, width = get_image_size(to_numpy_array(pixel_value))
                        new_width, new_height = get_hw_multiple_of((width, height), self.patch_size, self.max_size)
                        num_image_tokens = (new_height // self.patch_size) * (new_width // self.patch_size)# + 1
                        # if self.vision_feature_select_strategy == "default":
                        #     num_image_tokens -= 1
                        all_image_tokens.append(num_image_tokens)
                    
                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                final_text += _sample * all_image_tokens[image_index]
                                image_index += 1
                            else:
                                final_text += _sample
                        #sample = sample.replace(self.image_token, self.image_token * all_image_tokens)
                        prompt_strings.append(final_text)
                else:
                    logger.warning_once(
                        "Expanding inputs for image tokens in MiniMaxVL01 should be done in processing. "
                        "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                        "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )
                    raise ValueError(
                        "You need to provide `patch_size` and `vision_feature_select_strategy` in the model's processing config to expand inputs for image tokens."
                    )

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        #return {**text_inputs, **image_inputs}
        return CustomBatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))