# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/ManaEstras/transformers/blob/v4.57.1.hyvl/src/transformers/models/hunyuan_vl/processing_hunyuan_vl.py

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput


class HunYuanVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"  # ("AutoTokenizer", None)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        # TODO Fix the init
        self.tokenizer = tokenizer
        self.image_token_id = 120120  # self.tokenizer.image_token_id
        self.image_token = self.tokenizer.convert_ids_to_tokens(self.image_token_id)
        self.im_start_token_id = 120118  # self.tokenizer.im_start_id
        self.im_start_token = self.tokenizer.convert_ids_to_tokens(
            self.im_start_token_id
        )
        self.im_end_token_id = 120119  # self.tokenizer.im_end_id
        self.im_end_token = self.tokenizer.convert_ids_to_tokens(self.im_end_token_id)
        self.placeholder_token = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.vocab_size - 1
        )
        self.pad_id = 120002  # self.tokenizer.pad_token_id

        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        image_tokens_cumsum = [0]
        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid_h, grid_w = image_grid_thw[index][-2:]
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    num_image_tokens = patch_h * (patch_w + 1) + 2
                    image_tokens_cumsum.append(
                        image_tokens_cumsum[-1] + num_image_tokens
                    )
                    # text[i] = text[i].replace(self.image_token, self.im_start_token + self.placeholder_token * num_image_tokens + self.im_end_token, 1) # noqa: E501
                    text[i] = text[i].replace(
                        self.image_token, self.placeholder_token * num_image_tokens, 1
                    )
                    index += 1
                text[i] = text[i].replace(self.placeholder_token, self.image_token)
                # text[i] = self.tokenizer.bos_token + text[i]

        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        input_ids = text_inputs["input_ids"]
        position_ids = torch.arange(len(input_ids[0]))
        position_ids_w = torch.arange(len(input_ids[0]))
        position_ids_h = torch.arange(len(input_ids[0]))
        position_ids_t = torch.arange(len(input_ids[0]))

        if images is not None:
            image_token_pos_indices = torch.where(input_ids[0] == self.image_token_id)[
                0
            ]
            for i in range(len(image_grid_thw)):
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // self.image_processor.merge_size
                patch_w = grid_w // self.image_processor.merge_size
                start_pos = image_token_pos_indices[image_tokens_cumsum[i]].item() + 1
                replace_num = (patch_w + 1) * patch_h
                position_ids_w[start_pos : start_pos + replace_num] = torch.tensor(
                    list(range(patch_w + 1)) * patch_h, dtype=torch.int64
                )
                patch_h_list = []
                for h in range(patch_h):
                    patch_h_list += [h] * (patch_w + 1)
                position_ids_h[start_pos : start_pos + replace_num] = torch.tensor(
                    patch_h_list, dtype=torch.int64
                )
                position_ids_t[start_pos : start_pos + replace_num] = 0

        position_ids = torch.stack(
            [position_ids, position_ids_w, position_ids_h, position_ids_t]
        ).unsqueeze(0)
        text_inputs["position_ids"] = position_ids

        attention_mask = input_ids.ne(self.pad_id)
        text_inputs["attention_mask"] = attention_mask
        text_inputs["imgs_pos"] = [self.get_imgs_pos(input_ids)]
        # image_inputs["imgs"] = [[image_inputs["pixel_values"]]]

        return_tensors = kwargs.pop("return_tensors", None)
        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        assert 0

    def apply_chat_template(self, *args, **kwargs):
        token_ids = self.tokenizer.apply_chat_template(*args, **kwargs)
        return token_ids

    def get_imgs_pos(self, doc_ids):
        doc_ids = np.array(doc_ids, dtype=np.int64)
        img_begin_index = np.where(doc_ids == self.im_start_token_id)[0]
        img_end_index = np.where(doc_ids == self.im_end_token_id)[0]
        imgs_pos = np.concatenate(
            (
                np.reshape(img_begin_index + 1, (-1, 1)),
                np.reshape(img_end_index, (-1, 1)),
            ),
            axis=-1,
        ).tolist()
        return imgs_pos

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


def split_image_into_patch_blocks(
    pixel_values: torch.Tensor,  # shape: [batch_size, 3, H, W]
    patch_size: int = 16,  # e.g. 16
    adaptor_patch_div: int = 4,  # e.g. 4 --> each patch_size is cut into 4x4 small regions, i.e. patch_size // 4 # noqa: E501
) -> torch.Tensor:
    """
    Split the input image tensor (supporting batch) into large patches of size `patch_size`,
    and then further divide each large patch into smaller regions of size
    (patch_size // adaptor_patch_div) x (patch_size // adaptor_patch_div).
    Each small region is extracted as a tensor of shape [3, patch_size, patch_size].
    The final output contains all such small region tensors.

    Args:
        pixel_values: Input image tensor of shape [batch_size, 3, H, W].
        patch_size: Size of the large patch, e.g., 16.
        adaptor_patch_div: Each large patch is divided into
                          (patch_size // adaptor_patch_div) x (patch_size // adaptor_patch_div)
                          smaller regions.

    Returns:
        patches: A tensor of shape [N, 3, patch_size, patch_size],
                 where N = batch_size * (H // patch_size) * (W // patch_size) * (patch_size // adaptor_patch_div)^2.
                 Each element in the batch corresponds to one small image region.
    """  # noqa: E501
    batch_size, channels, height, width = pixel_values.shape
    assert channels == 3, "Pixel values must have 3 channels in dim=1"
    assert height % patch_size == 0 and width % patch_size == 0, (
        "H and W must be divisible by patch_size"
    )

    patch_height_num = height // patch_size
    patch_width_num = width // patch_size

    # Reshape to [B, 3, ph, ps, pw, ps]
    img = pixel_values.reshape(
        batch_size, 3, patch_height_num, patch_size, patch_width_num, patch_size
    )

    # Further split each psxps patch into (ps//aps)x(ps//aps) small regions
    img = img.reshape(
        batch_size,
        3,
        patch_height_num,
        patch_size // adaptor_patch_div,  # ps // aps
        adaptor_patch_div,
        patch_width_num,
        patch_size // adaptor_patch_div,  # ps // aps
        adaptor_patch_div,
    )

    # Permute to group the small regions: [B, ph, pw, ps//aps, ps//aps, 3, aps, aps]
    img = img.permute(0, 2, 5, 3, 6, 1, 4, 7)

    # Reshape into [B * ph * pw * (ps//aps)^2, 3, patch_size, patch_size]
    patches = img.reshape(-1, 3, patch_size, patch_size)

    return patches


AutoProcessor.register("HunYuanVLProcessor", HunYuanVLProcessor)
