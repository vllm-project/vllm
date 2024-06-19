from typing import List, Tuple

import math
import torch
import pytest
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoConfig, AutoModel

from vllm import LLM, SamplingParams
from vllm.config import VisionLanguageConfig
from vllm.multimodal.image import ImagePixelData

from ..conftest import IMAGE_FILES


IMAGE_PROMPT = "What is in this image?"


def slice_image(image,
                max_slice_nums=9,
                scale_resolution=448,
                patch_size=14,
                never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution *
                                                scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    best_grid = None

    if multiple > 1 and not never_split:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

    return best_grid


def get_grid_placeholder(grid, query_num):
    image_placeholder = query_num + 2

    cols = grid[0]
    rows = grid[1]
    slices = 0
    for i in range(rows):
        lines = 0
        for j in range(cols):
            lines += image_placeholder
        if i < rows - 1:
            slices += lines + 1
        else:
            slices += lines
    slice_placeholder = 2 + slices
    return slice_placeholder


class MiniCPMV_VLLM:

    def __init__(self, model_name) -> None:
        self.config = AutoConfig.from_pretrained(model_name,
                                                 trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       trust_remote_code=True)
        self.llm = LLM(
            model=model_name,
            image_input_type="pixel_values",
            image_token_id=101,
            image_input_shape="1,3,448,448",
            image_feature_size=64,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
        )

    def get_slice_image_placeholder(self, image):
        image_placeholder = self.config.query_num + 2

        best_grid = slice_image(
            image,
            self.config.max_slice_nums,
            self.config.scale_resolution,
            self.config.patch_size,
        )
        final_placeholder = image_placeholder

        if best_grid is not None:
            final_placeholder += get_grid_placeholder(best_grid,
                                                      self.config.query_num)

        return final_placeholder - 1

    def generate(self, image, question, sampling_params):
        addtion_tokens = self.get_slice_image_placeholder(image)
        image = transforms.Compose([transforms.ToTensor()])(img=image)
        images = torch.stack([image])

        prompt = "<用户><image></image>" + \
            question + \
            "<AI>" + '<unk>' * addtion_tokens

        outputs = self.llm.generate({
                "prompt": prompt,
                "multi_modal_data": ImagePixelData(images),
            },
            sampling_params=sampling_params
        )
        return outputs[0].outputs[0].text, outputs[0].outputs[0].token_ids


model_names = [
    "openbmb/MiniCPM-V-2"
]


def get_hf_results(model_name, image, question):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    hf_model = hf_model.to(device='cuda', dtype=torch.bfloat16)
    hf_model.eval()
    msgs = [{'role': 'user', 'content': question}]
    outputs, _, _ = hf_model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=False
    )
    output_ids = tokenizer.encode(outputs)[1:]
    return outputs, output_ids


def get_vllm_results(model_name, image, question):
    model = MiniCPMV_VLLM(model_name)
    sampling_params = SamplingParams(
        use_beam_search=False,
        # length_penalty=1.2,
        # best_of=3,
        max_tokens=1024,
        temperature=0
    )
    outputs, output_ids = model.generate(image, question, sampling_params)
    return outputs, output_ids[:-1]


@pytest.mark.parametrize("model_name", model_names)
@pytest.mark.parametrize("image", IMAGE_FILES)
def test_models(model_name, image) -> None:
    if not torch.cuda.is_available():
        return
    image = Image.open(image).convert("RGB")
    hf_outputs, hf_output_ids = get_hf_results(model_name, image, IMAGE_PROMPT)
    vllm_outputs, vllm_output_ids = get_vllm_results(model_name, image, IMAGE_PROMPT)
    common_prefix_len = 0
    for x in range(min(len(hf_output_ids), len(vllm_output_ids))):
        if hf_output_ids[x] != vllm_output_ids[x]:
            break
        common_prefix_len += 1
