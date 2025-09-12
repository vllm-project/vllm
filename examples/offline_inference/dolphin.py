# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import os
from dataclasses import dataclass

import cv2
import numpy as np
import regex as re
from PIL import Image
from transformers import DonutProcessor

from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt
from vllm.multimodal.utils import fetch_image


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
@dataclass
class ImageDimensions:
    original_w: int
    original_h: int
    padded_w: int
    padded_h: int


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def map_to_original_coordinates(
    x1, y1, x2, y2, dims: ImageDimensions
) -> tuple[int, int, int, int]:
    try:
        top = (dims.padded_h - dims.original_h) // 2
        left = (dims.padded_w - dims.original_w) // 2
        orig_x1 = max(0, x1 - left)
        orig_y1 = max(0, y1 - top)
        orig_x2 = min(dims.original_w, x2 - left)
        orig_y2 = min(dims.original_h, y2 - top)
        if orig_x2 <= orig_x1:
            orig_x2 = min(orig_x1 + 1, dims.original_w)
        if orig_y2 <= orig_y1:
            orig_y2 = min(orig_y1 + 1, dims.original_h)
        return int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)
    except Exception as e:
        print(f"map_to_original_coordinates error: {str(e)}")
        return 0, 0, min(100, dims.original_w), min(100, dims.original_h)


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def adjust_box_edges(image, boxes: list[list[float]], max_pixels=15, threshold=0.2):
    if isinstance(image, str):
        image = cv2.imread(image)
    img_h, img_w = image.shape[:2]
    new_boxes = []
    for box in boxes:
        best_box = copy.deepcopy(box)

        def check_edge(img, current_box, i, is_vertical):
            edge = current_box[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            if is_vertical:
                line = binary[current_box[1] : current_box[3] + 1, edge]
            else:
                line = binary[edge, current_box[0] : current_box[2] + 1]
            transitions = np.abs(np.diff(line))
            return np.sum(transitions) / len(transitions)

        edges = [(0, -1, True), (2, 1, True), (1, -1, False), (3, 1, False)]
        current_box = copy.deepcopy(box)
        current_box[0] = min(max(current_box[0], 0), img_w - 1)
        current_box[1] = min(max(current_box[1], 0), img_h - 1)
        current_box[2] = min(max(current_box[2], 0), img_w - 1)
        current_box[3] = min(max(current_box[3], 0), img_h - 1)

        for i, direction, is_vertical in edges:
            best_score = check_edge(image, current_box, i, is_vertical)
            if best_score <= threshold:
                continue
            for step in range(max_pixels):
                current_box[i] += direction
                if i == 0 or i == 2:
                    current_box[i] = min(max(current_box[i], 0), img_w - 1)
                else:
                    current_box[i] = min(max(current_box[i], 0), img_h - 1)
                score = check_edge(image, current_box, i, is_vertical)
                if score < best_score:
                    best_score = score
                    best_box = copy.deepcopy(current_box)
                if score <= threshold:
                    break
        new_boxes.append(best_box)
    return new_boxes


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def process_coordinates(coords, padded_image, dims: ImageDimensions, previous_box=None):
    try:
        x1, y1 = int(coords[0] * dims.padded_w), int(coords[1] * dims.padded_h)
        x2, y2 = int(coords[2] * dims.padded_w), int(coords[3] * dims.padded_h)
        x1, y1, x2, y2 = (
            max(0, min(x1, dims.padded_w - 1)),
            max(0, min(y1, dims.padded_h - 1)),
            max(0, min(x2, dims.padded_w)),
            max(0, min(y2, dims.padded_h)),
        )
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)
        new_boxes = adjust_box_edges(padded_image, [[x1, y1, x2, y2]])
        x1, y1, x2, y2 = new_boxes[0]
        x1, y1, x2, y2 = (
            max(0, min(x1, dims.padded_w - 1)),
            max(0, min(y1, dims.padded_h - 1)),
            max(0, min(x2, dims.padded_w)),
            max(0, min(y2, dims.padded_h)),
        )
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)
        if previous_box is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            if (x1 < prev_x2 and x2 > prev_x1) and (y1 < prev_y2 and y2 > prev_y1):
                y1 = prev_y2
                y1 = min(y1, dims.padded_h - 1)
                if y2 <= y1:
                    y2 = min(y1 + 1, dims.padded_h)
        new_previous_box = [x1, y1, x2, y2]
        orig_x1, orig_y1, orig_x2, orig_y2 = map_to_original_coordinates(
            x1, y1, x2, y2, dims
        )
        return x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box
    except Exception as e:
        print(f"process_coordinates error: {str(e)}")
        orig_x1, orig_y1, orig_x2, orig_y2 = (
            0,
            0,
            min(100, dims.original_w),
            min(100, dims.original_h),
        )
        return 0, 0, 100, 100, orig_x1, orig_y1, orig_x2, orig_y2, [0, 0, 100, 100]


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def prepare_image(image) -> tuple[np.ndarray, ImageDimensions]:
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_h, original_w = image_cv.shape[:2]
        max_size = max(original_h, original_w)
        top = (max_size - original_h) // 2
        bottom = max_size - original_h - top
        left = (max_size - original_w) // 2
        right = max_size - original_w - left
        padded_image = cv2.copyMakeBorder(
            image_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        padded_h, padded_w = padded_image.shape[:2]
        dimensions = ImageDimensions(
            original_w=original_w,
            original_h=original_h,
            padded_w=padded_w,
            padded_h=padded_h,
        )
        return padded_image, dimensions
    except Exception as e:
        print(f"prepare_image error: {str(e)}")
        h, w = image.height, image.width
        dimensions = ImageDimensions(original_w=w, original_h=h, padded_w=w, padded_h=h)
        return np.zeros((h, w, 3), dtype=np.uint8), dimensions


# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def parse_layout_string(bbox_str):
    """Parse layout string using regular expressions"""
    pattern = r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
    matches = re.finditer(pattern, bbox_str)

    parsed_results = []
    for match in matches:
        coords = [float(match.group(i)) for i in range(1, 5)]
        label = match.group(5).strip()
        parsed_results.append((coords, label))

    return parsed_results


model_id = "ByteDance/Dolphin"

# The input image size for Dolphin is 896 x 896,
# and the patch_size is 4 x 4.
# Therefore, the initial number of patches is:
# Height: 896 / 4 = 224 patches
# Width: 896 / 4 = 224 patches

# The Dolphin model uses a staged downsampling approach,
# defined by the "depths": [2, 2, 14, 2] configuration.
# Before entering stages 2, 3, and 4, a "Patch Merging" operation is performed,
# which halves the feature map's dimensions (dividing both height and width by 2).
# Before Stage 2: The size changes from 224 x 224 to (224/2) x (224/2) = 112 x 112.
# Before Stage 3: The size changes from 112 x 112 to (112/2) x (112/2) = 56 x 56.
# Before Stage 4: The size changes from 56 x 56 to (56/2) x (56/2) = 28 x 28.

# Because vLLM needs to fill the image features with an encoder_prompt,
# and the encoder_prompt will have `<pad>` tokens added when tokenized,
# we need to construct an encoder_prompt with a length of 28 x 28 - 1 = 783.
encoder_prompt = "".join(["0"] * 783)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
)

processor = DonutProcessor.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    dtype="float16",
    max_num_seqs=8,
    hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path", type=str, default=None, help="Path to a local image file."
)
args = parser.parse_args()

if args.image_path:
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Error: File not found at {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
else:
    image = fetch_image(
        "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/0.jpg"
    )


prompt = "Parse the reading order of this document. "
decoder_prompt = f"<s>{prompt}<Answer/>"
decoder_prompt_tokens = TokensPrompt(
    prompt_token_ids=processor.tokenizer(decoder_prompt, add_special_tokens=False)[
        "input_ids"
    ]
)
enc_dec_prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt=TextPrompt(prompt=encoder_prompt, multi_modal_data={"image": image}),
    decoder_prompt=decoder_prompt_tokens,
)
layout_outputs = llm.generate(prompts=enc_dec_prompt, sampling_params=sampling_params)
layout_result_str = layout_outputs[0].outputs[0].text
print(f"Layout analysis output:\n{layout_result_str}")

padded_image, dims = prepare_image(image)
layout_results = parse_layout_string(layout_result_str)
text_table_elements = []
previous_box = None
reading_order = 0
for bbox_coords, label in layout_results:
    if label == "fig":
        continue
    try:
        x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = (
            process_coordinates(bbox_coords, padded_image, dims, previous_box)
        )
        cropped = padded_image[y1:y2, x1:x2]
        if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            prompt_ocr = (
                "Parse the table in the image. "
                if label == "tab"
                else "Read text in the image. "
            )
            text_table_elements.append(
                {
                    "crop": pil_crop,
                    "prompt": prompt_ocr,
                    "reading_order": reading_order,
                }
            )
        reading_order += 1
    except Exception as e:
        print(f"Error processing bbox (label: {label}): {str(e)}")
        continue

if text_table_elements:
    batch_prompts = []
    for elem in text_table_elements:
        decoder_prompt_str = f"<s>{elem['prompt']}<Answer/>"
        decoder_prompt_tokens = TokensPrompt(
            prompt_token_ids=processor.tokenizer(
                decoder_prompt_str, add_special_tokens=False
            )["input_ids"]
        )
        enc_dec_prompt = ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt=encoder_prompt, multi_modal_data={"image": elem["crop"]}
            ),
            decoder_prompt=decoder_prompt_tokens,
        )
        batch_prompts.append(enc_dec_prompt)
    batch_outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)
    for i, output in enumerate(batch_outputs):
        text_table_elements[i]["text"] = output.outputs[0].text.strip()

print("------" * 8)
text_table_elements.sort(key=lambda x: x["reading_order"])
for elem in text_table_elements:
    print(elem.get("text", ""))
