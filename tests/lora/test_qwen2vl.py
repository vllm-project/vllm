# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform


@dataclass
class TestConfig:
    model_path: str
    lora_path: str
    max_num_seqs: int = 2
    max_loras: int = 2
    max_lora_rank: int = 16
    max_model_len: int = 4096
    mm_processor_kwargs: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.mm_processor_kwargs is None:
            self.mm_processor_kwargs = {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }


class Qwen2VLTester:
    """Test helper for Qwen2 VL models with LoRA"""

    PROMPT_TEMPLATE = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is in the image?<|im_end|>\n"
        "<|im_start|>assistant\n")

    def __init__(self, config: TestConfig):
        self.config = config
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> vllm.LLM:
        """Initialize the LLM with given configuration"""
        return vllm.LLM(
            model=self.config.model_path,
            max_num_seqs=self.config.max_num_seqs,
            enable_lora=True,
            max_loras=self.config.max_loras,
            max_lora_rank=self.config.max_lora_rank,
            trust_remote_code=True,
            mm_processor_kwargs=self.config.mm_processor_kwargs,
            max_model_len=self.config.max_model_len,
        )

    def run_test(self,
                 images: List[ImageAsset],
                 expected_outputs: List[str],
                 lora_id: Optional[int] = None,
                 temperature: float = 0,
                 max_tokens: int = 5) -> List[str]:

        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        inputs = [{
            "prompt": self.PROMPT_TEMPLATE,
            "multi_modal_data": {
                "image": asset.pil_image
            },
        } for asset in images]

        lora_request = LoRARequest(str(lora_id), lora_id,
                                   self.config.lora_path)
        outputs = self.llm.generate(inputs,
                                    sampling_params,
                                    lora_request=lora_request)
        generated_texts = [
            output.outputs[0].text.strip() for output in outputs
        ]

        # Validate outputs
        for generated, expected in zip(generated_texts, expected_outputs):
            assert expected.startswith(
                generated), f"Generated text {generated} doesn't "
            f"match expected pattern {expected}"

        return generated_texts


TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

EXPECTED_OUTPUTS = [
    "A red stop sign stands prominently in the foreground, with a traditional Chinese gate and a black SUV in the background, illustrating a blend of modern and cultural elements.",  # noqa: E501
    "A majestic skyscraper stands tall, partially obscured by a vibrant canopy of cherry blossoms, against a clear blue sky.",  # noqa: E501
]

QWEN2VL_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25VL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="Qwen2-VL dependency xformers incompatible with ROCm")
def test_qwen2vl_lora(qwen2vl_lora_files):
    """Test Qwen 2.0 VL model with LoRA"""
    config = TestConfig(model_path=QWEN2VL_MODEL_PATH,
                        lora_path=qwen2vl_lora_files)
    tester = Qwen2VLTester(config)

    # Test with different LoRA IDs
    for lora_id in [1, 2]:
        tester.run_test(TEST_IMAGES,
                        expected_outputs=EXPECTED_OUTPUTS,
                        lora_id=lora_id)


@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="Qwen2.5-VL dependency xformers incompatible with ROCm",
)
@pytest.mark.skipif(
    Version(TRANSFORMERS_VERSION) < Version("4.49.0"),
    reason="Qwen2.5-VL require transformers version no lower than 4.49.0",
)
def test_qwen25vl_lora(qwen25vl_lora_files):
    """Test Qwen 2.5 VL model with LoRA"""
    config = TestConfig(model_path=QWEN25VL_MODEL_PATH,
                        lora_path=qwen25vl_lora_files)
    tester = Qwen2VLTester(config)

    # Test with different LoRA IDs
    for lora_id in [1, 2]:
        tester.run_test(TEST_IMAGES,
                        expected_outputs=EXPECTED_OUTPUTS,
                        lora_id=lora_id)
