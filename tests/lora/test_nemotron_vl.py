# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for Llama Nemotron Nano VL with tower/connector LoRA enabled.

`enable_tower_connector_lora=True` requires the model to implement
`get_num_mm_encoder_tokens`/`get_num_mm_connector_tokens`: the engine sizes
the tower/connector punica wrappers from them at startup and builds
per-request LoRA mappings from them for every scheduled image, even for
requests without an active adapter.
"""

import vllm
from tests.conftest import VllmRunner
from vllm.assets.image import ImageAsset

MODEL_PATH = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "<image>\nWhat is in the image?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

EXPECTED_KEYWORDS = [
    ("stop", "street", "gate", "sign"),
    ("blossom", "flower", "tree", "tower"),
]


def test_nemotron_vl_tower_connector_lora():
    """Serve image requests with tower/connector LoRA support enabled."""
    with VllmRunner(
        model_name=MODEL_PATH,
        max_num_seqs=2,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=32,
        enable_tower_connector_lora=True,
        # Currently, tower_connector_lora is incompatible with
        # the multi-modal processor cache.
        mm_processor_cache_gb=0,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    ) as runner:
        llm = runner.get_llm()
        sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
        inputs = [
            {
                "prompt": PROMPT_TEMPLATE,
                "multi_modal_data": {"image": asset.pil_image},
            }
            for asset in TEST_IMAGES
        ]

        outputs = llm.generate(inputs, sampling_params)

        generated_texts = [output.outputs[0].text.lower() for output in outputs]
        for generated, keywords in zip(generated_texts, EXPECTED_KEYWORDS):
            assert any(keyword in generated for keyword in keywords), (
                f"Expected one of {keywords!r} in generated text {generated!r}"
            )
