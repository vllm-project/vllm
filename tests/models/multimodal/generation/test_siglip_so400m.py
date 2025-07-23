# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
from transformers import AutoModel, SiglipImageProcessor

from vllm.assets.image import ImageAsset
from vllm.distributed import (destroy_model_parallel,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.models.siglip_so400m import SiglipSo400mVisionModel

MODEL_ID = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_model_correctness(dtype: str):
    """
    This test verifies that the vLLM implementation of SiglipSo400mVisionModel
    produces a numerically identical output to the original Hugging Face model.
    This is the standard way to validate a new model implementation in vLLM.
    """
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this device.")

    torch_dtype = getattr(torch, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        torch_dtype = torch.float32

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # Use any free port

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="env://",
    )

    initialize_model_parallel(tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1)

    try:
        hf_model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            torch_dtype=torch_dtype).vision_model.to(device)
        hf_model.eval()

        hf_processor = SiglipImageProcessor(
            do_resize=True,
            size={
                "height": 980,
                "width": 980
            },
            resample=3,  # BILINEAR
            do_rescale=True,
            rescale_factor=1 / 255.0,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

        vllm_config = hf_model.config
        vllm_model = SiglipSo400mVisionModel(vllm_config).to(device,
                                                             dtype=torch_dtype)

        vllm_model.load_weights(hf_model.state_dict().items())
        vllm_model.eval()

        image = ImageAsset("cherry_blossom").pil_image
        pixel_values = hf_processor(images=image,
                                    return_tensors="pt").pixel_values.to(
                                        device, dtype=torch_dtype)

        with torch.no_grad():
            hf_output = hf_model(pixel_values=pixel_values,
                                 output_hidden_states=False).pooler_output
            vllm_output = vllm_model(pixel_values=pixel_values)

        atol = 1e-5 if device == 'cpu' else 1e-3
        rtol = 1e-5 if device == 'cpu' else 1e-3
        assert torch.allclose(hf_output, vllm_output, atol=atol, rtol=rtol)

        print(f"Correctness test passed for {MODEL_ID} on {device}")

    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
