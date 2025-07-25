# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModel, SiglipImageProcessor

from vllm.assets.image import ImageAsset
from vllm.distributed import (destroy_model_parallel,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.models.siglip_so400m import SiglipSo400mVisionModel

MODEL_ID = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"


@pytest.mark.core_model
@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_model_correctness(model_id: str, dtype: str):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this device.")

    torch_dtype = getattr(torch, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        pytest.skip("This correctness test requires a GPU environment.")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"  # Use a unique port

    backend = "nccl" if device == "cuda" else "gloo"
    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method="env://",
                                 backend=backend)
    initialize_model_parallel(tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1)

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_class = AutoModel._model_mapping[type(config)]
        model_class._supports_sdpa = True

        # Load Original Hugging Face Model
        hf_model = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa").vision_model.to(device)
        hf_model.eval()

        vllm_model = SiglipSo400mVisionModel(hf_model.config).to(
            device, dtype=torch_dtype)
        vllm_model.load_weights(hf_model.state_dict().items())
        vllm_model.eval()

        hf_processor = SiglipImageProcessor(do_resize=True,
                                            size={
                                                "height": 980,
                                                "width": 980
                                            },
                                            resample=3,
                                            do_rescale=True,
                                            rescale_factor=1 / 255.0,
                                            do_normalize=True,
                                            image_mean=[0.5, 0.5, 0.5],
                                            image_std=[0.5, 0.5, 0.5])

        image = ImageAsset("cherry_blossom").pil_image
        pixel_values = hf_processor(images=image,
                                    return_tensors="pt").pixel_values.to(
                                        device, dtype=torch_dtype)

        with torch.no_grad():
            hf_output = hf_model(pixel_values=pixel_values,
                                 output_hidden_states=False).pooler_output
            vllm_output = vllm_model(pixel_values=pixel_values)

        assert torch.allclose(hf_output, vllm_output, atol=1e-3, rtol=1e-3)

        print(f"Correctness test passed for {model_id} on {device}")

    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
