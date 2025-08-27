# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
# from ....conftest import ImageTestAssets
from PIL import Image
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.models.radio import RadioModel

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]


def map_hf_radio_to_vllm_intern(hf_sd: dict, radio_vllm) -> dict:
    mapped = {}
    for k, v in hf_sd.items():
        if not k.startswith("radio_model."):
            continue
        k2 = k[len("radio_model."):]

        # skip buffers not used in vLLM
        if k2 in {"summary_idxs"}:
            continue

        # patch generator: keep same after stripping prefix
        if k2.startswith("model.patch_generator."):
            mapped_key = f"model.patch_generator.{k2.split('.', 2)[-1]}"
            mapped[mapped_key] = v
            continue

        # input conditioner
        if k2.startswith("input_conditioner."):
            mapped_key = f"input_conditioner.{k2.split('.', 1)[-1]}"
            mapped[mapped_key] = v
            continue

        # blocks -> encoder.layers
        if k2.startswith("model.blocks."):
            parts = k2.split(".")
            layer_idx = parts[2]
            suffix = ".".join(
                parts[3:]
            )  # e.g. norm1.weight, attn.qkv.weight, mlp.fc1.weight, etc.
            # ls1/ls2 do not exist in HF (Identity); vLLM has params – leave them default
            if suffix in {"ls1", "ls2"} or suffix.startswith(("ls1.", "ls2.")):
                continue
            mapped_key = f"model.encoder.layers.{layer_idx}.{suffix}"
            mapped[mapped_key] = v
            continue

    return mapped


VIT_DIMS = {
    "vit_small_patch16_224": (384, 12, 6, 1536),
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}


def get_args_from_model_type(model_type):
    return VIT_DIMS[model_type]


@torch.inference_mode()
def _test_radio_vllm_vs_hf():
    hf_repo = "nvidia/C-RADIOv2-H"

    # Init single-process distributed + model parallel so InternVisionModel can construct
    import tempfile
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend=backend,
    )
    initialize_model_parallel(1, 1)

    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
    config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)

    hf_model = AutoModel.from_pretrained(hf_repo,
                                         config=config,
                                         trust_remote_code=True)
    hf_model.eval().cuda()

    images = [
        Image.open('/home/dafrimi/projects/vllm/images/horse.jpg').convert(
            'RGB')
    ]

    pixel_values = [
        image_processor(images, return_tensors='pt').pixel_values.to(
            hf_model.dtype)[:, :, :432, :640] for images in images
    ]


    hf_outputs_per_image = [
        hf_model(pixel_value.to("cuda")).features
        for pixel_value in pixel_values
    ]

    try:
        hidden_size, num_layers, num_heads, intermediate_size = get_args_from_model_type(
            config.args["model"])
        config.num_hidden_layers = num_layers
        config.hidden_size = hidden_size
        config.num_attention_heads = num_heads
        config.intermediate_size = intermediate_size
        config.norm_type = "layer_norm"
        config.image_size = 224
        config.hidden_act = "gelu"
        config.layer_norm_eps = 1e-6
        config.initializer_factor = 1.0
        config.qkv_bias = True
        config.qk_normalization = False
        config.max_img_size = 2048
        config.reg_tokens = config.args["register_multiple"]

        radio_vllm = RadioModel(config).to("cuda")

        hf_state_dict = hf_model.state_dict()
        vllm_state_dict = map_hf_radio_to_vllm_intern(hf_state_dict,
                                                      radio_vllm)
        missing, unexpected = radio_vllm.load_state_dict(vllm_state_dict,
                                                         strict=False)
        print(f"missing: {missing}")
        print(f"unexpected: {unexpected}")

        vllm_outputs_per_image = [
            radio_vllm(pixel_values=pixel_value.to("cuda"))
            for pixel_value in pixel_values
        ]

        cos_similar = nn.CosineSimilarity(dim=-1)
        for vllm_output, timm_output in zip(vllm_outputs_per_image,
                                            hf_outputs_per_image):
            assert cos_similar(vllm_output, timm_output).mean() > 0.99
            print(cos_similar(vllm_output, timm_output).mean())
            print("PASSED")
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    _test_radio_vllm_vs_hf()
