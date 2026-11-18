# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.transformers_utils.processors.test_hy_v4 import (
    hy_v4_processor_dir,  # noqa: F401
)
from vllm import SamplingParams
from vllm.models.hy_v4 import HYV4VLForConditionalGeneration
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)


def _requests():
    image = Image.new("RGB", (64, 64), "red")
    image_prompt = HYV4VLForConditionalGeneration.get_placeholder_str("image", 0)
    video_prompt = HYV4VLForConditionalGeneration.get_placeholder_str("video", 0)
    video = np.stack([np.asarray(image)] * 2)
    metadata = {
        "total_num_frames": 2,
        "fps": 2.0,
        "frames_indices": [0, 1],
        "do_sample_frames": False,
    }
    return [
        {"prompt": "Describe a red square."},
        {
            "prompt": image_prompt + "Describe this image.",
            "multi_modal_data": {"image": image},
        },
        {
            "prompt": image_prompt * 2 + "Compare these images.",
            "multi_modal_data": {"image": [image, image]},
        },
        {
            "prompt": video_prompt + "Describe this video.",
            "multi_modal_data": {"video": (video, metadata)},
        },
    ]


@pytest.mark.parametrize("enforce_eager,tp", [(True, 1), (False, 1), (True, 2)])
def test_hy_v4_vl_small_model_generation(request, vllm_runner, enforce_eager, tp):
    """Exercise iHC/DSA/MoE prefill/decode, not accuracy, with small dummy weights."""
    if torch.accelerator.device_count() < tp:
        pytest.skip("Not enough GPUs for the requested parallel layout")
    path = request.getfixturevalue("hy_v4_processor_dir")
    config = HYV4VLConfig.from_pretrained(path)
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 256
    config.text_config.moe_intermediate_size = 128
    config.text_config.num_hidden_layers = 2
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 4
    config.text_config.q_lora_rank = 128
    config.text_config.n_routed_experts = 4
    config.text_config.num_experts_per_tok = 2
    # Preserve the checkpoint's indexer shape for AITER's tiled paged logits
    # kernel even when the rest of the dummy model is smaller.
    config.text_config.index_n_heads = 32
    config.text_config.index_topk = 128
    config.text_config.vocab_size = 128
    config.text_config.max_position_embeddings = 256
    config.text_config.layer_types = ["indexed_attention"] * 2
    config.text_config.mlp_layer_types = ["dense", "sparse"]
    config.text_config.indexer_types = ["full", "shared"]
    config.text_config.num_nextn_predict_layers = 0
    config.text_config.bos_token_id = 0
    config.text_config.eos_token_id = 1
    config.text_config.pad_token_id = 1
    config.vision_config.out_hidden_size = 128
    config.vision_config.text_hidden_size = 128
    config.save_pretrained(path)
    with vllm_runner(
        str(path),
        trust_remote_code=False,
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        block_size=64,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.1,
        limit_mm_per_prompt={"image": 2, "video": 1},
        enable_prefix_caching=False,
    ) as runner:
        outputs = runner.llm.generate(
            _requests(), SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        )
        assert len(outputs) == 4
        assert all(len(output.outputs[0].token_ids) == 2 for output in outputs)


def test_hy_v4_vl_checkpoint_generation(vllm_runner):
    """Full local checkpoint smoke test; a resource skip is not an accuracy result."""
    path = Path(HF_EXAMPLE_MODELS.get_hf_info("HYV4VLForConditionalGeneration").default)
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        pytest.skip("The internal HYV4 VL checkpoint is not available")
    weight_map = json.loads(index.read_text())["weight_map"]
    weight_bytes = sum(
        (path / name).stat().st_size for name in set(weight_map.values())
    )
    devices = torch.accelerator.device_count()
    tp = max((n for n in (1, 2, 4, 8, 16, 32, 64) if n <= devices), default=1)
    capacity = sum(torch.accelerator.get_memory_info(i)[0] for i in range(tp)) * 0.9
    if weight_bytes > capacity:
        pytest.skip(
            f"Full BF16 checkpoint needs {weight_bytes / 2**30:.1f} GiB "
            f"before KV/workspace; {tp} GPUs provide only "
            f"{capacity / 2**30:.1f} GiB usable"
        )
    with vllm_runner(
        str(path),
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        block_size=64,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        limit_mm_per_prompt={"image": 2, "video": 1},
    ) as runner:
        outputs = runner.llm.generate(
            _requests(), SamplingParams(temperature=0, max_tokens=16)
        )
        assert len(outputs) == 4
        assert all(output.outputs[0].token_ids for output in outputs)


def test_hy_v4_vl_checkpoint_vision_weights(dist_init):
    """Load the actual ViT/projector only; this does not validate the full LLM."""
    from collections import defaultdict

    from safetensors import safe_open

    from vllm.models.hy_v4.nvidia.vision import HYV4VisionTransformer
    from vllm.transformers_utils.processors.hy_v4 import (
        HYV4VLProcessor,
        vision_token_count,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    path = Path(HF_EXAMPLE_MODELS.get_hf_info("HYV4VLForConditionalGeneration").default)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip("The internal HYV4 VL checkpoint is not available")
    if torch.accelerator.get_memory_info()[0] < 8 * 2**30:
        pytest.skip("At least 8 GiB of free VRAM is required for the actual ViT")
    config = HYV4VLConfig.from_pretrained(path)
    weights = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    by_shard = defaultdict(list)
    for name, shard in weights.items():
        if name.startswith("vit."):
            by_shard[shard].append(name)

    def vision_weights():
        for shard, names in by_shard.items():
            with safe_open(path / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    yield name.removeprefix("vit."), handle.get_tensor(name)

    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        model = HYV4VisionTransformer(config.vision_config).eval()
    loaded = model.load_weights(vision_weights())
    assert loaded == set(dict(model.named_parameters()))
    processor = HYV4VLProcessor.from_pretrained(path)
    inputs = processor(images=[Image.new("RGB", (64, 64), "red")], return_tensors="pt")
    with torch.inference_mode():
        output = model(inputs["pixel_values"], inputs["image_grid_thw"].tolist())[0]
    count = vision_token_count(
        inputs["image_grid_thw"][0], config.vision_config, "image"
    )
    assert output.shape == (count, config.text_config.hidden_size)
    assert torch.isfinite(output).all()
