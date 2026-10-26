# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm import SamplingParams
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget


def _observe_encoder_tokens(model):
    model.encoder_token_counts = []

    def record(module, args):
        model.encoder_token_counts.append(
            args[0].shape[0] // module.spatial_merge_size**2
        )

    model.visual.register_forward_pre_hook(record)


@pytest.mark.core_model
@pytest.mark.parametrize("model", ["Qwen/Qwen3-VL-2B-Instruct"])
@pytest.mark.parametrize("pruning_rate", [0.0, 0.75])
def test_video_pruning_respects_encoder_budget(
    vllm_runner, model, pruning_rate, monkeypatch
):
    # Enqueue every request before scheduling to make the batch deterministic.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    prompt = (
        "<|im_start|>user\n<|video_pad|>"
        "Describe the video in one sentence.<|im_end|>\n<|im_start|>assistant\n"
    )
    metadata = {
        "fps": 2.0,
        "duration": 8.0,
        "total_num_frames": 16,
        "frames_indices": list(range(16)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "video": [(np.full((16, 256, 256, 3), i * 30, np.uint8), metadata)]
            },
        }
        for i in range(8)
    ]
    with vllm_runner(
        model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 1},
        video_pruning_rate=pruning_rate,
        mm_processor_cache_gb=0,
        mm_processor_kwargs={
            "videos_kwargs": {
                "size": {"shortest_edge": 65536, "longest_edge": 1048576},
                "num_frames": 16,
            }
        },
    ) as runner:
        llm = runner.get_llm()
        budget = MultiModalBudget(
            llm.llm_engine.vllm_config, MULTIMODAL_REGISTRY
        ).encoder_compute_budget
        runner.apply_model(_observe_encoder_tokens)
        outputs = llm.generate(inputs, SamplingParams(temperature=0, max_tokens=16))
        counts = runner.apply_model(lambda model: model.encoder_token_counts)[0]

        assert len(outputs) == len(inputs)
        assert all(output.outputs[0].token_ids for output in outputs)
        assert sum(counts) > budget  # The workload requires multiple encoder steps.
        assert max(counts) <= budget, (counts, budget)
