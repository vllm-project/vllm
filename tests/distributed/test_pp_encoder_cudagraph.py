# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import PretrainedConfig

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

from ..models.utils import dummy_hf_overrides
from ..utils import multi_gpu_test

MODEL = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n<|im_start|>assistant\n"
)


def pp_dummy_hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Shrink the model but keep one text layer per pipeline stage."""
    hf_config = dummy_hf_overrides(
        hf_config, model_arch="Qwen2VLForConditionalGeneration"
    )
    hf_config.get_text_config().update({"num_hidden_layers": 2})
    return hf_config


def probe_encoder_cudagraph(worker) -> dict[str, int | bool]:
    from vllm.distributed.parallel_state import get_pp_group

    manager = getattr(worker.model_runner, "encoder_cudagraph_manager", None)
    stats = manager.get_cumulative_stats() if manager is not None else {}
    return {
        "pp_rank": get_pp_group().rank_in_group,
        "present": manager is not None,
        "graph_hits": stats.get("graph_hits", 0),
        "graph_misses": stats.get("graph_misses", 0),
    }


@multi_gpu_test(num_gpus=2)
def test_encoder_cudagraph_only_on_first_pp_stage(monkeypatch: pytest.MonkeyPatch):
    # LLM.collective_rpc requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    llm = LLM(
        model=MODEL,
        load_format="dummy",
        hf_overrides=pp_dummy_hf_overrides,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.6,
        compilation_config={
            "cudagraph_mm_encoder": True,
            "cudagraph_capture_sizes": [1, 2],
            "encoder_cudagraph_token_budgets": [4096],
            "encoder_cudagraph_max_vision_items_per_batch": 1,
        },
    )

    outputs = llm.generate(
        [
            {
                "prompt": PROMPT,
                "multi_modal_data": {"image": ImageAsset("stop_sign").pil_image},
            }
        ],
        SamplingParams(temperature=0, max_tokens=8),
    )
    # Weights are dummy, so only the mechanics of generation are meaningful.
    assert outputs and outputs[0].outputs[0].text is not None

    probes = sorted(
        llm.collective_rpc(probe_encoder_cudagraph),
        key=lambda probe: probe["pp_rank"],
    )

    assert [probe["present"] for probe in probes] == [True, False]
    assert probes[0]["graph_hits"] >= 1
    assert probes[0]["graph_misses"] == 0
