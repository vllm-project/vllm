import pytest
import torch
from vllm import LLM, SamplingParams

pytestmark = pytest.mark.cuda
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MODEL_ID = "facebook/opt-125m"


def _gen_ids(llm: LLM, prompt: str):
    out = llm.generate(prompt, SamplingParams(max_tokens=24, temperature=0.0, seed=123))
    return out[0].outputs[0].token_ids


def test_last_layer_equals_baseline():
    prompt = "The quick brown fox"

    base = LLM(model=MODEL_ID, tensor_parallel_size=1, gpu_memory_utilization=0.35)
    base_ids = _gen_ids(base, prompt)

    # OPT-125M has 12 layers (0..11) in most builds; adjust if needed
    ls = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        speculative_config={"method": "layer_skip", "layer_skip": 11, "num_speculative_tokens": 3},
    )
    ls_ids = _gen_ids(ls, prompt)

    assert base_ids == ls_ids, f"Last-layer exit must equal baseline.\nBase: {base_ids}\nLS: {ls_ids}"


def test_layer_skip_smoke_mid_layer():
    llm = LLM(
        model=MODEL_ID,
        speculative_config={"method": "layer_skip", "layer_skip": 6, "num_speculative_tokens": 4},
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
    )
    outputs = llm.generate(
        ["Hello world", "The capital of"],
        SamplingParams(max_tokens=12, temperature=0.8, seed=7),
    )
    assert len(outputs) == 2
    for o in outputs:
        assert len(o.outputs[0].token_ids) <= 12
        assert len(o.outputs[0].text) > 0