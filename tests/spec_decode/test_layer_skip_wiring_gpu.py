import pytest
import torch
from vllm import LLM, SamplingParams
from .conftest import get_proposer_runner_from_llm

pytestmark = pytest.mark.cuda
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MODEL_ID = "facebook/opt-125m"


def _wrap_layers_count_calls(model):
    """Wrap each transformer block .forward to count invocations."""
    m = model.model if hasattr(model, "model") else model
    layers = getattr(m, "layers", None) or getattr(getattr(m, "decoder", object), "layers", None)
    assert layers is not None, "Unsupported model structure for wiring test."

    call_count = {"n": 0}
    originals = []

    def make_wrapper(f):
        def wrapped(*args, **kwargs):
            call_count["n"] += 1
            return f(*args, **kwargs)
        return wrapped

    for layer in layers:
        f = layer.forward
        originals.append((layer, f))
        layer.forward = make_wrapper(f)

    def restore():
        for layer, f in originals:
            layer.forward = f

    return call_count, restore


def test_early_exit_executes_exactly_k_layers():
    exit_layer = 6
    expected = exit_layer + 1  # inclusive semantics (0..k)

    llm = LLM(
        model=MODEL_ID,
        speculative_config={"method": "layer_skip", "layer_skip": exit_layer, "num_speculative_tokens": 2},
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
    )
    runner = get_proposer_runner_from_llm(llm)
    # Count proposer-side early-exit layer calls
    calls, restore = _wrap_layers_count_calls(runner.model)

    try:
        _ = llm.generate("Hello", SamplingParams(max_tokens=1, temperature=0.0, seed=1))
    finally:
        restore()

    assert calls["n"] == expected, f"Expected {expected} layers, got {calls['n']}"