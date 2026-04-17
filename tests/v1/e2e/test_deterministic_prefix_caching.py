"""
Test deterministic prefix caching.

When --deterministic-prefix-caching is enabled, cache-miss and cache-hit
prefills should produce identical outputs by ensuring the suffix GEMM
uses the same batch dimension regardless of cache state.

Requires: ROCm or CUDA GPU, prefix caching support.
"""
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_RUNS = 5


@pytest.fixture(scope="module")
def server_with_deterministic_pc():
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--enable-prefix-caching",
        "--deterministic-prefix-caching",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def generate_n_times(client, n, prompt, max_tokens=10):
    outputs = []
    for _ in range(n):
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        outputs.append(response.choices[0].text)
    return outputs


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="Requires GPU",
)
class TestDeterministicPrefixCaching:

    def test_cache_miss_equals_cache_hit(self, server_with_deterministic_pc):
        client = server_with_deterministic_pc.get_openai_client()
        prompt = (
            "The European Union is a political and economic union of "
            "member states that are located primarily in Europe. The EU "
            "has developed an internal single market through a "
            "standardised system of laws. Summarize in one sentence:"
        )
        outputs = generate_n_times(client, NUM_RUNS, prompt)
        assert len(set(outputs)) == 1, (
            f"Expected identical outputs, got {len(set(outputs))} unique: {outputs}"
        )

    def test_different_prompts_different_outputs(self, server_with_deterministic_pc):
        client = server_with_deterministic_pc.get_openai_client()
        out_a = generate_n_times(client, 1, "What is the capital of France?")[0]
        out_b = generate_n_times(client, 1, "What is the speed of light?")[0]
        assert out_a != out_b

    def test_shared_prefix_determinism(self, server_with_deterministic_pc):
        client = server_with_deterministic_pc.get_openai_client()
        shared = (
            "In the year 2025 artificial intelligence systems became "
            "capable of performing complex reasoning tasks and the field "
            "of machine learning experienced rapid advancement in "
            "several key areas including natural language processing. "
        )
        out_1 = generate_n_times(client, 2, shared + "What happened next?")
        out_2 = generate_n_times(client, 2, shared + "Why did this matter?")
        assert out_1[0] == out_1[1], f"Prompt 1 non-deterministic: {out_1}"
        assert out_2[0] == out_2[1], f"Prompt 2 non-deterministic: {out_2}"
