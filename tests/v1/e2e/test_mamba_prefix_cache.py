import pytest
from vllm import LLM, SamplingParams
import time
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


# def _fake_sample(self, logits: torch.Tensor | None, spec_decode_metadata: SpecDecodeMetadata | None) -> SamplerOutput:


def test_mamba_prefix_cache(monkeypatch: pytest.MonkeyPatch):
    MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    PROMPT_MULTIPLE = 6
    monkeypatch.setenv("VLLM_USE_LIGHTER_MAMBA_CACHE", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=30)
    prefix = (  # examples/offline_inference/prefix_caching.py
        "Your name is QQQQ "
        "You are an expert school principal, skilled in effectively managing "
        "faculty and staff. Draft 10-15 questions for a potential first grade "
        "Head Teacher for my K-12, all-girls', independent school that emphasizes "
        "community, joyful discovery, and life-long learning. The candidate is "
        "coming in for a first-round panel interview for a 8th grade Math "
        "teaching role. They have 5 years of previous teaching experience "
        "as an assistant teacher at a co-ed, public school with experience "
        "in middle school math teaching. "
    )
    prefix2 = "Based on these information, fulfill the following paragraph: "
    prompt = PROMPT_MULTIPLE * prefix + prefix2 + "Hello, my name is"
    # print('Prompt length:', )
    # for APC in [False, True]:
    for APC in [True]:
        engine = LLM(
            model=MODEL,
            enable_prefix_caching=APC,
            enforce_eager=True,
            block_size=288,
            speculative_config={
                "method": "qwen3_next_mtp",
                "num_speculative_tokens": 2,
            },
            hf_overrides={"num_hidden_layers": 8},
            seed=42,
        )
        for i in range(3):
            if i == 0:
                print("Warm-up")
            if i == 1:
                print("Measuring")
                start_time = time.time()
            outputs = engine.generate(prompt, sampling_params)
            print("APC:", APC, i, f"Generated text: {outputs[0].outputs[0].text!r}")
            # for m in engine.llm_engine.get_metrics():
            #     if 'vllm:prefix_cache_hits' in m.name:
            #         print(m.name, m.value)
        print("APC:", APC, "loop took --- %s seconds ---" % (time.time() - start_time))
