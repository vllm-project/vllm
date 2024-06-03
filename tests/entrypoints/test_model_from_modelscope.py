from vllm import LLM, SamplingParams

# model: https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary
MODEL_NAME = "qwen/Qwen1.5-0.5B-Chat"


def test_offline_inference(monkeypatch):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")
    llm = LLM(model=MODEL_NAME)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == 4
