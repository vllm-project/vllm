import sys

from vllm import LLM, SamplingParams


def test_lazy_outlines(sample_regex):
    """If users don't use guided decoding, outlines should not be imported.
    """
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="facebook/opt-125m",
              enforce_eager=True,
              gpu_memory_utilization=0.3)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # make sure outlines is not imported
    assert 'outlines' not in sys.modules

    llm = LLM(model="facebook/opt-125m",
              enforce_eager=True,
              guided_decoding_backend="lm-format-enforcer",
              gpu_memory_utilization=0.3)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(
        prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2,
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options_request=dict(guided_regex=sample_regex))

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # make sure outlines is not imported
    assert 'outlines' not in sys.modules
