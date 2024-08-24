import sys

from vllm import LLM, SamplingParams


def test_lazy_outlines():
    """If users don't use guided decoding, outlines should not be imported.
    """
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # check if outlines is imported
    assert 'outlines' not in sys.modules
