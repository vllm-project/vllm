import os

from vllm import LLM, SamplingParams


def test_compile_correctness():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    all_outputs = []
    all_levels = [0, 1, 2]
    for level in all_levels:
        os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(level)
        llm = LLM(model="meta-llama/Meta-Llama-3-8B",
                  enforce_eager=True,
                  tensor_parallel_size=1,
                  disable_custom_all_reduce=True,
                  gpu_memory_utilization=0.3)
        outputs = llm.generate(prompts, sampling_params)
        all_outputs.append(outputs)
    reference_outputs = all_outputs[0]
    for level, outputs in zip(all_levels[1:], all_outputs[1:]):
        for ref_output, output in zip(reference_outputs, outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            ref_generated_text = ref_output.outputs[0].text
            assert generated_text == ref_generated_text, f"level: {level}, prompt: {prompt}, generated_text: {generated_text}, ref_generated_text: {ref_generated_text}"  # noqa
