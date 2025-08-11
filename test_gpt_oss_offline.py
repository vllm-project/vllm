import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.llm import LLM
import torch
import numpy as np

MODEL_PATH = "unsloth/gpt-oss-20b-BF16"
original_output = "Roses are red, violets are blue, I love you, and I love you too!"
original_logprobs = [
        -0.037353515625,
        -0.08154296875,
        -1.21875,
        -1.953125,
        -2.234375,
        -0.96875,
        -1.546875,
        -1.640625,
        -0.93359375,
        -1.609375,
        -1.625,
        -0.85546875,
        -1.7265625,
    ]

def do_sample(llm: LLM) -> list[str]:
    prompts = [
        "Roses are red, violets",
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          logprobs=1,)
    outputs = llm.generate(
        prompts,
        sampling_params)

    # Print the outputs.
    generated_texts: list[str] = []
    logprobs: list[float] = []
    for output in outputs:
        for probs in output.outputs[0].logprobs:
            logprobs.append(list(probs.values())[0].logprob)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    assert prompts[0]+generated_texts[0] == original_output, "Generated text does not match the expected output."
    assert np.allclose(np.array(logprobs[:-1]),np.array(original_logprobs),rtol=1e-01, atol=1e-01), "Logprobs do not match the expected values."
    return generated_texts


if __name__ == "__main__":
    llm = LLM(MODEL_PATH,
                    max_num_seqs=8,
                    dtype='bfloat16',
                    enforce_eager=False,
                    max_model_len=20,
                    max_num_batched_tokens=512,
                    )

    do_sample(llm)
