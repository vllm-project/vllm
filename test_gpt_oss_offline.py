import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.llm import LLM
import numpy as np

RUN_20B_MODEL = True  # Set to False to run the 120B model instead
MODEL_PATH = "unsloth/gpt-oss-20b-BF16"
MODEL_PATH_120 = "unsloth/gpt-oss-120b-BF16"
# reference https://github.com/huggingface/transformers/blob/68eb1a9a6353911f491b1c8139eb73d052a8e9b9/tests/models/gpt_oss/test_modeling_gpt_oss.py#L397
original_output = "Roses are red, violets are blue, I love you, and I love you too!"
# reference https://github.com/huggingface/transformers/blob/68eb1a9a6353911f491b1c8139eb73d052a8e9b9/tests/models/gpt_oss/test_modeling_gpt_oss.py#L462
original_output_120 = "Roses are red, violets are blue,\nI am a language model, not a human being"
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
original_logprobs_120 = [
        -0.90234375,
        -0.66015625,
        -1.546875,
        -2.703125,
        -2.078125,
        -1.21875,
        -2.484375,
        -0.031982421875,
        -0.84765625,
        -1.890625,
        -0.1923828125,
        -2.046875,
        -1.65625,
    ]

def do_sample(llm: LLM, original_output: str, original_logprobs: list[float], rtol: float, atol: float ) -> list[str]:
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
    assert np.allclose(np.array(logprobs[:-1]),np.array(original_logprobs),rtol=rtol, atol=atol), "Logprobs do not match the expected values."
    return generated_texts


if __name__ == "__main__":
    if RUN_20B_MODEL:
        llm = LLM(MODEL_PATH,
                        max_num_seqs=8,
                        dtype='bfloat16',
                        enforce_eager=False,
                        max_model_len=20,
                        max_num_batched_tokens=512,
                        tensor_parallel_size=1,
                        )
        do_sample(llm, original_output=original_output,
                  original_logprobs=original_logprobs, rtol=1e-01, atol=1e-01)
    else:
        llm = LLM(MODEL_PATH_120,
                        max_num_seqs=8,
                        dtype='bfloat16',
                        enforce_eager=False,
                        max_model_len=20,
                        max_num_batched_tokens=512,
                        tensor_parallel_size=4,
                        )
        do_sample(llm, original_output=original_output_120,
                  original_logprobs=original_logprobs_120, rtol=1e-01, atol=3e-01)
