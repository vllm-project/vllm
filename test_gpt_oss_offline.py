import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.llm import LLM

MODEL_PATH = "unsloth/gpt-oss-20b-BF16"
original_output = "Roses are red, violets are blue, I love you, and I love you too!"

def do_sample(llm: LLM) -> list[str]:
    prompts = [
        "Roses are red, violets",
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,)
    outputs = llm.generate(
        prompts,
        sampling_params)

    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    assert prompts[0]+generated_texts[0] == original_output, "Generated text does not match the expected output."
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
