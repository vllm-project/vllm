import gc
import time
from typing import List

from vllm import LLM, SamplingParams


def time_generation(llm: LLM, prompts: List[str],
                    sampling_params: SamplingParams):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    latency_per_token = (end - start) / sum(
        [len(o.outputs[0].token_ids) for o in outputs])
    # Print the outputs.
    ret = []
    for output in outputs:
        generated_text = output.outputs[0].text
        ret.append(generated_text)
    return ret, latency_per_token


if __name__ == "__main__":

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=20)

    # Create an LLM without spec decoding
    print("==============Without speculation==================")
    llm = LLM(model="JackFram/llama-68m")

    ret_non_spec, latency_per_token_non_spec = time_generation(
        llm, prompts, sampling_params)

    del llm
    gc.collect()

    # Create an LLM with spec decoding
    print("==============With speculation=====================")
    llm = LLM(
        model="JackFram/llama-68m",
        speculative_model="abhigoyal/vllm-medusa-llama-68m-random",
        num_speculative_tokens=5,
        use_v2_block_manager=True,
    )

    ret_spec, latency_per_token_spec = time_generation(llm, prompts,
                                                       sampling_params)

    del llm
    gc.collect()
    print("================= Summary =====================")
    print("input is ", prompts, "\n")
    print("Non Spec Decode - latency_per_token is ",
          latency_per_token_non_spec)
    print("Generated Text is :", ret_non_spec, "\n")
    print("Spec Decode - latency_per_token is ", latency_per_token_spec)
    print("Generated Text is :", ret_spec)
