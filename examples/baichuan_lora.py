from vllm import LLM, SamplingParams
import time
if __name__ == "__main__":
    prompt = "Hello and welcome, "
    prompts = [prompt]
    path = "./baichuan2-13b"
    lora_path = "./baichuan2-13b-20231013174626"
    lora_path_2 = "./baichuan2-13b-20231013192059"
    llm = LLM(model=path,
              trust_remote_code=True,
              lora_paths=[lora_path, lora_path_2],
              adapter_names=["adapter_1", "adapter_2"])

    print(llm.llm_engine.workers[0].model)

    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=2,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=True,
                                     lora_id="adapter_1")
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)
    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=2,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=True,
                                     lora_id="adapter_2")
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)
    start = time.time()
    outputs = llm._run_engine(use_tqdm=True)
    end = time.time()
    print(f"cost: {end - start}")
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
