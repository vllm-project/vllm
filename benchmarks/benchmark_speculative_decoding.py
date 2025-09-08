import os
from vllm import LLM, SamplingParams

from icecream import install
install()

def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
SD = getenv("SD", False)
FR = getenv("FR", False)

# NousResearch model is identical to meta-llama/Meta-Llama-3-8B-Instruct but doesn't need Meta to grant permission
model_path: str = "NousResearch/Meta-Llama-3-8B-Instruct"
draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
draft_vocab_pruned: str = 'thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt'

def test_llm_engine():
    # init llm engine
    sd_config={"model": draft_model_path, "draft_tensor_parallel_size": 1, "num_speculative_tokens": 2, "method": "eagle"}
    llm = LLM(
        model=model_path,
        max_model_len=32, # use less memory
        tensor_parallel_size=1,
        speculative_config=sd_config | ({'draft_vocab_pruned': draft_vocab_pruned} if FR else {}) if SD else None,
        )

    # sample from llm
    prompts = ["The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print(llm.llm_engine.stat_logger)

if __name__ == '__main__':
   test_llm_engine()
