
from typing import List, Optional
from transformers import PretrainedConfig

class MLPSpeculatorConfig(PretrainedConfig):
    model_type = "mlp_speculator"

    attribute_map = {
        "hidden_size": "emb_dim",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: List[int] = [5, 4, 3],
        n_candidates: int = 5,
        **kwargs
    ):
        """
        Initialize an MLPSpeculatorConfig

        Args:
            vocab_size: int
                the model vocab size
            emb_dim: int
                the model embedding dimension
            inner_dim: int
                the inner dimension of the model. If 0, will be the emb_dim.
            n_predict: int
                the number of lookaheads for the speculator
            top_k_tokens_per_head: List[int]
                Number of tokens to consider from each head when forming the candidate tree.
                For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
            n_candidates: int
                number of child candidates to create per sequence
        """
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        super().__init__(**kwargs)


from transformers import AutoConfig
from vllm import LLM, SamplingParams
AutoConfig.register("mlp_speculator", MLPSpeculatorConfig)

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
prompts = [template.format(prompt) for prompt in prompts]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="ibm-granite/granite-7b-instruct", use_v2_block_manager=True, enforce_eager=True, speculative_model="ibm-granite/granite-7b-instruct-accelerator", num_speculative_tokens=5)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
import time
outputs = llm.generate(prompts, sampling_params)
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()
print((end-start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
