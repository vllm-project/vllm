import torch

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata


class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        logits.zero_()
        logits[:, 0] += 1.0
        return logits


def test_oot_registration():
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model="facebook/opt-125m")
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""
