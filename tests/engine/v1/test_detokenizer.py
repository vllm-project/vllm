import pytest

from typing import List
from transformers import AutoTokenizer

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.detokenizer import Detokenizer, DetokenizerRequest

TOKENIZER_NAME="mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

FULL_STRINGS=[
    "My name is Robert from Neural Magic and I love working on vLLM so much!",
    # "Red Hat is the best open source company by far across Linux, K8s, and AI.",
    # "Nick is the name of my brother in addition to my colleague from Red Hat.",
]
FULL_TOKENS = [tokenizer(text).input_ids for text in FULL_STRINGS]
PROMPT_LEN=5
PROMPT_TOKENS = [tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS]
GENERATION_TOKENS = [tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS]
PROMPT_STRINGS=[tokenizer.decode(prompt_tokens, skip_special_tokens=True) 
                for prompt_tokens in PROMPT_TOKENS]
PROMPT_STRINGS_LEN=[len(prompt_string) for prompt_string in PROMPT_STRINGS]
GENERATION_STRINGS=[text[prompt_len:] for text, prompt_len in zip(FULL_STRINGS, PROMPT_STRINGS_LEN)]


class MockEngineCore:
    """Mock outputs form premade tokens lists."""

    def __init__(self, tokens_list: List[List[int]]):
        self.tokens_list = tokens_list
        self.current_idx = 0
    
    def get_outputs(self) -> List[EngineCoreOutput]:
        token_idx = self.current_idx
        self.current_idx += 1

        outputs = []
        for req_idx, token_ids in enumerate(self.tokens_list):
            if len(token_ids) > token_idx:
                output = EngineCoreOutput(
                    request_id=f"request-{req_idx}",
                    new_token_ids=[token_ids[token_idx]],
                    finished=False)
                if token_idx == len(token_ids) - 1:
                    output.finished = True
                    output.finish_reason = "stopped"
                outputs.append(output)
        
        return outputs


pytest.mark.parametrize()
def test_delta_outputs():
    detokenizer = Detokenizer(TOKENIZER_NAME)
    engine_core = MockEngineCore(GENERATION_TOKENS)

    # Make N requests.
    requests = [
        DetokenizerRequest(
            request_id=f"request-{idx}",
            prompt=prompt,
            prompt_token_ids=prompt_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind=RequestOutputKind.DELTA,
            stop=[],
            include_stop_str_in_output=False,
        ) for idx, (prompt, prompt_tokens) in enumerate(zip(PROMPT_STRINGS, PROMPT_TOKENS))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        detokenizer.add_request(request)

    generation_strings = {}
    generation_tokens = {}
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break
        
        # Step the Detokenizer.
        request_outputs, request_to_abort = detokenizer.step(outputs)
        assert len(request_to_abort) == 0

        for request_output in request_outputs:
            request_id = request_output.request_id
            if request_id not in generation_strings:
                generation_strings[request_id] = request_output.outputs[0].text
                generation_tokens[request_id] = request_output.outputs[0].token_ids
            else:
                generation_strings[request_id] += request_output.outputs[0].text
                generation_tokens[request_id].extend(request_output.outputs[0].token_ids)
    
    for idx, (ref_gen_str, ref_gen_tokens) in enumerate(zip(GENERATION_STRINGS, GENERATION_TOKENS)):
        gen_str = generation_strings[f"request-{idx}"]
        gen_tokens = generation_tokens[f"request-{idx}"]
        
        assert gen_str == ref_gen_str, f"{gen_str=}, {ref_gen_str=}"
        assert gen_tokens == ref_gen_tokens, f"{gen_tokens=}, {ref_gen_tokens=}"
