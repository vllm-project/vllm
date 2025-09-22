# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
end-to-end tests for context length corner cases of vLLM v1 model runner
versus HuggingFace's transformers.

This test verifies the following behavior: allow a prefill that fills the 
model's maximum context length and then request a single new token. 

Test strategy
- Build a textual prompt that tokenizes to exactly ``max_model_len`` tokens.
- Run vLLM generation requesting a single new token (max_tokens=1).
- Run HF generation on the same prompt requesting a single token too. 
- Assert both return the same number of generated tokens and the same ids.

"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@pytest.mark.parametrize("model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_tokens", [1])
def test_models(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    max_model_len: int,
    max_tokens: int,
) -> None:
    """Compare vLLM and HuggingFace when the prompt already fills the
    model's maximum context length and we request a single new token.

    The test ensures vLLM does not raise the "Sampled token IDs exceed the
    max model length" assertion and that both vLLM and HF produce the same
    single token when given the same inputs.
    """

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Construct a deterministic textual prompt that repeats a short token
        # sequence ("1 2 3 ... 9 ") until the resulting tokenized sequence has
        # length max_model_len.
        token_block = " ".join(str(i) for i in range(1, 10)) + " "
        repeats = max_model_len // len(token_block) + 1
        # accounting for BOS and EOS tokens
        long_text = (token_block * repeats)[:max_model_len - 2]
        prompts = [long_text]

        # Tokenize with HF tokenizer and verify we produced the expected number
        # of tokens.
        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt_ids = tokenizer(prompts)["input_ids"]
        len_prompt_ids = len(prompt_ids[0])
        assert len_prompt_ids == max_model_len, (
            f"Tokenized prompt length ({len_prompt_ids}) != "
            f"max model len ({max_model_len})")
        # Generate max_tokens new tokens deterministically.
        sampling_params = [
            SamplingParams(max_tokens=max_tokens,
                           temperature=0.0,
                           ignore_eos=True)
        ]

        vllm_token_prompts = [
            TokensPrompt(prompt_token_ids=p) for p in prompt_ids
        ]

        # --- vLLM generation ---
        with torch.no_grad():
            llm = LLM(
                model=model,
                tokenizer=model,
                max_num_seqs=1,
                tensor_parallel_size=1,
            )

            vllm_results = llm.generate(vllm_token_prompts, sampling_params)

        vllm_output_ids = vllm_results[0].outputs[0].token_ids

        # --- HuggingFace generation ---
        with torch.no_grad():
            hf_model = AutoModelForCausalLM.from_pretrained(model)

            # HF expects a tensor of input ids shaped (batch, seq_len).
            hf_input_tokens = torch.tensor(prompt_ids[0]).unsqueeze(0)

            # Generate max_tokens new tokens deterministically.
            hf_generated = hf_model.generate(
                hf_input_tokens,
                do_sample=False,
                min_new_tokens=max_tokens,
                max_new_tokens=max_tokens,
            )

            # HF returns the prompt + generated tokens. Slice off the prompt.
            hf_output_ids = hf_generated.cpu().tolist()[0][len_prompt_ids:]

        # Same number of tokens generated
        assert len(vllm_output_ids) == len(hf_output_ids), (
            "Different number of generated tokens: "
            f"vLLM={len(vllm_output_ids)} HF={len(hf_output_ids)}")
        # Same token ids generated
        assert vllm_output_ids == hf_output_ids, (
            f"Mismatch between vLLM and HF outputs: "
            f"{vllm_output_ids} != {hf_output_ids}")


if __name__ == "__main__":
    """
    Run tests locally for development.
    
    Usage:
        cd vllm/
        VLLM_USE_V1=1 python -m pytest tests/v1/e2e/test_context_length.py -v
    """
    pytest.main([__file__, "-v"])
