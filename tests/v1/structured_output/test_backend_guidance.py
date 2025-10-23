# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(
            backend="guidance",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    dummy = tokenizer.encode("d")
    for token in prompt:
        assert grammar.accept_tokens("test", [token])
    assert grammar.is_terminated()
    # We are in a terminated state, giving EOS should be accepted
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy)
    # Rollback is done from where state was terminated, so from 'prompt[-1]'
    grammar.rollback(1)
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", prompt[-1:])
    assert grammar.is_terminated()
    grammar.rollback(len(prompt))
    assert not grammar.accept_tokens("", dummy)
