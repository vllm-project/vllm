# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoTokenizer

from vllm.config import DecodingConfig, VllmConfig
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
        decoding_config=DecodingConfig(
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
    grammar.accept_tokens("test", prompt + [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    grammar.rollback(1)
    assert not grammar.is_terminated()
