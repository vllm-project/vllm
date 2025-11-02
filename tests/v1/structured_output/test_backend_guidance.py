# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "gpt2"


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
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    dummy_wrong = tokenizer.encode('{"a"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy_wrong)
    # Rollback is done from where state was terminated, so from '}' not EOS
    grammar.rollback(len(prompt) - 1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.validate_tokens(dummy_wrong) != dummy_wrong
    assert grammar.accept_tokens("", prompt[1:])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Rollback of <= 0 should not change the terminated state
    grammar.rollback(0)
    assert grammar.is_terminated()
    grammar.rollback(-1)
    assert grammar.is_terminated()


def test_grammar_bitmask_with_specdec():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    for i in range(1, 2):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                json='{"type": "object"}',
            ),
        )
        sampling_params.structured_outputs._backend = "guidance"

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=tokenizer.eos_token_id,
        )

        structured_output_manager.grammar_init(request)

        def grammar_bitmask(req: Request, tokens: list[int]) -> None:
            structured_output_manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids={req.request_id: 0},
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert not req.structured_output_request.grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion():
            continue

        assert request.structured_output_request.grammar.accept_tokens(
            request.request_id, prompt[:i]
        )

        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
        grammar_bitmask(
            request, prompt[i:] + [tokenizer.eos_token_id] + prompt
        )  # EOS not the final token
        grammar_bitmask(request, prompt[i:])  # EOS not present
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
