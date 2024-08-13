"""Tests whether draft model quantization can be loaded from SpeculativeConfig.

Run `pytest tests/spec_decode/test_configs.py --forked`.
"""

from typing import List, Tuple

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig


# Model Id // Quantization Arg // Expected Type
MODEL_ARG_EXPTYPES = [
    # The expected type could be different for various device capabilities.
    ("TheBloke/Llama-2-7B-Chat-GPTQ", None, ["gptq_marlin", "gptq"]),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "gptq", ["gptq"]),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "awq", ["ERROR"]),
]

@pytest.mark.parametrize("model_arg_exptype", MODEL_ARG_EXPTYPES)
def test_speculative_model_quantization_config(
    model_arg_exptype: Tuple[str, None, List[str]]
) -> None:
    model_path, quantization_arg, expected_type = model_arg_exptype

    try:
        target_model_name = 'JackFram/llama-68m'
        target_model_config = ModelConfig(
            model=target_model_name,
            tokenizer=target_model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
            revision=None,
            quantization=None)
        target_parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        
        spec_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=target_model_config,
            target_parallel_config=target_parallel_config,
            target_dtype="float16",
            speculative_model=model_path,
            speculative_model_quantization=quantization_arg,
            speculative_draft_tensor_parallel_size=1,
            num_speculative_tokens=3,
            speculative_max_model_len=2048,
            speculative_disable_by_batch_size=None,
            enable_chunked_prefill=False,
            use_v2_block_manager=True,
            disable_log_stats=True,
            ngram_prompt_lookup_max=None,
            ngram_prompt_lookup_min=None,
            draft_token_acceptance_method="rejection_sampler",
            typical_acceptance_sampler_posterior_threshold=None,
            typical_acceptance_sampler_posterior_alpha=None,
            disable_logprobs=None)
        draft_model_config = spec_config.draft_model_config
        found_quantization_type = draft_model_config.quantization
    except ValueError:
        found_quantization_type = "ERROR"

    assert found_quantization_type in expected_type, (
        f"Expected quant_type in {expected_type} for {model_path}, "
        f"but found {found_quantization_type} "
        f"for no --quantization {quantization_arg} case")
