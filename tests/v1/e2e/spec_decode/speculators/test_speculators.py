# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ..utils import (
    assert_request_outputs_match,
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)


@pytest.mark.parametrize(
    ["model_path", "expected_accuracy_threshold"],
    [
        # Measured reference: 75%-80%.
        ("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", 0.72),
        # Measured reference: 87%-92%.
        ("RedHatAI/Qwen3-8B-speculator.eagle3", 0.84),
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
@single_gpu_only
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_path: str,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    """Test that speculators models work with the simplified integration.

    This verifies the `vllm serve <speculator-model>` use case where
    speculative config is automatically detected from the model config
    without requiring explicit --speculative-config argument.

    Tests:
    1. Speculator model is correctly detected
    2. Verifier model is extracted from speculator config
    3. Speculative decoding is automatically enabled
    4. Text generation works correctly
    5. GSM8k accuracy of the model passes a sanity check when speculative decoding on
    6. Output matches reference (non-speculative) generation
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Generate test prompts
    test_prompts = get_test_prompts(mm_enabled=False)

    # First run: Direct speculator model (simplified integration)
    with vllm_runner(
        model_path,
        block_size=None,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        max_model_len=4096,
    ) as spec_runner:
        evaluate_llm_for_gsm8k(
            spec_runner.llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        spec_outputs = spec_runner.llm.chat(test_prompts, sampling_config)

        # Verify speculative config was auto-detected
        assert spec_runner.llm.llm_engine.vllm_config.speculative_config is not None, (
            f"Speculative config should be auto-detected for {model_path}"
        )

        spec_config = spec_runner.llm.llm_engine.vllm_config.speculative_config
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )

        # Verify draft model is set to the speculator model
        assert spec_config.model == model_path, (
            f"Draft model should be {model_path}, got {spec_config.model}"
        )

        # Extract verifier model for reference run
        verifier_model = spec_runner.llm.llm_engine.vllm_config.model_config.model

    # Second run: Reference without speculative decoding
    with vllm_runner(
        verifier_model,
        block_size=None,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        max_model_len=4096,
    ) as ref_runner:
        ref_outputs = ref_runner.llm.chat(test_prompts, sampling_config)

    # Heuristic: expect at least 66% of prompts to match exactly
    assert_request_outputs_match(
        ref_outputs,
        spec_outputs,
        required_matches=int(0.66 * len(ref_outputs)),
        context=f"speculator={model_path}, verifier={verifier_model}",
    )


@single_gpu_only
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_compressed_tensors_dspark_checkpoint(tmp_path, monkeypatch, vllm_runner):
    """Serialized split Q/K/V and FC weights load and draft through LLM."""
    from compressed_tensors.quantization import preset_name_to_scheme
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3Model

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    target, draft = tmp_path / "target", tmp_path / "draft"
    model_config = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        target_model = LlamaForCausalLM(
            LlamaConfig(num_hidden_layers=4, **model_config)
        )
        target_model.to(torch.bfloat16).save_pretrained(target)
        config = Qwen3Config(num_hidden_layers=1, **model_config)
        weights = dict(Qwen3Model(config).to(torch.bfloat16).state_dict())
        del weights["embed_tokens.weight"]
        for name, shape in (
            ("fc.weight", (128, 384)),
            ("markov_head.markov_w1.weight", (128, 128)),
            ("markov_head.markov_w2.weight", (128, 128)),
        ):
            weights[name] = torch.randn(shape, dtype=torch.bfloat16) * 0.01
        weights["hidden_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)

    quantized = ["fc", *(f"layers.0.self_attn.{name}_proj" for name in "qkv")]
    for name in quantized:
        weight = weights[f"{name}.weight"].float()
        scale = weight.abs().amax(dim=1) / torch.finfo(torch.float8_e4m3fn).max
        weights[f"{name}.weight"] = (weight / scale[:, None]).to(torch.float8_e4m3fn)
        weights[f"{name}.weight_scale"] = scale

    scheme = preset_name_to_scheme(
        "FP8_DYNAMIC", targets=[r"re:.*(?:fc|self_attn\.[qkv]_proj)$"]
    )
    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {"group_0": scheme.model_dump()},
    }
    config.dflash_config = {
        "target_layer_ids": [0, 1, 2],
        "mask_token_id": 127,
        "causal": False,
    }
    draft_config = {
        "speculators_model_type": "dspark",
        "architectures": ["Qwen3DSparkModel"],
        "transformer_layer_config": config.to_dict(),
        "speculators_config": {
            "proposal_methods": [{"speculative_tokens": 2}],
            "verifier": {"name_or_path": str(target)},
        },
        "aux_hidden_state_layer_ids": [1, 2, 3],
        "markov_rank": 128,
        "block_size": 16,
        "sample_from_anchor": True,
        "quantization_config": quantization_config,
    }
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(draft_config))
    save_file(weights, str(draft / "model.safetensors"))

    def check_draft(worker):
        model = worker.get_draft_model().model
        qkv = model.layers[0].self_attn.qkv_proj
        assert qkv.weight.dtype == current_platform.fp8_dtype()
        expected_scale = torch.cat(
            [weights[f"layers.0.self_attn.{name}_proj.weight_scale"] for name in "qkv"]
        )
        torch.testing.assert_close(qkv.weight_scale.cpu(), expected_scale)
        assert model._fused_kv_weight is None

    with vllm_runner(
        str(target),
        speculative_config=dict(
            method="dspark", model=str(draft), num_speculative_tokens=2
        ),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        enforce_eager=True,
        enable_flashinfer_autotune=False,
        disable_log_stats=False,
        max_model_len=32,
        max_num_seqs=1,
        kv_cache_memory_bytes=16 << 20,
    ) as llm:
        llm.collective_rpc(check_draft)
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        generated_token_ids = llm.generate([[1, 5, 7, 9]], params)[0][0][0]
        drafted = get_spec_decode_metric_value(
            llm.llm.get_metrics(), "vllm:spec_decode_num_draft_tokens"
        )
    assert len(generated_token_ids) == 12
    assert drafted > 0
