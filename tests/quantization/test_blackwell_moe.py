# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )


@pytest.fixture(scope="module", autouse=True)
def set_test_environment():
    """Sets environment variables required for this test module."""
    # Make sure TRTLLM attention is available
    os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
    # Set compilation threads to 16 to speed up startup
    os.environ["FLASHINFER_NVCC_THREADS"] = "16"


# Override the backbone layers to 4 for faster startup
HF_OVERRIDE_TEXT = {
    "num_layers": 4,
    "num_hidden_layers": 4,
}
HF_OVERRIDE_MM = {
    "text_config": {"num_layers": 4, "num_hidden_layers": 4},
}


def can_initialize(
    model: str,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
):
    # Server arguments
    extra_args = extra_args if extra_args is not None else []
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0}),
        *extra_args,
    ]

    # Launch server and make a simple request
    with RemoteOpenAIServer(
        model,
        server_args,
        max_wait_seconds=1500,  # Due to FlashInfer compile
        override_hf_configs=hf_overrides,
    ) as server:
        client = server.get_client()
        # Make a simple request to verify the server works
        completion = client.completions.create(
            model=model,
            prompt=["Hello, World!"],
            temperature=0,
            max_tokens=2,
        )
        print(completion)
        assert completion.choices[0].text is not None


## Llama4 ##


@pytest.mark.skip(
    reason=(
        "RuntimeError: run_moe() Expected a value of type "
        "'Optional[List[Tensor]]' for argument '_9' but instead found type "
        "'list'."
    )
)
def test_llama4_fp8_tensor_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


def test_llama4_fp8_tensor_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


def test_llama4_nvfp4_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


def test_llama4_nvfp4_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


## DeepSeekV3 ##


def test_deepseek_fp8_block_moe_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=deep_gemm"],
    )


def test_deepseek_fp8_block_moe_vllm_triton(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=triton"],
    )


@pytest.mark.skip(
    reason=(
        "Known issue: lack of kernel support. "
        "Expected failure: assert self.block_quant is None"
    )
)
def test_deepseek_fp8_block_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


def test_deepseek_fp8_block_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


def test_deepseek_nvfp4_moe_flashinfer_vllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/DeepSeek-R1-0528-FP4-v2",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=cutlass"],
    )


def test_deepseek_nvfp4_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/DeepSeek-R1-0528-FP4-v2",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


def test_deepseek_nvfp4_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/DeepSeek-R1-0528-FP4-v2",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


## GPT-OSS ##


def test_gptoss_mxfp4bf16_moe_flashinfer(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "openai/gpt-oss-20b",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


def test_gptoss_mxfp4mxfp8_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "openai/gpt-oss-20b",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--moe-backend",
            "flashinfer_cutlass",
            "--quantization-config.moe.activation",
            "mxfp8",
        ],
    )


def test_gptoss_mxfp4mxfp8_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "openai/gpt-oss-20b",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--quantization-config.moe.activation",
            "mxfp8",
        ],
    )


def test_gptoss_eager(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "openai/gpt-oss-20b",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--enforce-eager"],
    )


## Qwen3 Next ##


@pytest.mark.skip(
    reason=(
        "FLASHINFER TRTLLM MoE has a bug with all negative router logits "
        "for models with RENORMALIZE. This will be re-enabled once the "
        "issue is fixed in flashinfer."
    )
)
def test_qwen3_next_bf16_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


## NemoTron ##


def test_nemotron_fp8_moe_flashinfer_throughput(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


@pytest.mark.skip(
    reason=(
        "FP8 MoE backend FLASHINFER_TRTLLM does not support the "
        "deployment configuration since kernel does not support "
        "no act_and_mul MLP layer."
    )
)
def test_nemotron_fp8_moe_flashinfer_latency(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


@pytest.mark.skip(
    reason=(
        "FP8 MoE backend TRITON does not support the "
        "deployment configuration since kernel does not support "
        "no act_and_mul MLP layer."
    )
)
def test_nemotron_fp8_moe_vllm_triton(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=triton"],
    )


def test_nemotron_fp4_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


def test_nemotron_fp4_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_trtllm"],
    )


def test_nemotron_nvfp4_per_token_moe_flashinfer_trtllm(
    monkeypatch: pytest.MonkeyPatch,
):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--moe-backend=flashinfer_trtllm",
            "--quantization-config.moe.activation=nvfp4_per_token",
        ],
    )


def test_modelopt_nvfp4_dummy_scales(tmp_path, monkeypatch, vllm_runner):
    """Dummy NVFP4 MoE weights load with valid scales and produce finite logits."""
    import math

    import torch
    from transformers import Qwen3MoeConfig

    from vllm import SamplingParams
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["lm_head", "model.layers.*.self_attn.*"],
    }
    config.save_pretrained(tmp_path)

    def check_scales(model):
        experts = [m for m in model.modules() if isinstance(m, RoutedExperts)]
        assert len(experts) == 1
        layer = experts[0]
        assert isinstance(layer.quant_method, ModelOptNvFp4FusedMoE)
        assert layer.moe_config.moe_backend == "flashinfer_trtllm"
        for name, param in layer.named_parameters():
            if "scale" in name:
                values = param.float()
                assert torch.isfinite(values).all(), name
                assert (values > 0).all(), name

    with vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        load_format="dummy",
        moe_backend="flashinfer_trtllm",
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_flashinfer_autotune=False,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        kv_cache_memory_bytes=16 << 20,
    ) as llm:
        llm.apply_model(check_scales)
        outputs = llm.get_llm().generate(
            [{"prompt_token_ids": [1, 2, 3]}],
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, logprobs=1),
        )
        completion = outputs[0].outputs[0]
        assert len(completion.token_ids) == 2
        assert len(completion.logprobs) == 2
        assert all(
            math.isfinite(value.logprob)
            for step in completion.logprobs
            for value in step.values()
        )


@pytest.mark.parametrize("nvfp4", [False, True])
def test_muse_megamoe_dummy_loading(tmp_path, monkeypatch, vllm_runner, nvfp4):
    """Dummy loading finalizes Muse's experts and residual gates before generation."""
    import math

    import torch

    from vllm import SamplingParams
    from vllm.model_executor.models import ModelRegistry
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )
    from vllm.models.muse_spark.nvidia.mega_moe import MuseSparkMegaMoEExperts

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    ModelRegistry.register_model(
        "MuseSpark1xForCausalLM",
        "vllm.models.muse_spark.nvidia.model:MuseSpark1xForCausalLM",
    )
    config = dict(
        architectures=["MuseSpark1xForCausalLM"],
        model_type="muse_spark_text",
        vocab_size=128,
        hidden_size=512,
        intermediate_size=512,
        moe_hidden_size=512,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=0,
        max_position_embeddings=64,
        hidden_act="silu",
        attention_bias=False,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        layer_rope_theta=[10000.0],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        final_logit_softcapping=None,
        output_multiplier=0.5,
        routed_scaling_factor=1.0,
        qk_scale_factor=1.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    if nvfp4:
        config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head", "model.layers.*.self_attn.*"],
        }
    (tmp_path / "config.json").write_text(json.dumps(config))

    def check_finalized(model):
        experts = [m for m in model.modules() if isinstance(m, MuseSparkMegaMoEExperts)]
        assert len(experts) == 1
        layer = experts[0]
        assert layer.is_nvfp4 == nvfp4
        assert layer._transformed_l1 is not None
        assert layer._transformed_l2 is not None
        if nvfp4:
            for name, param in layer.named_parameters():
                if "scale" in name:
                    assert torch.isfinite(param.float()).all(), name
                    assert (param.float() > 0).all(), name
            assert torch.equal(
                layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
            )
        gates = [
            m
            for m in model.modules()
            if isinstance(m, MuseSpark1xNormPreservingResidualConnection)
        ]
        assert len(gates) == 2
        assert all(gate._coefficients_ready for gate in gates)

    with vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        load_format="dummy",
        moe_backend="deep_gemm_mega_moe",
        enable_expert_parallel=True,
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_flashinfer_autotune=False,
        kernel_config={"enable_jit_warmup": False, "enable_cutedsl_warmup": False},
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        kv_cache_memory_bytes=16 << 20,
    ) as llm:
        llm.apply_model(check_finalized)
        outputs = llm.get_llm().generate(
            [{"prompt_token_ids": [1, 2, 3]}],
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, logprobs=1),
        )
        completion = outputs[0].outputs[0]
        assert len(completion.token_ids) == 2
        assert all(
            math.isfinite(value.logprob)
            for step in completion.logprobs
            for value in step.values()
        )
