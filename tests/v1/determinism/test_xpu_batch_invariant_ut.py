# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU-only batch-invariance tests for norm, collectives, and sampler behavior."""

import pytest
import torch

from vllm import envs
from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU batch-invariance tests"
)


@pytest.fixture
def tiny_qwen3_moe_path(tmp_path):
    """Create a local Qwen3-MoE config for dummy-loading tests."""
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config.save_pretrained(tmp_path)
    return tmp_path


@pytest.mark.parametrize("source", ["argument", "checkpoint"])
def test_rejects_weight_quantization(tiny_qwen3_moe_path, source):
    """Reject both explicit and checkpoint-inferred quantization before loading."""
    from transformers import Qwen3MoeConfig

    if source == "checkpoint":
        config = Qwen3MoeConfig.from_pretrained(tiny_qwen3_moe_path)
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        }
        config.save_pretrained(tiny_qwen3_moe_path)

    model_config = ModelConfig(
        model=str(tiny_qwen3_moe_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        quantization="fp8" if source == "argument" else None,
    )
    assert model_config.quantization == "fp8"
    with pytest.raises(ValueError, match="supports only unquantized models"):
        VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format="dummy"),
        )


@pytest.mark.parametrize("cache_dtype", ["fp8", "int8_per_token_head", "nvfp4"])
def test_rejects_quantized_kv_cache(cache_dtype):
    with pytest.raises(ValueError, match="does not support quantized KV caches"):
        VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))


@pytest.mark.parametrize("cache_dtype", ["auto", "float16", "bfloat16"])
def test_accepts_unquantized_kv_cache(cache_dtype):
    config = VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))
    assert config.cache_config.cache_dtype == cache_dtype


def test_quantized_kv_cache_allowed_without_batch_invariance(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    config = VllmConfig(cache_config=CacheConfig(cache_dtype="fp8"))
    assert config.cache_config.cache_dtype == "fp8"


# These dimensions and seeds locally reproduce non-invariant norm mismatches.
# They make the invariant equality assertion exercise a discriminating case.
@pytest.mark.parametrize(
    (
        "norm_name",
        "batch_size",
        "hidden_size",
        "position",
        "weight_seed",
        "input_seed",
        "input_pattern",
    ),
    [
        ("rms", 1024, 16384, 512, 7, 42, "normal"),
        ("gemma", 257, 8192, 255, 1, 2, "normal"),
        ("rms", 5, 8192, 3, 7, 46, "scaled"),
        ("gemma", 5, 8192, 3, 1, 46, "scaled"),
    ],
)
def test_residual_norm_preserves_batch_invariance(
    norm_name,
    batch_size,
    hidden_size,
    position,
    weight_seed,
    input_seed,
    input_pattern,
    monkeypatch,
    default_vllm_config,
):
    """Residual norm output must not change when its row changes batch position."""
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    init_batch_invariance()
    norm_cls = RMSNorm if norm_name == "rms" else GemmaRMSNorm
    norm = norm_cls(hidden_size, eps=1e-6).to(device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.manual_seed(weight_seed)
        norm.weight.uniform_(-0.5, 0.5)

    if input_pattern == "scaled":
        # Heterogeneous columns expose native mean's batch-dependent rounding.
        generator = torch.Generator(device="cpu").manual_seed(input_seed)
        scale = torch.randn(hidden_size, generator=generator).exp()
        x = (torch.randn(batch_size, hidden_size, generator=generator) * scale).to(
            device="xpu", dtype=torch.bfloat16
        )
        residual = (
            torch.randn(batch_size, hidden_size, generator=generator) * scale
        ).to(device="xpu", dtype=torch.bfloat16)
    else:
        torch.manual_seed(input_seed)
        x = torch.randn(batch_size, hidden_size, device="xpu", dtype=torch.bfloat16)
        residual = torch.randn_like(x)

    single_output, single_residual = norm(
        x[position : position + 1].clone(),
        residual[position : position + 1].clone(),
    )
    batch_output, batch_residual = norm(x.clone(), residual.clone())

    assert torch.equal(single_output, batch_output[position : position + 1])
    assert torch.equal(single_residual, batch_residual[position : position + 1])


@pytest.mark.parametrize(
    "operation", ["all_reduce", "reduce_scatter", "reduce_scatterv"]
)
@pytest.mark.parametrize("rank", range(4))
def test_collectives_preserve_rank_order_and_partition(monkeypatch, operation, rank):
    """Cancellation exposes reordered sums; mock ranks require no extra XPUs."""
    from vllm.distributed.device_communicators import xpu_communicator

    inputs = [
        torch.full((8, 3), value, dtype=torch.float32, device="xpu")
        for value in (1e20, -1e20, 3.0, 0.0)
    ]
    inputs[2].add_(torch.arange(24, device="xpu").reshape(8, 3))

    def all_gather_single(output, input_, group):
        torch.testing.assert_close(input_, inputs[rank].reshape(-1), rtol=0, atol=0)
        output.copy_(torch.stack(inputs).reshape(-1))

    monkeypatch.setattr(xpu_communicator.dist, "all_gather_single", all_gather_single)
    communicator = XpuCommunicator.__new__(XpuCommunicator)
    communicator.world_size = 4
    communicator.rank_in_group = rank
    communicator.device_group = object()
    expected = inputs[0].clone()
    for value in inputs[1:]:
        expected.add_(value)
    assert torch.equal(expected, inputs[2])

    if operation == "all_reduce":
        actual = communicator.all_reduce(inputs[rank])
    elif operation == "reduce_scatter":
        actual = communicator.reduce_scatter(inputs[rank], dim=0)
        expected = expected.chunk(4)[rank]
    else:
        sizes = [1, 3, 2, 2]
        actual = communicator.reduce_scatterv(inputs[rank], dim=0, sizes=sizes)
        expected = expected.split(sizes)[rank]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# This batch size, vocabulary size, and positions locally reproduce an
# unseeded non-invariant XPU sampler mismatch.
@pytest.mark.parametrize("position", [0, 128, 255])
def test_seeded_sampler_preserves_full_distribution_and_rng(position, monkeypatch):
    """Moving a request must preserve its distribution and per-request RNG."""
    from vllm.triton_utils import HAS_TRITON
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    assert HAS_TRITON, "XPU sampling must exercise the Triton top-k/top-p path"
    monkeypatch.setattr(envs, "VLLM_XPU_USE_SAMPLER_KERNEL", True)
    sampler = TopKTopPSampler(logprobs_mode="processed_logprobs")
    torch.manual_seed(42)
    logits = torch.randn(256, 8192, device="xpu", dtype=torch.float32)
    single_rng = torch.Generator(device="xpu").manual_seed(123)
    batch_rng = torch.Generator(device="xpu").manual_seed(123)
    k = torch.full((256,), 32, dtype=torch.int32, device="xpu")
    p = torch.full((256,), 0.9, device="xpu")
    for _ in range(3):
        single_token, single_logprobs = sampler(
            logits[position : position + 1].clone(),
            {0: single_rng},
            k[position : position + 1],
            p[position : position + 1],
        )
        batch_tokens, batch_logprobs = sampler(
            logits.clone(), {position: batch_rng}, k, p
        )
        assert single_logprobs is not None and batch_logprobs is not None
        torch.testing.assert_close(
            single_logprobs[0], batch_logprobs[position], rtol=0, atol=0
        )
        assert single_token.item() == batch_tokens[position].item()
        assert torch.equal(single_rng.get_state(), batch_rng.get_state())
