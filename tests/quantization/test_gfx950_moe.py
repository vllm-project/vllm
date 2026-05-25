# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm gfx950 quantized-MoE initialization coverage.

This file mirrors the intent of ``test_blackwell_moe.py`` using ROCm-native
features instead of CUDA-only backends:

- public Neural Magic compressed-tensors MoE models
- public Quark INT8 MoE smoke coverage
- ROCm Quark MXFP4/BF16 MoE with explicit ``aiter`` and ``triton`` backends
- ROCm GPT-OSS MXFP4/FP8 MoE in the same shape the repo already advertises
- ROCm DeepSeek Quark MXFP4/UINT8 MoE with explicit backend coverage
"""

import importlib.metadata
import importlib.util
from typing import Any

import huggingface_hub
import pytest
import torch
from packaging import version

from tests.utils import RemoteOpenAIServer
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not on_mi3xx(),
    reason="MI300/MI350 ROCm only",
)


def _has_quark_mxfp4_support() -> bool:
    if importlib.util.find_spec("quark") is None:
        return False
    try:
        return version.parse(importlib.metadata.version("amd-quark")) >= version.parse(
            "0.9.0"
        )
    except importlib.metadata.PackageNotFoundError:
        return False


QUARK_MXFP4_AVAILABLE = _has_quark_mxfp4_support()
QUARK_AVAILABLE = importlib.util.find_spec("quark") is not None

HF_OVERRIDE_TEXT = {
    "num_layers": 4,
    "num_hidden_layers": 4,
}
ROCM_ATTENTION_BACKENDS = [
    pytest.param("ROCM_ATTN", id="rocm_attn"),
    pytest.param("ROCM_AITER_UNIFIED_ATTN", id="rocm_aiter_unified_attn"),
]

ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_AVAILABLE = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_AVAILABLE = rocm_aiter_ops.is_fused_moe_enabled()


def _has_huggingface_access(repo_id: str) -> bool:
    try:
        huggingface_hub.list_repo_refs(repo_id)
        return True
    except huggingface_hub.errors.RepositoryNotFoundError:
        return False
    except huggingface_hub.errors.HfHubHTTPError:
        return False


def _require_repo_access(repo_id: str) -> None:
    if not _has_huggingface_access(repo_id):
        pytest.skip(f"Read access to huggingface.co/{repo_id} is required.")


def _can_initialize(
    model: str,
    *,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "1",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-uvicorn-access-log",
        *(extra_args or []),
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        env_dict=env,
        max_wait_seconds=1500,
        override_hf_configs=hf_overrides,
    ) as server:
        client = server.get_client()
        completion = client.completions.create(
            model=model,
            prompt=["Hello, World!"],
            temperature=0,
            max_tokens=2,
        )
        print(completion)
        assert completion.choices[0].text is not None


def _make_w4a4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size_per_partition=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(not ROCM_AITER_AVAILABLE, reason="Requires AITER enabled")
def test_w4a4_dispatches_to_aiter():
    """With AITER enabled + GFX950, W4A4 selects AITER_MXFP4_MXFP4."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is not None


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(
    ROCM_AITER_AVAILABLE,
    reason="Test requires AITER disabled (unset VLLM_ROCM_USE_AITER)",
)
def test_w4a4_raises_without_aiter_and_no_moe_backend():
    """Without AITER and no --moe-backend, raises NotImplementedError
    with hint to use --moe-backend emulation."""
    config = _make_w4a4_moe_config()
    with pytest.raises(NotImplementedError, match="--moe-backend emulation"):
        select_mxfp4_moe_backend(config, activation_key=kMxfp4Dynamic)


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
def test_w4a4_dispatches_to_emulation_with_moe_backend():
    """With --moe-backend emulation, W4A4 selects EMULATION."""
    config = _make_w4a4_moe_config(moe_backend="emulation")
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


@pytest.mark.parametrize("attention_backend", ROCM_ATTENTION_BACKENDS)
def test_nm_qwen15_w4a16_moe_initializes_across_rocm_attention_backends(
    attention_backend: str,
):
    """A public Neural Magic W4A16 MoE model should initialize with both ROCm
    attention backends that are meaningful on MI3xx."""
    repo_id = "nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--attention-backend", attention_backend],
    )


def test_nm_mixtral_w4a16_moe_initializes():
    """A second public Neural Magic MoE family should initialize on ROCm."""
    repo_id = "nm-testing/Mixtral-8x7B-Instruct-v0.1-W4A16-quantized"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
    )


@pytest.mark.skipif(
    not QUARK_AVAILABLE,
    reason="quark package is required for ROCm Quark MoE tests",
)
def test_tiny_quark_int8_moe_initializes():
    """A small public Quark INT8 MoE model should initialize on MI3xx."""
    _can_initialize(
        "nameistoken/tiny-qwen3-moe-w8a8-int8-quark",
        hf_overrides=HF_OVERRIDE_TEXT,
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason="amd-quark>=0.9.0 is required for ROCm MXFP4 MoE tests",
)
@pytest.mark.parametrize("moe_backend", ["aiter", "triton"])
def test_gptoss_rocm_quark_mxfp4_bf16_moe_backends_initialize(
    moe_backend: str,
):
    """The ROCm GPT-OSS MXFP4/BF16 Quark MoE path should initialize with the
    two real ROCm MoE backends exposed at the CLI."""
    repo_id = "amd/gpt-oss-20b-w-mxfp4-a-bf16"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--attention-backend",
            "ROCM_AITER_UNIFIED_ATTN",
            "--moe-backend",
            moe_backend,
            "--tokenizer",
            "openai/gpt-oss-20b",
            "--tensor-parallel-size",
            "1",
        ],
        env={"VLLM_ROCM_USE_AITER": "1"} if moe_backend == "aiter" else None,
    )


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason="amd-quark>=0.9.0 is required for ROCm MXFP4 MoE tests",
)
def test_gptoss_rocm_quark_mxfp4_fp8_moe_initializes():
    """The ROCm GPT-OSS MXFP4/FP8 Quark MoE path should initialize in the same
    form the repo already advertises for ROCm evals."""
    repo_id = "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--attention-backend",
            "ROCM_AITER_UNIFIED_ATTN",
            "--tokenizer",
            "openai/gpt-oss-20b",
            "--tensor-parallel-size",
            "1",
        ],
        env={"VLLM_ROCM_USE_AITER": "1"},
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason="amd-quark>=0.9.0 is required for ROCm MXFP4 MoE tests",
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        pytest.param(None, id="auto"),
        pytest.param("aiter", id="aiter"),
        pytest.param("triton", id="triton"),
    ],
)
def test_deepseek_rocm_quark_mxfp4_uint8_moe_backends_initialize(
    moe_backend: str | None,
):
    """The ROCm DeepSeek MXFP4/UINT8 Quark MoE path should initialize across
    the real ROCm backend choices for the MXFP4 MoE oracle."""
    repo_id = "amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--tensor-parallel-size",
            "1",
            *([] if moe_backend is None else ["--moe-backend", moe_backend]),
        ],
        env={"VLLM_ROCM_USE_AITER": "1"} if moe_backend == "aiter" else None,
    )
