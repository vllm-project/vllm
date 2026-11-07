# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for online MXFP8 quantization.

Loads a BF16 model with ``--quantization mxfp8`` (online quantization) and
compares log-probabilities against the same model served in BF16 without
quantization.  This exercises the full pipeline: config parsing,
``Mxfp8OnlineLinearMethod``, ``Mxfp8OnlineMoEMethod``, weight loading,
online quantization / shuffling, and inference through ``apply_monolithic``.

Layer skipping (``modules_to_not_convert``) is configured in the model's
``config.json`` under ``quantization_config`` and is not tested here.

``example_prompts`` is a pytest fixture (from conftest.py) that loads 8
diverse prompts from ``tests/prompts/example.txt``.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.platforms import current_platform

from ..utils import check_logprobs_close

# Small MoE models that fit on a single GPU and exercise both linear + MoE,
# one per modeling backend.
MOE_MODEL = "ibm-granite/granite-3.0-1b-a400m-base"
TRANSFORMERS_MOE_MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
# A small dense model (no MoE) to validate the linear-only path.
DENSE_MODEL = "Qwen/Qwen3-0.6B"

MAX_MODEL_LEN = 1024
MAX_TOKENS = 4
NUM_LOG_PROBS = 8


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize(
    "model",
    [DENSE_MODEL, MOE_MODEL, TRANSFORMERS_MOE_MODEL],
    ids=["dense", "moe", "moe_transformers"],
)
def test_mxfp8_logprobs(
    vllm_runner,
    quant_baseline_logprobs,
    example_prompts,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare BF16 baseline logprobs against online MXFP8-quantized model.

    Runs the same model twice -- once in BF16 (baseline) and once with
    online MXFP8 quantization -- then checks that the top log-probabilities
    are close.  Only 4 tokens are generated to keep the test fast while
    still catching numerical divergence.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", "true")

        baseline_outputs = quant_baseline_logprobs(
            model,
            example_prompts,
            max_model_len=MAX_MODEL_LEN,
            max_tokens=MAX_TOKENS,
            num_logprobs=NUM_LOG_PROBS,
        )

        with vllm_runner(
            model,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="mxfp8",
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16",
            name_1="mxfp8",
        )


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_accepts_silu_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
    )

    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    backend, experts_cls = select_mxfp8_moe_backend(config)

    assert backend == Fp8MoeBackend.AITER_MXFP8
    assert experts_cls is aiter_mxfp8_moe.AiterMxfp8Experts


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_requires_swigluoai_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
    )

    with pytest.raises(ValueError, match="hardcodes SwiGLU-OAI"):
        select_mxfp8_moe_backend(config)


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_accepts_swigluoai_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
        swiglu_alpha=aiter_mxfp8_moe._AITER_SWIGLU_ALPHA,
        swiglu_beta=aiter_mxfp8_moe._AITER_SWIGLU_BETA,
    )

    backend, experts_cls = select_mxfp8_moe_backend(config)

    assert backend == Fp8MoeBackend.AITER_MXFP8
    assert experts_cls is aiter_mxfp8_moe.AiterMxfp8Experts


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize(
    "model",
    [DENSE_MODEL, MOE_MODEL, TRANSFORMERS_MOE_MODEL],
    ids=["dense", "moe", "moe_transformers"],
)
def test_mxfp8_generation(vllm_runner, model: str) -> None:
    """Smoke test: verify online MXFP8 model generates coherent text."""
    prompt = "1 2 3 4 5"
    with vllm_runner(
        model,
        enforce_eager=True,
        quantization="mxfp8",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.generate_greedy([prompt], max_tokens=5)

    generated = output[0][1]
    assert len(generated) > len(prompt), (
        f"MXFP8 model produced no new tokens. Output: {generated!r}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Exercises CUDA weight-only MXFP8 kernels."
)
def test_quark_mxfp8_weight_only_checkpoint(vllm_runner, tmp_path, enable_pickle):
    """Loading a weight-only checkpoint must not add activation quantization."""
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from vllm import SamplingParams
    from vllm.model_executor.kernels.linear import (
        EmulationMxfp8LinearKernel,
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

    torch.manual_seed(65193)
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    state = dict(model.state_dict())
    for name, weight in list(state.items()):
        if weight.ndim == 2 and name not in (
            "model.embed_tokens.weight",
            "lm_head.weight",
        ):
            quantized, scale = _mxfp8_e4m3_quantize_torch(weight)
            state[name] = quantized.contiguous()
            state[name.removesuffix("weight") + "weight_scale"] = scale.contiguous()
    config.quantization_config = {
        "quant_method": "quark",
        "export": {"kv_cache_group": [], "pack_method": "reorder"},
        "global_quant_config": {
            "weight": {
                "dtype": "fp8_e4m3",
                "qscheme": "per_group",
                "group_size": 32,
                "scale_format": "e8m0",
                "is_dynamic": False,
            },
        },
        "layer_type_quant_config": {},
        "layer_quant_config": {},
        "exclude": ["lm_head"],
    }
    config.save_pretrained(tmp_path)
    save_file(state, tmp_path / "model.safetensors")

    def check_kernels(loaded_model):
        kernels = [
            layer.scheme.ocp_mx_linear
            for layer in loaded_model.modules()
            if isinstance(getattr(layer, "scheme", None), QuarkOCP_MX)
        ]
        assert kernels
        assert all(
            isinstance(k, (MarlinMxfp8LinearKernel, EmulationMxfp8LinearKernel))
            for k in kernels
        )

    with vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        kv_cache_memory_bytes=16 * 1024**2,
        gpu_memory_utilization=0.1,
        load_format="safetensors",
        trust_remote_code=False,
    ) as runner:
        runner.apply_model(check_kernels)
        result = runner.llm.generate(
            [{"prompt_token_ids": [1, 3, 5]}],
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        )
        assert len(result[0].outputs[0].token_ids) == 2
