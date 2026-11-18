# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3's native shapes through the Flash specialized decode kernels."""

from contextlib import nullcontext

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Requires the SM100 family",
)


@pytest.mark.parametrize("backend", [None, "auto", "stock", "latency_specialized"])
@pytest.mark.parametrize("use_mtp", [False, True])
@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize(
    "config_args,unsupported",
    [
        pytest.param({}, None, id="tp1"),
        pytest.param({"tensor_parallel_size": 8}, None, id="tp8"),
        pytest.param({"pipeline_parallel_size": 2}, None, id="pp2"),
        pytest.param({"data_parallel_size": 2}, "data parallelism", id="dp2"),
        pytest.param(
            {"enable_return_routed_experts": True},
            "routed expert capture",
            id="routed-experts",
        ),
        pytest.param(
            {"tensor_parallel_size": 2, "enable_expert_parallel": True},
            "expert parallelism",
            id="ep",
        ),
        pytest.param(
            {
                "tensor_parallel_size": 2,
                "enable_expert_parallel": True,
                "enable_eplb": True,
            },
            "expert parallelism",
            id="eplb",
        ),
    ],
)
def test_glm53_decode_configuration(
    backend, use_mtp, enforce_eager, config_args, unsupported
):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.models.deepseek_v32.nvidia.specialized_decode import (
        specialized_decode_config,
    )

    config = EngineArgs(
        model="zai-org/GLM-5.3",
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enforce_eager=enforce_eager,
        compilation_config={
            "decode_cudagraph_capture_request_sizes": [1, 2, 4],
        },
        additional_config={"decode_backend": backend} if backend else {},
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": 2} if use_mtp else None
        ),
        distributed_executor_backend="mp",
        **config_args,
    ).create_engine_config()
    text = config.model_config.hf_text_config
    unsupported = unsupported or ("FULL decode graphs" if enforce_eager else None)
    if unsupported and backend == "latency_specialized":
        with pytest.raises(ValueError, match=unsupported):
            specialized_decode_config(config, text)
    else:
        decode = specialized_decode_config(config, text)
        if backend == "stock" or unsupported:
            assert decode is None
        else:
            assert decode is not None
            assert decode.tokens == ((1, 2, 3, 4, 6, 12) if use_mtp else (1, 2, 4))


@pytest.mark.parametrize(
    "mode,capture,prefills,decodes,prefill_tokens,expected",
    [
        ("FULL", False, 0, 4, 0, True),
        ("NONE", False, 0, 4, 0, False),
        ("PIECEWISE", False, 0, 4, 0, False),
        ("FULL", False, 1, 0, 4, True),
        ("FULL", False, 1, 1, 3, True),
        ("FULL", False, 1, 1, 2, True),
        ("NONE", True, 0, 4, 0, True),
        ("NONE", True, 1, 0, 4, True),
        ("NONE", True, 1, 1, 3, True),
        ("PIECEWISE", True, 0, 4, 0, False),
    ],
)
def test_decode_step_selects_prepared_graphs(
    mode, capture, prefills, decodes, prefill_tokens, expected, default_vllm_config
):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v32.nvidia.specialized_decode import (
        DecodeConfig,
        is_decode_step,
    )
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.full((prefills + decodes,), 16, dtype=torch.int32),
        max_seq_len=16,
        slot_mapping=torch.arange(4),
        num_decodes=decodes,
        num_decode_tokens=decodes,
        num_prefills=prefills,
        num_prefill_tokens=prefill_tokens,
    )
    marker = torch.zeros(1, device="cuda")
    with (
        set_forward_context(
            {"indexer": metadata},
            default_vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode[mode],
        ),
        torch.cuda.graph(torch.cuda.CUDAGraph()) if capture else nullcontext(),
    ):
        marker.add_(1)
        assert is_decode_step(4, DecodeConfig((4,))) == expected
        assert not is_decode_step(3, DecodeConfig((4,)))


@pytest.mark.parametrize("tokens", [1, 6, 8])
@pytest.mark.parametrize(
    "n,k",
    [
        (2624, 6144),
        (2048, 2048),
        (4096, 2048),
        (8192, 2048),
        (16384, 2048),
        (6144, 2048),
        (6144, 4096),
        (6144, 8192),
        (6144, 16384),
    ],
)
def test_glm53_fp8_projection(tokens, n, k, dist_init):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.deepseek_v32.nvidia.specialized_decode import (
        DecodeLinearMethod,
        Fp8Projection,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.linear import planned_step
    from vllm.utils.torch_utils import set_default_torch_dtype

    from .glm5next.test_specialized_decode_shared_expert import _quant

    torch.manual_seed(53)
    config = VllmConfig(
        model_config=ModelConfig(
            model="zai-org/GLM-5.3", dtype="bfloat16", max_model_len=256
        )
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        module = ReplicatedLinear(
            k,
            n,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
            ),
        )
        module.weight.data.copy_(torch.randn(n, k, device="cuda") * 0.5)
        module.weight_scale_inv.data.fill_(2**-7)
        module.quant_method.process_weights_after_loading(module)
    projection = Fp8Projection(module, [1, tokens])
    x = torch.randn(tokens, k, device="cuda").bfloat16()
    scales = projection.scales.double().repeat_interleave(128, 0)[:n]
    weight = module.weight.double() * scales.repeat_interleave(128, 1)
    ref = _quant(x, True).double() @ weight.T
    original = module.quant_method
    module.quant_method = DecodeLinearMethod(original, projection, projection.tokens)
    with planned_step():
        actual, _ = module(x)
    if k == 16384 and tokens >= 6:
        assert tokens not in projection.tokens
        torch.testing.assert_close(actual, original.apply(module, x), atol=0, rtol=0)
    else:
        assert tokens in projection.tokens
    torch.testing.assert_close(
        actual.double(), ref, atol=5e-3 * ref.abs().max(), rtol=0
    )
    if (tokens, n, k) == (6, 6144, 4096):
        strided = torch.empty(tokens, k + 128, device="cuda").bfloat16()[:, :k]
        strided.copy_(x)
        for inputs, bias in (
            (strided, None),
            (x.unflatten(0, (2, 3)), torch.randn(n, device="cuda").bfloat16()),
            (x[0], torch.randn(n, device="cuda", dtype=torch.float32)),
        ):
            stock = original.apply(module, inputs, bias)
            fast = projection(inputs.reshape(-1, k).contiguous()).view_as(stock)
            if bias is not None:
                fast = (fast + bias).to(inputs.dtype)
            with planned_step():
                actual = module.quant_method.apply(module, inputs, bias)
            torch.testing.assert_close(actual, fast, atol=0, rtol=0)
            torch.testing.assert_close(
                actual, stock, atol=5e-3 * stock.abs().max(), rtol=0
            )


@pytest.mark.parametrize("use_mtp", [False, True])
def test_glm53_engine(vllm_runner, use_mtp, monkeypatch, enable_pickle, capfd):
    """Default LLM settings select specialized decode, including with MTP."""
    from .utils import check_logprobs_close

    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    outputs = []
    for backend in ("stock", None):
        capfd.readouterr()
        with vllm_runner(
            "zai-org/GLM-5.3",
            load_format="dummy",
            hf_overrides={
                "num_hidden_layers": 4,
                "num_nextn_predict_layers": int(use_mtp),
                "indexer_types": ["full", "full", "full", "shared"],
                "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            },
            max_model_len=256,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
            block_size=64,
            gpu_memory_utilization=0.4,
            kernel_config={
                "enable_flashinfer_autotune": False,
                "enable_jit_warmup": False,
            },
            compilation_config={
                "decode_cudagraph_capture_request_sizes": [1, 2, 4],
            },
            additional_config={"decode_backend": backend} if backend else {},
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": 5} if use_mtp else None
            ),
            seed=53,
        ) as runner:
            if backend is None:
                assert runner.llm.apply_model(
                    lambda model: [
                        type(layer._decode.mlp).__name__ for layer in model.model.layers
                    ]
                ) == [["DenseMlp", "DenseMlp", "DenseMlp", "GroupedExperts"]]
            outputs.append(
                runner.generate_greedy_logprobs(
                    [
                        "Explain tensor parallelism briefly.",
                        "What is sparse attention?",
                    ],
                    max_tokens=8,
                    num_logprobs=5,
                )
            )
        if backend is None:
            captured = capfd.readouterr()
            assert "GLM-5.3 specialized decode selected" in captured.out + captured.err
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="stock",
        name_1="default",
    )
