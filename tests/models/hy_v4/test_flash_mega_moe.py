# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.layers.linear import LinearBase
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="HY4 Flash block-FP8 MegaMoE requires SM90",
)


def _write_checkpoint(model, path: str) -> list[str]:
    """Save packed FP8 experts with BF16 block scales, as in Hy4-flash-FP8."""
    generator = torch.Generator().manual_seed(17)
    weights = {}
    for name, param in model.named_parameters():
        if ".mla_attn." in name:
            # These absorbed MLA weights are derived from kv_b_proj on load.
            continue
        if ".experts.routed_experts." in name:
            name = name.replace(".routed_experts", "")
            name = name.replace("w13_weight", "gate_up_proj")
            name = name.replace("w2_weight", "down_proj")
            e, n, k = param.shape
            weight = torch.randn(e, n, k, generator=generator)
            scale = (
                (0.5 + torch.rand(e, n // 128, k // 128, generator=generator)) / k**0.5
            ).bfloat16()
            weights[name] = weight.to(torch.float8_e4m3fn)
            weights[name + "_scale_inv"] = scale
        elif name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".gate.e_score_correction_bias")
            weights[name] = torch.linspace(-0.25, 0.25, param.numel())
        elif param.ndim == 1:
            weights[name] = (
                torch.ones(param.shape, dtype=param.dtype)
                if "norm" in name
                else param.detach().cpu().clone()
            )
        else:
            weight = (
                torch.randn(param.shape, generator=generator) / param.shape[-1] ** 0.5
            )
            dtype = torch.bfloat16 if name.endswith(".gate.weight") else param.dtype
            weights[name] = weight.to(dtype)
    save_file(weights, Path(path) / "model.safetensors")
    return [
        module.prefix for module in model.modules() if isinstance(module, LinearBase)
    ]


def _check_native_experts(model):
    mlp = model.model.layers[0].mlp
    assert mlp.use_mega_moe
    assert mlp.experts.weight_format.name == "fp8_block_sm90"
    assert mlp.experts.hidden_size == mlp.routed_expert_down_proj.output_size == 256
    assert mlp.experts.num_shared_experts == 0
    assert mlp.shared_experts.gate_up_proj.weight.dtype == torch.bfloat16
    assert mlp.routed_expert_up_proj.weight.dtype == torch.bfloat16
    assert mlp.experts.w13_weight_scale.dtype == torch.float32
    assert torch.isfinite(mlp.experts.w13_weight_scale).all()


@pytest.fixture
def flash_fp8_checkpoint(vllm_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = HYV4FlashConfig(
        vocab_size=256,
        hidden_size=512,
        routed_expert_hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=1,
        mlp_layer_types=["sparse"],
        layer_types=["sliding_attention"],
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=3.998,
        compressed_indexer_group_topk=32,
        compressed_indexer_candidate_topk=128,
        index_topk=128,
        sliding_window=128,
        max_position_embeddings=512,
        num_nextn_predict_layers=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.architectures = ["HYV4FlashForCausalLM"]
    config.save_pretrained(tmp_path)
    kwargs = dict(
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        block_size=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        gpu_memory_utilization=0.1,
        seed=17,
    )
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        enforce_eager=True,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        **kwargs,
    ) as runner:
        (ignored,) = runner.apply_model(partial(_write_checkpoint, path=str(tmp_path)))
    config.quantization_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ignored,
    }
    config.save_pretrained(tmp_path)
    return str(tmp_path), kwargs


def _generation_inputs():
    prompts = [
        {"prompt_token_ids": [3 + i % 251 for i in range(length)]}
        for length in (257, 17)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=8)
    return prompts, sampling


def test_flash_block_fp8_mega_moe_generation(vllm_runner, flash_fp8_checkpoint):
    """Packed checkpoint loading and latent/shared experts preserve generation."""
    path, kwargs = flash_fp8_checkpoint
    prompts, sampling = _generation_inputs()
    outputs = []
    for backend, eager in (
        ("triton", True),
        ("deep_gemm_mega_moe", True),
        ("deep_gemm_mega_moe", False),
    ):
        with vllm_runner(
            path,
            load_format="safetensors",
            enable_expert_parallel=True,
            enforce_eager=eager,
            compilation_config={"cudagraph_capture_sizes": [1, 2]},
            kernel_config={"moe_backend": backend, "enable_jit_warmup": True},
            **kwargs,
        ) as runner:
            if backend == "deep_gemm_mega_moe":
                runner.apply_model(_check_native_experts)
            result = runner.generate_w_logprobs(prompts, sampling)
            assert all(len(output[0]) == 4 for output in result)
            outputs.append(result)
    for name, output in zip(("MegaMoE eager", "MegaMoE CUDA graph"), outputs[1:]):
        check_logprobs_close(
            outputs_0_lst=outputs[0],
            outputs_1_lst=output,
            name_0="Triton FP8",
            name_1=name,
        )


@multi_gpu_test(num_gpus=2)
def test_flash_block_fp8_mega_moe_ep(vllm_runner, flash_fp8_checkpoint):
    """TP=1, EP=2 preserves outputs with unequal prefill and idle DP ranks."""
    path, kwargs = flash_fp8_checkpoint
    prompts, sampling = _generation_inputs()
    with vllm_runner(
        path,
        enforce_eager=True,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        **kwargs,
    ) as runner:
        expected = runner.generate_w_logprobs(prompts, sampling)

    async def generate_ep(eager):
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=path,
                data_parallel_size=2,
                data_parallel_backend="mp",
                enable_expert_parallel=True,
                enforce_eager=eager,
                compilation_config={"cudagraph_capture_sizes": [1, 2]},
                kernel_config={
                    "moe_backend": "deep_gemm_mega_moe",
                    "enable_jit_warmup": True,
                },
                **kwargs,
            )
        )

        async def generate(rank):
            async for result in engine.generate(
                prompts[rank], sampling, f"rank-{rank}", data_parallel_rank=rank
            ):
                output = result.outputs[0]
            assert len(output.token_ids) == sampling.max_tokens
            return list(output.token_ids), output.text, output.logprobs

        try:
            await engine.collective_rpc("apply_model", args=(_check_native_experts,))
            results = await asyncio.gather(generate(0), generate(1))
            # Only rank 0 gets a request; rank 1 must still join the MoE collectives.
            single = await generate(0)
            return results, [single]
        finally:
            engine.shutdown()

    for eager in (True, False):
        results, single = asyncio.run(generate_ep(eager))
        for reference, actual in ((expected, results), (expected[:1], single)):
            check_logprobs_close(
                outputs_0_lst=reference,
                outputs_1_lst=actual,
                name_0="Triton EP=1",
                name_1=f"MegaMoE EP=2, eager={eager}",
            )
