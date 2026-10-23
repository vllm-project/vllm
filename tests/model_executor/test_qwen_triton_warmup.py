# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp as mtp_ops  # noqa: E501
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_gdn_decode_post_conv_mtp_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_kernel,
)
from vllm.utils.platform_utils import num_compute_units

from ..utils import create_new_process_for_each_test

NUM_SPEC = 3


def _gdn_update_kernel():
    # triton.heuristics wraps the JITFunction that holds the compile cache.
    return fused_sigmoid_gating_delta_rule_update_kernel.fn


def _gdn_update_compiled_keys() -> set:
    return {
        key
        for caches in _gdn_update_kernel().device_caches.values()
        for key in caches[0]
    }


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only Triton kernel")
@pytest.mark.parametrize("max_num_reqs", [1, 1024])
def test_qwen_gdn_fused_mtp_warmup_covers_reachable_tile_sizes(
    max_num_reqs: int,
) -> None:
    """The MTP decode warmup compiles every V tile size the wrapper can pick
    for up to max_num_reqs requests."""
    k = v = 128
    config = replace(
        _cuda_gdn_config(),
        k=k,
        v=v,
        state_stride_token=2 * v * k,
        fused_spec_decode=True,
    )
    device = torch.device("cuda")
    kernel = mtp_ops.fused_gdn_decode_post_conv_mtp_triton
    with patch.object(
        mtp_ops, "fused_gdn_decode_post_conv_mtp_triton", wraps=kernel
    ) as wrapped:
        _warm_fused_gdn_decode_post_conv_mtp_kernel(
            device,
            config,
            torch.bfloat16,
            num_spec_tokens=3,
            max_num_reqs=max_num_reqs,
        )
    torch.accelerator.synchronize(device)

    num_cus = num_compute_units(device.index)
    tiles = set()
    for call in wrapped.call_args_list:
        num_reqs = call.kwargs["cu_seqlens"].numel() - 1
        assert num_reqs <= max_num_reqs
        heads = num_reqs * config.hv
        tiles.add(32 if heads < num_cus else 64 if heads <= 4 * num_cus else v)
    assert tiles == ({32} if max_num_reqs == 1 else {32, 64, v})


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@create_new_process_for_each_test("spawn")
def test_qwen_gdn_mtp_mixed_batch_reuses_warmed_gdn_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec-decode rows batched with prefill chunks must not compile the GDN
    update kernel after startup.

    The long prompt needs several chunks, so the short request verifies its
    drafts in the same steps. The engine runs in this process so the test can
    read the kernel's compile cache.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    max_num_batched_tokens = 2048
    llm = LLM(
        "Qwen/Qwen3.5-0.8B",
        load_format="dummy",
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={"method": "mtp", "num_speculative_tokens": NUM_SPEC},
    )
    warmed = _gdn_update_compiled_keys()
    spec_launches: list[bool] = []
    _gdn_update_kernel().add_pre_run_hook(
        lambda *args, **kwargs: spec_launches.append(kwargs["IS_SPEC_DECODING"])
    )

    short = TokensPrompt(prompt_token_ids=list(range(1, 33)))
    long = TokensPrompt(
        prompt_token_ids=list(range(33, 33 + 3 * max_num_batched_tokens))
    )
    llm.generate(
        [short, long],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    # Spec rows reached this Triton kernel, so the mixed step ran.
    assert any(spec_launches)
    compiled = _gdn_update_compiled_keys()
    assert compiled == warmed, f"compiled after startup: {compiled - warmed}"
