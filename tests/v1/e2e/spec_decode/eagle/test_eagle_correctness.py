# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict

import pytest
import torch

from tests.utils import (
    get_attn_backend_list_based_on_platform,
    single_gpu_only,
)
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform

from ..utils import compute_acceptance_rate
from .utils import _run_eagle_correctness


def _draft_graph_cap_state(worker):
    """Inspect a real captured runner without changing its execution path."""
    from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp

    runner = worker.model_runner
    assert type(runner).__module__ == "vllm.v1.worker.gpu.model_runner"
    first = runner.speculator.prefill_cudagraph_manager
    later = runner.speculator.decode_cudagraph_manager
    target = runner.cudagraph_manager
    kwargs = dict(num_reqs=2, num_tokens=128, uniform_token_count=None)
    synced = DPSyncState(
        torch.tensor([128, 128], dtype=torch.int32), None, None, False, 2
    )
    desc, reused = dispatch_cg_and_sync_dp(
        first, **kwargs, dp_size=2, dp_rank=0, dp_sync_state=synced
    )
    assert reused is not None

    def descriptor_state(desc):
        return asdict(desc) | {"cg_mode": desc.cg_mode.name}

    return {
        "target_sizes": target.compilation_config.cudagraph_capture_sizes,
        "first_sizes": first.compilation_config.cudagraph_capture_sizes,
        "first_full": [
            descriptor_state(d) for d in first._capture_descs[CUDAGraphMode.FULL]
        ],
        "later_full": [
            descriptor_state(d) for d in later._capture_descs[CUDAGraphMode.FULL]
        ],
        "target_dispatch": descriptor_state(
            target.dispatch(**kwargs, num_active_loras=0)
        ),
        "first_dispatch": descriptor_state(desc),
        "sync_reused": reused is synced,
        "sync_tokens": reused.num_tokens_across_dp.tolist(),
        "later_shares_config": later.vllm_config is runner.vllm_config,
    }


@single_gpu_only
@pytest.mark.parametrize("breakable", [True, False])
def test_qwen3_eagle3_draft_prefill_graph_ceiling(monkeypatch, vllm_runner, breakable):
    """Limit draft prefill graphs while keeping real FULL decode and output parity."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1" if breakable else "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    outputs, states = [], []
    for cap in (None, 64):
        with vllm_runner(
            "Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            dtype="bfloat16",
            trust_remote_code=False,
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=256,
            gpu_memory_utilization=0.7,
            enable_chunked_prefill=True,
            enforce_eager=False,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=False,
            seed=0,
            attention_config={"backend": get_attn_backend_list_based_on_platform()[0]},
            compilation_config={
                "mode": 3,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": sizes,
            },
            speculative_config={
                "method": "eagle3",
                "model": (
                    "inference-optimization/"
                    "Qwen3-8B-from-Qwen3-8B_regen-speculators.eagle3-qwen3arch-ckpt1"
                ),
                "revision": "689d89ab2b97a8aa70728763a6382ee931f51aef",
                "num_speculative_tokens": 3,
                "draft_prefill_cudagraph_max_tokens": cap,
            },
        ) as model:
            states.append(
                model.llm.llm_engine.collective_rpc(_draft_graph_cap_state)[0]
            )
            outputs.append(
                model.llm.generate(
                    [
                        {"prompt_token_ids": [285] * 96},
                        {"prompt_token_ids": [286] * 100},
                    ],
                    SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
                    use_tqdm=False,
                )
            )
            drafts = next(
                metric.value
                for metric in model.llm.get_metrics()
                if metric.name == "vllm:spec_decode_num_drafts"
            )
            assert drafts > 0
            assert compute_acceptance_rate(model.llm.get_metrics()) > 0
    default, limited = states
    assert default["target_sizes"] == limited["target_sizes"] == sizes
    assert limited["first_sizes"] == [n for n in sizes if n <= 64]
    assert default["first_full"] == limited["first_full"]
    assert default["later_full"] == limited["later_full"]
    assert limited["later_shares_config"]
    assert limited["target_dispatch"]["cg_mode"] == "PIECEWISE"
    assert default["first_dispatch"]["cg_mode"] == "PIECEWISE"
    assert limited["first_dispatch"]["cg_mode"] == "NONE"
    assert limited["first_dispatch"]["num_tokens"] == 128
    assert limited["sync_reused"] and limited["sync_tokens"] == [128, 128]
    assert [o.outputs[0].token_ids for o in outputs[0]] == [
        o.outputs[0].token_ids for o in outputs[1]
    ]
    assert all(len(o.outputs[0].token_ids) == 16 for arm in outputs for o in arm)


@single_gpu_only
@pytest.mark.skipif(
    current_platform.is_device_capability_family(100),
    reason="DeepSeek head_dim=192 not supported on SM100/SM110 (Blackwell)",
)
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
            "auto",
            0.0,
        ),
    ],
    ids=["deepseek_eagle"],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_light(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )


@single_gpu_only
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "transformers",
            0.8,
            # TODO(hmellor): figure out why memory usage is so high
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(),
                reason="Feature is experimental and uses too much memory in CUDA CI",
            ),
        ),
        (
            (
                "eagle3",
                "Qwen/Qwen3-VL-8B-Instruct",
                "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
                1,
            ),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
            # TODO: Re-measure the reference threshold when re-enabling this case.
            # Text-only mode starts but loses target correctness with ROCm/TRITON;
            # full multimodal profiling currently fails during engine startup.
            marks=pytest.mark.skip(
                reason="Qwen2.5-VL Eagle3 is not yet reliable in this test setup"
            ),
        ),
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen3_eagle3-transformers",
        "qwen3_vl_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle3",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_medium(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )
