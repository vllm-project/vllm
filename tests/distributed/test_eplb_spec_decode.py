# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from types import SimpleNamespace

import lm_eval
import pytest
import torch

from tests.utils import large_gpu_mark
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.gpu.eplb_utils import EPLBController


def get_model_args(
    model_name: str,
    spec_model_name: str | None,
    spec_method: str,
    tp_size: int,
    model_max_len: int,
    num_speculative_tokens: int = 1,
    use_async: bool = True,
) -> dict:
    speculative_config = {
        "method": spec_method,
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
        "max_model_len": model_max_len,
    }
    if spec_method == "dspark":
        speculative_config["model"] = model_name
        speculative_config["draft_sample_method"] = "probabilistic"
    eplb_config = {
        "num_redundant_experts": tp_size,
        "window_size": 128,
        "step_interval": 1024,
        "log_balancedness": False,
        "use_async": use_async,
    }
    model_args = {
        "pretrained": model_name,
        "dtype": "auto",
        "add_bos_token": True,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": 0.7,
        "speculative_config": speculative_config,
        "enable_expert_parallel": True,
        "eplb_config": eplb_config,
        "enable_eplb": True,
        "max_model_len": model_max_len,
    }
    if spec_method == "dspark":
        model_args["trust_remote_code"] = True
    return model_args


pytestmark = pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="EPLB with Spec Decode is a work in progress on ROCm.",
)


class FakeEplbState:
    instances: list[FakeEplbState] = []

    def __init__(self, parallel_config, device: torch.device):
        self.add_model_calls: list[tuple[object, object]] = []
        FakeEplbState.instances.append(self)

    def add_model(self, model: object, model_config: object) -> None:
        self.add_model_calls.append((model, model_config))


def _make_dsv4_dspark_hf_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=128,
        num_hidden_layers=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_hash_layers=0,
        n_shared_experts=1,
        moe_intermediate_size=128,
        hc_mult=1,
        hc_eps=1e-5,
        rms_norm_eps=1e-5,
        dspark_target_layer_ids=[0],
        dspark_markov_rank=8,
        index_topk=4,
        head_dim=64,
        num_attention_heads=4,
        vocab_size=256,
        n_mtp_layers=2,
        enable_confidence_head=False,
        compress_ratios=[1, 1],
    )


@pytest.fixture
def dspark_vllm_config(dist_init):
    hf_config = _make_dsv4_dspark_hf_config()
    model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_config=hf_config, model="dspark"
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_eplb=True,
            enable_elastic_ep=False,
            eplb_config=SimpleNamespace(num_redundant_experts=4),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=model_config,
        ),
    )


def _build_dspark_draft(vllm_config, monkeypatch: pytest.MonkeyPatch):
    if not current_platform.is_cuda():
        pytest.skip("DSpark EPLB registration tests require CUDA")
    if not current_platform.is_device_capability_family(100):
        pytest.skip("DeepGEMM MegaMoE requires SM100")

    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.get_current_vllm_config",
        lambda: vllm_config,
    )
    with (
        set_default_torch_dtype(vllm_config.model_config.dtype),
        torch.device("cuda"),
    ):
        return DSparkDeepseekV4ForCausalLM(vllm_config=vllm_config)


def _make_moe_topology(
    *,
    num_routed_experts: int = 8,
    num_redundant_experts: int = 4,
    num_expert_groups: int = 1,
) -> SimpleNamespace:
    num_physical_experts = num_routed_experts + num_redundant_experts
    return SimpleNamespace(
        num_routed_experts=num_routed_experts,
        num_redundant_experts=num_redundant_experts,
        num_physical_experts=num_physical_experts,
        num_logical_experts=num_routed_experts,
        num_expert_groups=num_expert_groups,
    )


def test_eplb_state_accepts_matching_dsv4_draft_and_target():
    state = SimpleNamespace(model_states={})
    draft = _make_moe_topology()
    target = _make_moe_topology()
    state.model_states["draft"] = SimpleNamespace(model=draft)

    EplbState.validate_ep_configuration(state, target)


def test_eplb_state_rejects_mismatched_dsv4_draft_redundant_experts():
    state = SimpleNamespace(model_states={})
    draft = _make_moe_topology(num_redundant_experts=0)
    target = _make_moe_topology(num_redundant_experts=4)
    state.model_states["draft"] = SimpleNamespace(model=draft)

    with pytest.raises(RuntimeError, match="mismatch"):
        EplbState.validate_ep_configuration(state, target)


def test_eplb_registers_dspark_draft_model(
    dspark_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    """DSpark MoE drafts register with EPLB like MTP/Eagle drafters."""
    FakeEplbState.instances.clear()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.eplb_utils.EplbState",
        FakeEplbState,
    )

    draft = _build_dspark_draft(dspark_vllm_config, monkeypatch)
    assert get_mixture_of_experts_model(draft) is draft

    controller = EPLBController(dspark_vllm_config.parallel_config, torch.device("cpu"))
    controller.prepare_load()
    speculator = SimpleNamespace(model=draft, eplb_state=None)

    def set_eplb_state(state) -> None:
        speculator.eplb_state = state

    speculator.set_eplb_state = set_eplb_state

    registered = controller.maybe_register_speculator(
        speculator,
        dspark_vllm_config.speculative_config,
        load_dummy_weights=False,
    )

    assert registered is True
    assert controller.state is not None
    assert controller.state.add_model_calls == [
        (draft, dspark_vllm_config.speculative_config.draft_model_config)
    ]
    assert speculator.eplb_state is controller.state


def test_eplb_skips_dsv41_dspark_registration(
    dspark_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    """V4.1 DSpark drafts use a different expert topology and are not registered."""
    FakeEplbState.instances.clear()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.eplb_utils.EplbState",
        FakeEplbState,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.eplb_utils.get_mixture_of_experts_model",
        lambda model: model,
    )

    dspark_vllm_config.speculative_config.draft_model_config.hf_config.model_type = (
        "deepseek_v41"
    )
    draft = SimpleNamespace()
    controller = EPLBController(dspark_vllm_config.parallel_config, torch.device("cpu"))
    controller.prepare_load()
    speculator = SimpleNamespace(model=draft, eplb_state=None)
    speculator.set_eplb_state = lambda state: setattr(speculator, "eplb_state", state)

    registered = controller.maybe_register_speculator(
        speculator,
        dspark_vllm_config.speculative_config,
        load_dummy_weights=False,
    )

    assert registered is False
    assert controller.state is not None
    assert controller.state.add_model_calls == []


def test_eplb_skips_dspark_registration_with_dummy_weights(
    dspark_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    FakeEplbState.instances.clear()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.eplb_utils.EplbState",
        FakeEplbState,
    )

    draft = _build_dspark_draft(dspark_vllm_config, monkeypatch)
    controller = EPLBController(dspark_vllm_config.parallel_config, torch.device("cpu"))
    controller.prepare_load()
    speculator = SimpleNamespace(model=draft, eplb_state=None)

    def set_eplb_state(state) -> None:
        speculator.eplb_state = state

    speculator.set_eplb_state = set_eplb_state

    registered = controller.maybe_register_speculator(
        speculator,
        dspark_vllm_config.speculative_config,
        load_dummy_weights=True,
    )

    assert registered is False
    assert controller.state is not None
    assert controller.state.add_model_calls == []


@pytest.mark.parametrize(
    "model_setup",
    [
        pytest.param(
            ("mtp", "Qwen/Qwen3-Next-80B-A3B-Instruct", None, 4, 0.86, 1),
            marks=large_gpu_mark(min_gb=80),
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
                0.92,
                1,
            ),
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues"),
        ),
        pytest.param(
            (
                "dspark",
                "deepseek-ai/DeepSeek-V4-Flash-DSpark",
                None,
                4,
                0.92,
                7,
            ),
            marks=large_gpu_mark(min_gb=80),
        ),
    ],
    ids=["qwen3_next_mtp", "llama4_eagle", "dsv4_dspark"],
)
def test_eplb_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str | None, int, float, int],
):
    """Test the correctness of EPLB speculative decoding with GSM8K dataset.

    Applicable to MoE models with mtp, eagle, or dspark spec decode.
    """
    (
        method,
        model_name,
        spec_model_name,
        tp_size,
        expected_gsm8k_value,
        num_speculative_tokens,
    ) = model_setup
    if method == "dspark" and not current_platform.is_device_capability_family(100):
        pytest.skip("DSV4 DSpark EPLB requires SM100 for MegaMoE")

    TASK = "gsm8k"
    FILTER = "exact_match,strict-match"
    RTOL = 0.03

    model_args = get_model_args(
        model_name=model_name,
        spec_model_name=spec_model_name,
        spec_method=method,
        tp_size=tp_size,
        model_max_len=4096,
        num_speculative_tokens=num_speculative_tokens,
    )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=TASK,
        batch_size=64,
        num_fewshot=8,
    )
    measured_value = results["results"][TASK][FILTER]
    assert (
        measured_value - RTOL < expected_gsm8k_value
        and measured_value + RTOL > expected_gsm8k_value
    ), f"Expected: {expected_gsm8k_value} |  Measured: {measured_value}"


@large_gpu_mark(min_gb=80)
def test_eplb_spec_decode_qwen3_next_mtp_async() -> None:
    """Ensure async EPLB works with MTP speculative decoding for Qwen3-Next."""
    TASK = "gsm8k"
    FILTER = "exact_match,strict-match"
    RTOL = 0.03
    expected_gsm8k_value = 0.86

    model_args = get_model_args(
        model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        spec_model_name=None,
        spec_method="mtp",
        tp_size=4,
        model_max_len=4096,
        use_async=True,
    )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=TASK,
        batch_size=64,
        num_fewshot=8,
    )
    measured_value = results["results"][TASK][FILTER]
    assert (
        measured_value - RTOL < expected_gsm8k_value
        and measured_value + RTOL > expected_gsm8k_value
    ), f"Expected: {expected_gsm8k_value} |  Measured: {measured_value}"
