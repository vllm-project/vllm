# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    bind_routed_experts_capturer,
    get_routed_experts_attn_gid,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _make_model_config(hf_config):
    num_experts_per_token = ModelArchConfigConvertorBase(
        hf_config, hf_config
    ).get_num_experts_per_token()
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(
            num_experts_per_token=num_experts_per_token,
        ),
    )
    model_config.get_num_experts = lambda: hf_config.num_experts
    model_config.get_num_experts_per_tok = lambda: (
        ModelConfig.get_num_experts_per_tok(model_config)
    )
    model_config.get_total_num_hidden_layers = lambda: hf_config.num_hidden_layers
    return model_config


def test_routed_experts_manager_uses_gemma4_top_k_experts():
    hf_config = SimpleNamespace(
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_routed_experts_manager_uses_kimi_k3_experts_per_token():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_masks_padding_before_capture_and_eplb():
    router = _make_router()
    captured = []
    is_padding = torch.tensor([False, True])

    router.set_capture_fn(lambda ids: captured.append(ids.clone()))
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router."
            "is_forward_context_available",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router."
            "get_forward_context",
            return_value=SimpleNamespace(is_padding=is_padding),
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router."
            "envs.VLLM_MOE_SKIP_PADDING",
            True,
        ),
    ):
        topk_weights, topk_ids = router.select_experts(
            hidden_states=torch.empty(1),
            router_logits=torch.empty(1),
        )

    assert torch.equal(captured[0], torch.tensor([[1, 2], [-1, -1]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [9, 9]]))
    assert torch.equal(topk_weights, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_public_binding_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            types.SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
        )


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_attention_group_is_shared_and_fail_closed(monkeypatch):
    class FullAttentionSpec:
        pass

    monkeypatch.setattr(f"{_REC_MODULE}.FullAttentionSpec", FullAttentionSpec)
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object()),
            SimpleNamespace(kv_cache_spec=FullAttentionSpec()),
        ]
    )
    assert get_routed_experts_attn_gid(config) == 1

    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(SimpleNamespace(kv_cache_groups=[]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_returns_existing_routed_experts_field():
    from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    routed_experts = RoutedExpertsTensors(
        routing_data=torch.arange(6, dtype=torch.int32, device="cuda").reshape(3, 1, 2),
        slot_mapping=torch.tensor([11, 12, 13], device="cuda"),
    )
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        routed_experts=routed_experts,
    ).get_output()

    assert output.routed_experts is not None
    assert output.routed_experts.routing_data[:, 0, 0].tolist() == [0, 2, 4]
    assert output.routed_experts.slot_mapping.tolist() == [11, 12, 13]


@pytest.mark.parametrize("rank", [0, 1])
def test_all_tp_ranks_initialize_capture(monkeypatch, rank):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(model_runner, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(model_runner, "bind_routed_experts_capturer", bind)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=rank))
    runner.kv_cache_config = SimpleNamespace()
    runner.model = Mock()

    runner.init_routed_experts_capturer()

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
        kv_cache_config=runner.kv_cache_config,
    )
    bind.assert_called_once_with(runner.model, capturer)
    assert runner.routed_experts_capturer is capturer


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported
