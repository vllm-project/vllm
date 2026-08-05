# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
    dtype: torch.dtype = torch.int32,
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
        dtype=dtype,
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16])
def test_routed_experts_capturer_narrows_router_ids(dtype):
    capturer = _capturer_with_buffer(dtype=dtype)
    topk = torch.tensor([[1, 2], [254, 255]], dtype=torch.int64)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    assert capturer.device_buffer.dtype == dtype
    assert capturer.device_buffer[:2, 0, :].tolist() == topk.tolist()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_finishes_pending_artifact_output():
    from vllm.distributed.artifact_connector.connector import ArtifactConnectorOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    pending = Mock()
    artifact_output = ArtifactConnectorOutput({})
    pending.finish.return_value = artifact_output
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        pending_artifact_output=pending,
    ).get_output()

    pending.to_cpu_nonblocking.assert_called_once_with()
    pending.finish.assert_called_once_with(set())
    assert output.artifact_connector_output is artifact_output


def test_model_runner_initializes_capture(monkeypatch):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    connector = Mock()
    constructor = Mock(return_value=connector)
    monkeypatch.setattr(model_runner, "ArtifactWorkerConnector", constructor)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=0))
    runner.model = Mock()
    kv_cache_config = Mock()

    runner.init_artifact_connector(kv_cache_config)

    constructor.assert_called_once_with(
        model=runner.model,
        kv_cache_config=kv_cache_config,
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
    )
    assert runner.artifact_connector is connector


def test_artifact_worker_connector_owns_capture(monkeypatch):
    import vllm.distributed.artifact_connector.worker as artifact_worker

    snapshot = torch.tensor([1, 2, 3])
    capturer = Mock()
    capturer.get_routing_data.return_value = snapshot
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(artifact_worker, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(artifact_worker, "bind_routed_experts_capturer", bind)
    monkeypatch.setattr(
        artifact_worker,
        "get_tp_group",
        lambda: SimpleNamespace(is_first_rank=False),
    )

    config = SimpleNamespace(
        artifact_config=SimpleNamespace(enable_return_routed_experts=True),
        kv_transfer_config=None,
    )
    model = Mock()
    connector = artifact_worker.ArtifactWorkerConnector(
        vllm_config=config,
        model=model,
        kv_cache_config=SimpleNamespace(),
        max_num_batched_tokens=32,
    )

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=config,
    )
    bind.assert_called_once_with(model, capturer)
    assert connector.capture_routed_experts(3) is snapshot
    capturer.get_routing_data.assert_called_once_with(3)


def test_artifact_worker_connector_shm_capacity(monkeypatch, tmp_path):
    import vllm.distributed.artifact_connector.worker as artifact_worker

    tp_group = SimpleNamespace(is_first_rank=True, world_size=1)
    store_constructor = Mock()
    monkeypatch.setattr(artifact_worker, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(artifact_worker, "RoutedExpertsCapturer", Mock())
    monkeypatch.setattr(artifact_worker, "bind_routed_experts_capturer", Mock())
    monkeypatch.setattr(
        artifact_worker,
        "get_routing_shape_and_dtype",
        lambda _: ((2,), np.int32),
    )
    monkeypatch.setattr(
        artifact_worker,
        "resolve_kv_cache_block_sizes",
        lambda *_: (32, 16),
    )
    monkeypatch.setattr(
        artifact_worker, "LocalSharedMemoryArtifactStore", store_constructor
    )
    monkeypatch.setattr(artifact_worker, "BackgroundArtifactStore", Mock())
    monkeypatch.setattr(artifact_worker, "RoutedExpertsArtifactBuffer", Mock())

    config = SimpleNamespace(
        artifact_config=SimpleNamespace(
            max_shm_bytes=None,
            shm_dir=str(tmp_path),
            shm_ttl_seconds=60,
        ),
        kv_transfer_config=None,
        instance_id="instance",
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    kwargs = dict(
        vllm_config=config,
        model=Mock(),
        kv_cache_config=SimpleNamespace(num_blocks=10),
        max_num_batched_tokens=32,
    )

    artifact_worker.ArtifactWorkerConnector(**kwargs)
    assert store_constructor.call_args.kwargs["max_bytes"] == 2560

    config.kv_transfer_config = SimpleNamespace(is_kv_transfer_instance=True)
    with pytest.raises(AssertionError):
        artifact_worker.ArtifactWorkerConnector(**kwargs)


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        artifact_config=SimpleNamespace(enable_return_routed_experts=True),
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
