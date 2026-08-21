# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.model_executor.model_loader.reload import RuntimeReloadSession
from vllm.model_executor.reload_arena import get_reload_arena


class _RuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fp8_weight = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.fp4_weight = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.uint8), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.ones(2))
        self.runtime_alpha = torch.ones(2)
        get_reload_arena(self).put("runtime.gscale", torch.ones(2))


def test_runtime_reload_copies_in_place_without_pwal(monkeypatch) -> None:
    def fail_pwal(*args, **kwargs):
        raise AssertionError("runtime reload must not call PWAL")

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        fail_pwal,
    )
    model = _RuntimeModel()
    weight = model.fp8_weight
    scale = model.weight_scale
    weight_ptr = weight.data_ptr()
    scale_ptr = scale.data_ptr()

    session = RuntimeReloadSession(model)
    session.start()
    loaded = session.load_weights(
        [
            (
                "fp8_weight",
                torch.tensor([1, 2, 3, 4], dtype=torch.float8_e4m3fn),
            ),
            ("weight_scale", torch.tensor([2.0, 4.0])),
        ]
    )

    # Writes are visible immediately; finish has no PWAL/finalization work.
    assert loaded == {"fp8_weight", "weight_scale"}
    assert torch.equal(model.fp8_weight.float(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(model.weight_scale, torch.tensor([2.0, 4.0]))
    assert session.finish() == loaded
    assert model.fp8_weight is weight
    assert model.weight_scale is scale
    assert model.fp8_weight.data_ptr() == weight_ptr
    assert model.weight_scale.data_ptr() == scale_ptr


def test_runtime_reload_rejects_incompatible_tensor_before_its_write() -> None:
    model = _RuntimeModel()
    original = model.weight_scale.clone()
    session = RuntimeReloadSession(model)
    session.start()

    with pytest.raises(ValueError, match="Incompatible runtime tensor"):
        session.load_weights([("weight_scale", torch.ones(3))])

    assert torch.equal(model.weight_scale, original)


def test_runtime_reload_abort_does_not_rollback_in_place_writes() -> None:
    model = _RuntimeModel()
    session = RuntimeReloadSession(model)
    session.start()
    session.load_weights([("fp4_weight", torch.full((4,), 9, dtype=torch.uint8))])

    session.abort()

    assert torch.equal(model.fp4_weight, torch.full((4,), 9, dtype=torch.uint8))
    assert not session.active


def test_runtime_reload_updates_plain_and_arena_tensors() -> None:
    model = _RuntimeModel()
    alpha = model.runtime_alpha
    arena_scale = get_reload_arena(model).slots()["runtime.gscale"]
    session = RuntimeReloadSession(model)
    session.start()
    session.load_weights(
        [
            ("runtime_alpha", torch.tensor([3.0, 4.0])),
            ("@reload_arena/:runtime.gscale", torch.tensor([5.0, 6.0])),
        ]
    )
    session.finish()

    assert model.runtime_alpha is alpha
    assert get_reload_arena(model).slots()["runtime.gscale"] is arena_scale
    assert torch.equal(alpha, torch.tensor([3.0, 4.0]))
    assert torch.equal(arena_scale, torch.tensor([5.0, 6.0]))


def test_runtime_nccl_broadcasts_directly_into_live_parameter(monkeypatch) -> None:
    model = _RuntimeModel()
    target = model.fp4_weight

    class _Group:
        def __init__(self) -> None:
            self.destination = None

        def broadcast(self, tensor, **kwargs) -> None:
            self.destination = tensor
            tensor.fill_(5)

    config = WeightTransferConfig(backend="nccl", weight_format="runtime")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = NCCLWeightTransferEngine(config, vllm_config, torch.device("cpu"), model)
    group = _Group()
    engine.model_update_group = group
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime NCCL must not allocate a receive tensor")
        ),
    )

    engine.start_weight_update()
    engine.receive_weights(
        NCCLWeightTransferUpdateInfo(
            names=["fp4_weight"],
            dtype_names=["uint8"],
            shapes=[[4]],
        )
    )
    engine.finish_weight_update()

    assert group.destination is target
    assert torch.equal(target, torch.full((4,), 5, dtype=torch.uint8))


def test_runtime_nccl_rejects_packed_staging_buffers() -> None:
    model = _RuntimeModel()
    config = WeightTransferConfig(backend="nccl", weight_format="runtime")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = NCCLWeightTransferEngine(config, vllm_config, torch.device("cpu"), model)
    engine.model_update_group = object()
    engine.packed = True
    engine.start_weight_update()

    with pytest.raises(ValueError, match="Packed NCCL uses staging buffers"):
        engine.receive_weights(
            NCCLWeightTransferUpdateInfo(
                names=["fp4_weight"],
                dtype_names=["uint8"],
                shapes=[[4]],
            )
        )


def test_runtime_nccl_rejects_model_parallel_broadcast() -> None:
    model = _RuntimeModel()
    config = WeightTransferConfig(backend="nccl", weight_format="runtime")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, pipeline_parallel_size=1
        ),
        model_config=SimpleNamespace(),
    )
    engine = NCCLWeightTransferEngine(config, vllm_config, torch.device("cpu"), model)

    with pytest.raises(ValueError, match="rank-local TP/PP/EP layouts"):
        engine.start_weight_update()


def test_runtime_ipc_copies_directly_from_mapped_source(monkeypatch) -> None:
    pytest.importorskip("ray")
    from vllm.distributed.weight_transfer import ipc_engine as ipc_module
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCWeightTransferEngine,
        IPCWeightTransferUpdateInfo,
    )

    model = _RuntimeModel()
    target = model.fp4_weight
    source = torch.full((4,), 6, dtype=torch.uint8)
    config = WeightTransferConfig(backend="ipc", weight_format="runtime")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = IPCWeightTransferEngine(
        config, vllm_config, torch.device("cuda", 0), model
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="gpu-0"),
    )
    monkeypatch.setattr(ipc_module, "rebuild_cuda_tensor", lambda *args: source)
    engine.start_weight_update()
    engine.receive_weights(
        IPCWeightTransferUpdateInfo(
            names=["fp4_weight"],
            dtype_names=["uint8"],
            shapes=[[4]],
            ipc_handles=[{"gpu-0": (None,) * 7}],
        )
    )
    engine.finish_weight_update()

    assert model.fp4_weight is target
    assert model.fp4_weight.data_ptr() == target.data_ptr()
    assert torch.equal(target, source)


def test_runtime_ipc_rejects_packed_staging_buffers() -> None:
    pytest.importorskip("ray")
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCWeightTransferEngine,
        IPCWeightTransferUpdateInfo,
    )

    model = _RuntimeModel()
    config = WeightTransferConfig(backend="ipc", weight_format="runtime")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = IPCWeightTransferEngine(
        config, vllm_config, torch.device("cuda", 0), model
    )
    engine.packed = True
    engine.start_weight_update()

    with pytest.raises(ValueError, match="Packed IPC uses staging buffers"):
        engine.receive_weights(
            IPCWeightTransferUpdateInfo(
                names=["fp4_weight"],
                dtype_names=["uint8"],
                shapes=[[4]],
                ipc_handles={"gpu-0": (None,) * 7},
                tensor_sizes=[4],
            )
        )
