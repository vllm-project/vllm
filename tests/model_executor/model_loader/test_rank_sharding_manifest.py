# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ModuleSource,
    RankLocalWeightSource,
    TrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.model_executor.model_loader.reload.sharding import (
    RankScope,
    RankShard,
    RankShardingManifest,
    capture_rank_sharding,
    get_rank_sharding_manifest,
    install_sharding_recorders,
)
from vllm.model_executor.model_loader.reload.source import source_name_context


class _ShardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))

        def loader(param, loaded_weight, loaded_shard_id=None):
            param.data.copy_(loaded_weight)

        self.weight.weight_loader = loader


class _Client:
    def __init__(self, manifests) -> None:
        self.manifests = manifests

    def get_rank_sharding_manifests(self):
        return self.manifests


class _Trainer(TrainerWeightTransferEngine):
    @classmethod
    def trainer_init(cls, init_info, *, client, source=None):
        raise NotImplementedError

    def send_weights(self) -> None:
        raise NotImplementedError


def test_records_rank_local_source_target_fragment() -> None:
    model = _ShardModel()
    install_sharding_recorders(model)

    with (
        capture_rank_sharding(model, reset=True),
        source_name_context("layers.0.q_proj.weight"),
    ):
        model.weight.weight_loader(model.weight, torch.ones(2), loaded_shard_id="q")

    manifest = get_rank_sharding_manifest(model)
    assert manifest.source_names == ("layers.0.q_proj.weight",)
    assert manifest.shards[0].target_name == "weight"
    assert manifest.shards[0].fragment.items == (("loaded_shard_id", "q"),)


def test_rank_local_source_only_materializes_consumed_names() -> None:
    trainer = torch.nn.Module()
    trainer.register_parameter("local", torch.nn.Parameter(torch.ones(2)))
    trainer.register_parameter("remote", torch.nn.Parameter(torch.ones(3)))
    source = RankLocalWeightSource(ModuleSource(trainer), {"local"})

    assert [item.name for item in source.metadata()] == ["local"]
    assert [name for name, _ in source] == ["local"]


def test_rank_local_source_rejects_unknown_manifest_name() -> None:
    trainer = torch.nn.Linear(2, 2, bias=False)
    source = RankLocalWeightSource(ModuleSource(trainer), {"missing"})

    try:
        source.metadata()
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing manifest source must fail closed")


def test_trainer_uses_union_of_worker_rank_manifests() -> None:
    trainer = torch.nn.Module()
    trainer.register_parameter("q", torch.nn.Parameter(torch.ones(2)))
    trainer.register_parameter("k", torch.nn.Parameter(torch.ones(2)))
    trainer.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
    manifests = [
        RankShardingManifest(RankScope(), (RankShard("q", "q_proj.weight"),)),
        RankShardingManifest(RankScope(), (RankShard("k", "k_proj.weight"),)),
    ]
    engine = _Trainer(
        client=_Client(manifests), source=ModuleSource(trainer), is_sender=True
    )

    source = engine.rank_local_source()

    assert [item.name for item in source.metadata()] == ["q", "k"]
    assert [name for name, _ in source] == ["q", "k"]


def test_trainer_rejects_inexact_worker_manifest() -> None:
    trainer = torch.nn.Linear(2, 2, bias=False)
    manifest = RankShardingManifest(
        RankScope(), (), state="unavailable", reason="no source adapter"
    )
    engine = _Trainer(
        client=_Client([manifest]), source=ModuleSource(trainer), is_sender=True
    )

    try:
        engine.rank_local_source()
    except ValueError as error:
        assert "no source adapter" in str(error)
    else:
        raise AssertionError("inexact worker manifest must fail closed")


def test_ipc_skips_records_not_consumed_by_this_rank(monkeypatch) -> None:
    model = _ShardModel()
    install_sharding_recorders(model)
    with (
        capture_rank_sharding(model, reset=True),
        source_name_context("local.weight"),
    ):
        model.weight.weight_loader(model.weight, torch.ones(2))

    config = WeightTransferConfig(backend="ipc", weight_format="checkpoint")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = IPCWeightTransferEngine(config, vllm_config, torch.device("cpu"), model)

    class _Session:
        def __init__(self) -> None:
            self.received = None

        def load_weights(self, weights) -> None:
            self.received = list(weights)

    session = _Session()
    engine.reload_session = session
    monkeypatch.setattr(
        "vllm.distributed.weight_transfer.ipc_engine.rebuild_cuda_tensor",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("remote-rank IPC tensor must not be rebuilt")
        ),
    )

    engine.receive_weights(
        IPCWeightTransferUpdateInfo(
            names=["remote.weight"],
            dtype_names=["float32"],
            shapes=[[2]],
            ipc_handles=[{"gpu": ()}],
        )
    )

    assert session.received == []


def test_nccl_receives_but_does_not_load_remote_rank_record(monkeypatch) -> None:
    model = _ShardModel()
    install_sharding_recorders(model)
    with (
        capture_rank_sharding(model, reset=True),
        source_name_context("local.weight"),
    ):
        model.weight.weight_loader(model.weight, torch.ones(2))

    config = WeightTransferConfig(backend="nccl", weight_format="checkpoint")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(), model_config=SimpleNamespace()
    )
    engine = NCCLWeightTransferEngine(config, vllm_config, torch.device("cpu"), model)

    class _Group:
        def __init__(self) -> None:
            self.broadcasts = 0

        def broadcast(self, tensor, **kwargs) -> None:
            self.broadcasts += 1

    class _Session:
        def __init__(self) -> None:
            self.received = []

        def load_weights(self, weights) -> None:
            self.received.extend(weights)

    group = _Group()
    session = _Session()
    engine.model_update_group = group
    engine.reload_session = session
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)

    engine.receive_weights(
        NCCLWeightTransferUpdateInfo(
            names=["remote.weight"],
            dtype_names=["float32"],
            shapes=[[2]],
        )
    )

    assert group.broadcasts == 1
    assert session.received == []
