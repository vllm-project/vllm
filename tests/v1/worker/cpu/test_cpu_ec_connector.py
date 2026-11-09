# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.cpu import ec_connector as cpu_ec_connector
from vllm.v1.worker.cpu.ec_connector import (
    NO_OP_EC_CONNECTOR,
    ActiveECConnector,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

WORKER_META = object()
CONNECTOR_STATS = object()


class _ConnectorMetadata(ECConnectorMetadata):
    pass


class _RecordingECConnector(ECConnectorBase):
    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config, ECConnectorRole.WORKER)
        self.started_save_cache: dict[str, torch.Tensor] | None = None
        self.loaded_cache: dict[str, torch.Tensor] | None = None
        self.saved_hashes: list[str] = []
        self.finished_req_ids: set[str] | None = None
        self.metadata_cleared = False

    def start_save_caches(self, **kwargs: Any) -> None:
        self.started_save_cache = kwargs["encoder_cache"]

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        self.loaded_cache = encoder_cache

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        assert encoder_cache is self.started_save_cache
        self.saved_hashes.append(mm_hash)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        self.finished_req_ids = finished_req_ids
        return {"sent"}, {"received"}

    def build_connector_worker_meta(self) -> Any:
        return WORKER_META

    def get_ec_connector_stats(self) -> Any:
        return CONNECTOR_STATS

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self.metadata_cleared = True

    def has_cache_item(self, identifier: str) -> bool:
        return False

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        return _ConnectorMetadata()


def _config(ec_role: str) -> VllmConfig:
    config = VllmConfig()
    config.ec_transfer_config = ECTransferConfig(
        ec_connector="RecordingECConnector",
        ec_role=ec_role,
    )
    return config


def _scheduler_output() -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.ec_connector_metadata = _ConnectorMetadata()
    output.finished_req_ids = {"finished"}
    return output


def _connector(
    monkeypatch: pytest.MonkeyPatch,
    ec_role: str,
    encoder_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[ActiveECConnector, _RecordingECConnector]:
    config = _config(ec_role)
    recording_connector = _RecordingECConnector(config)
    monkeypatch.setattr(
        cpu_ec_connector, "get_ec_transfer", lambda: recording_connector
    )
    return (
        ActiveECConnector(config, encoder_cache if encoder_cache is not None else {}),
        recording_connector,
    )


@pytest.mark.parametrize(
    ("ec_role", "loads", "saves"),
    [
        ("ec_producer", False, True),
        ("ec_both", True, True),
        ("ec_consumer", True, False),
    ],
)
def test_transfers_encoder_caches_for_connector_role(
    monkeypatch: pytest.MonkeyPatch,
    ec_role: str,
    loads: bool,
    saves: bool,
) -> None:
    encoder_cache = {"old": torch.tensor([1.0])}
    connector, recording = _connector(monkeypatch, ec_role, encoder_cache)

    with connector.maybe_get_output(_scheduler_output()):
        encoder_cache["new"] = torch.tensor([2.0])

    assert (recording.loaded_cache is encoder_cache) is loads
    assert (recording.started_save_cache is encoder_cache) is saves
    assert recording.saved_hashes == (["new"] if saves else [])


def test_reports_connector_results_and_clears_step_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector, recording = _connector(monkeypatch, "ec_both")
    scheduler_output = _scheduler_output()

    with connector.maybe_get_output(scheduler_output) as output:
        assert output is not None
        assert output.ec_connector_worker_meta is None
        assert output.ec_connector_stats is None

    assert output.finished_sending == {"sent"}
    assert output.finished_recving == {"received"}
    assert output.ec_connector_worker_meta is WORKER_META
    assert output.ec_connector_stats is CONNECTOR_STATS
    assert recording.finished_req_ids == {"finished"}
    assert recording.metadata_cleared


def test_no_forward_completes_connector_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector, _ = _connector(monkeypatch, "ec_both")

    output = connector.no_forward(_scheduler_output())

    assert output.ec_connector_output is not None
    assert output.ec_connector_output.finished_sending == {"sent"}
    assert output.ec_connector_output.finished_recving == {"received"}
    assert output.ec_connector_output.ec_connector_worker_meta is WORKER_META
    assert NO_OP_EC_CONNECTOR.no_forward(_scheduler_output()) is (
        EMPTY_MODEL_RUNNER_OUTPUT
    )
