# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 GPU model runner's EC connector wrapper."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorWorkerMetadata,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu.ec_connector import (
    NO_OP_EC_CONNECTOR,
    ActiveECConnector,
    get_ec_connector,
)

pytestmark = pytest.mark.cpu_test


class FakeMetadata(ECConnectorMetadata):
    pass


class FakeWorkerMeta(ECConnectorWorkerMetadata):
    """Mirrors ECCPUWorkerMetadata: the transfers that completed this step."""

    def __init__(
        self,
        completed_saves: list[str] | None = None,
        completed_loads: list[str] | None = None,
    ):
        self.completed_saves = completed_saves or []
        self.completed_loads = completed_loads or []

    def aggregate(self, other: "FakeWorkerMeta") -> "FakeWorkerMeta":
        return self


class FakeECConnector(ECConnectorBase):
    """Records the calls that ActiveECConnector is expected to drive."""

    def __init__(self, is_producer: bool = True, is_consumer: bool = False):
        # ECConnectorBase.__init__ requires a full VllmConfig; set only what
        # the worker-side code reads.
        self._connector_metadata = None
        self._is_producer = is_producer
        self._is_consumer = is_consumer
        self.bound_metadata: list[ECConnectorMetadata] = []
        self.clear_calls = 0
        self.load_calls: list[dict] = []
        self.saved_hashes: list[str] = []
        self.worker_meta_calls = 0
        self.worker_meta: FakeWorkerMeta | None = None
        self.finished: tuple[set[str] | None, set[str] | None] = (None, None)

    def bind_connector_metadata(self, connector_metadata: ECConnectorMetadata) -> None:
        self.bound_metadata.append(connector_metadata)
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self.clear_calls += 1
        super().clear_connector_metadata()

    def start_load_caches(self, encoder_cache: dict, **kwargs) -> None:
        self.load_calls.append(encoder_cache)

    def save_caches(self, encoder_cache: dict, mm_hash: str) -> None:
        self.saved_hashes.append(mm_hash)

    def get_finished(self, finished_req_ids: set[str]):
        return self.finished

    def build_connector_worker_meta(self) -> FakeWorkerMeta | None:
        self.worker_meta_calls += 1
        return self.worker_meta

    # Scheduler-side abstract methods, unused by these tests.
    def has_cache_item(self, identifier: str) -> bool:
        return False

    def update_state_after_alloc(self, request, index: int) -> None:
        pass

    def build_connector_meta(self, scheduler_output) -> ECConnectorMetadata:
        return FakeMetadata()


def _scheduler_output(metadata: ECConnectorMetadata | None) -> SimpleNamespace:
    return SimpleNamespace(ec_connector_metadata=metadata, finished_req_ids=frozenset())


def _active_connector(
    encoder_cache: dict | None = None, **connector_kwargs
) -> tuple[ActiveECConnector, FakeECConnector]:
    fake = FakeECConnector(**connector_kwargs)
    with patch("vllm.v1.worker.gpu.ec_connector.get_ec_transfer", return_value=fake):
        connector = ActiveECConnector(SimpleNamespace(), encoder_cache or {})
    return connector, fake


def test_no_forward_is_noop_without_ec_connector():
    """`ec_connector_metadata is None` means no EC connector is configured:
    nothing to poll, and the shared empty output is returned as-is.
    """
    scheduler_output = _scheduler_output(metadata=None)

    assert NO_OP_EC_CONNECTOR.no_forward(scheduler_output) is EMPTY_MODEL_RUNNER_OUTPUT

    connector, fake = _active_connector()
    assert connector.no_forward(scheduler_output) is EMPTY_MODEL_RUNNER_OUTPUT
    assert fake.bound_metadata == []
    assert fake.worker_meta_calls == 0


def test_no_forward_polls_connector_with_empty_metadata():
    """A step with no work still reaps completed transfers.

    The scheduler sends metadata every step, empty or not, and that is what
    drives build_connector_worker_meta() -- the only channel by which finished
    saves and loads are reported.
    """
    connector, fake = _active_connector()
    fake.worker_meta = FakeWorkerMeta(completed_saves=["mm0"])
    scheduler_output = _scheduler_output(metadata=FakeMetadata())

    output = connector.no_forward(scheduler_output)

    assert fake.worker_meta_calls == 1
    assert output.ec_connector_output.ec_connector_worker_meta is fake.worker_meta
    assert fake.bound_metadata == [scheduler_output.ec_connector_metadata]
    assert fake.clear_calls == 1


@pytest.mark.parametrize(
    ("is_producer", "is_consumer"),
    [(True, False), (True, True), (False, True)],
)
def test_maybe_get_output_saves_only_newly_added_caches(is_producer, is_consumer):
    """Every producer offloads caches computed during the step, including an
    ec_both node, and never re-offloads ones that were already cached.
    """
    encoder_cache = {"mm_old": None}
    connector, fake = _active_connector(
        encoder_cache, is_producer=is_producer, is_consumer=is_consumer
    )

    with connector.maybe_get_output(_scheduler_output(FakeMetadata())):
        encoder_cache["mm_new"] = None

    assert fake.saved_hashes == (["mm_new"] if is_producer else [])
    assert fake.load_calls == ([encoder_cache] if is_consumer else [])


def test_worker_meta_is_populated_only_on_exit():
    """The worker's report lands in the finally block, so a caller returning
    from inside the `with` would drop it.
    """
    connector, fake = _active_connector()
    fake.worker_meta = FakeWorkerMeta(completed_loads=["mm0"])

    with connector.maybe_get_output(_scheduler_output(FakeMetadata())) as output:
        assert output.ec_connector_worker_meta is None
        assert fake.worker_meta_calls == 0

    assert output.ec_connector_worker_meta is fake.worker_meta


def test_get_ec_connector_activates_only_for_a_multimodal_ec_deployment():
    encoder_cache = SimpleNamespace(encoder_outputs={})
    config = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
    encoder_decoder = SimpleNamespace(
        model_config=SimpleNamespace(is_encoder_decoder=True)
    )

    with patch(
        "vllm.v1.worker.gpu.ec_connector.get_ec_transfer",
        return_value=FakeECConnector(),
    ):
        with patch(
            "vllm.v1.worker.gpu.ec_connector.has_ec_transfer", return_value=False
        ):
            assert get_ec_connector(config, encoder_cache) is NO_OP_EC_CONNECTOR

        with patch(
            "vllm.v1.worker.gpu.ec_connector.has_ec_transfer", return_value=True
        ):
            assert get_ec_connector(encoder_decoder, encoder_cache) is (
                NO_OP_EC_CONNECTOR
            )
            assert get_ec_connector(config, None) is NO_OP_EC_CONNECTOR
            connector = get_ec_connector(config, encoder_cache)

    assert isinstance(connector, ActiveECConnector)
    # Aliases the live dict, so caches added later in the step are visible.
    assert connector.encoder_cache is encoder_cache.encoder_outputs
