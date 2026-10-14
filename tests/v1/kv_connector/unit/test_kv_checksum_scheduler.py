# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums end to end through a producer's and a consumer's schedulers
with real checksum workers, for NIXL pull (kv_transfer_params carrier), NIXL
push (connector carrier) and minimal test connectors carrying them in
kv_transfer_params or in their own messages: the consumer accepts a correct
load whatever the TP and PP sizes, and handles a corrupted or unverifiable one
per kv_checksum_fail_closed and kv_load_failure_policy."""

from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum.checksum import KVChecksumCarrier
from vllm.distributed.kv_transfer.kv_checksum.scheduler import (
    KV_TRANSFER_PARAMS_KEY,
    KVChecksumPayload,
    KVChecksumScheduler,
)
from vllm.distributed.kv_transfer.kv_checksum.worker import KVChecksumWorker
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    MockKVConnector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_BLOCKS = 32
NUM_TOKENS = 40  # Three blocks, the last one partial.
HEADS = 2
CELL_BYTES = 32  # K + V of 8 bf16 values.
# The checksummed group's layers, split across PP stages.
LAYERS = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
# A second group, standing in for one a worker cannot checksum.
EXTRA_LAYER = "model.layers.2.self_attn.attn"


class KVTransferParamsCarrierConnector(KVConnectorBase_V1, SupportsHMA):
    """A pull-mode P/D connector that carries KV checksums in
    kv_transfer_params: a producer request (do_remote_decode) returns its
    params when it finishes, and a consumer request (do_remote_prefill)
    loads the rest of its prompt asynchronously. The test copies the KV."""

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params and params.get("do_remote_prefill"):
            return request.num_prompt_tokens - num_computed_tokens, True
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if request.kv_transfer_params:
            request.kv_transfer_params["do_remote_prefill"] = False

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return KVConnectorMetadata()

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_decode"):
            return False, None
        return False, {"do_remote_prefill": True}

    def kv_checksum_carrier(
        self, request: Request, receiving: bool
    ) -> KVChecksumCarrier | None:
        params = request.kv_transfer_params
        if not params or bool(params.get("do_remote_decode")) == receiving:
            return None
        return KVChecksumCarrier.KV_TRANSFER_PARAMS

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


class _ChecksumMetadata(KVConnectorMetadata):
    def __init__(self, kv_checksums: dict[str, bytes]):
        self.kv_checksums = kv_checksums


class ConnectorCarrierConnector(KVTransferParamsCarrierConnector):
    """Like KVTransferParamsCarrierConnector, but carries KV checksums in its
    own messages (KVChecksumCarrier.CONNECTOR): the producer's scheduler puts
    them in the next step's connector metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unsent: dict[str, bytes] = {}

    def kv_checksum_carrier(
        self, request: Request, receiving: bool
    ) -> KVChecksumCarrier | None:
        if super().kv_checksum_carrier(request, receiving) is None:
            return None
        return KVChecksumCarrier.CONNECTOR

    def send_kv_checksums(self, request: Request, checksums: bytes) -> None:
        self._unsent[request.request_id] = checksums

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = _ChecksumMetadata(self._unsent)
        self._unsent = {}
        return meta


for _connector in (KVTransferParamsCarrierConnector, ConnectorCarrierConnector):
    KVConnectorFactory.register_connector(
        _connector.__name__, __name__, _connector.__name__
    )


def _kv_cache_config(
    heads: int,
    layers: list[str],
    extra_group: bool = False,
    block_size: int = BLOCK_SIZE,
) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=8, dtype=torch.bfloat16
    )
    groups = [KVCacheGroupSpec(layers, spec)]
    if extra_group:
        groups.append(KVCacheGroupSpec([EXTRA_LAYER], spec))
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
    )


class _Engine:
    """A scheduler and its workers' checksum workers over one logical KV cache
    per layer, standing in for the model runner and the connector's
    transfers."""

    def __init__(
        self,
        kv_role: str,
        connector: str = "KVTransferParamsCarrierConnector",
        tp_size: int = 1,
        pp_size: int = 1,
        fail_closed: bool = False,
        policy: str = "fail",
        extra_group: str | None = None,
        block_size: int = BLOCK_SIZE,
    ):
        """``extra_group``: None, or whether the workers can checksum a second
        group ("covered") or skip it for lack of a cache ("skipped")."""
        vllm_config = create_vllm_config(
            kv_connector=connector,
            kv_role=kv_role,
            block_size=block_size,
            kv_load_failure_policy=policy,
        )
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_transfer_config.enable_kv_checksum = True
        kv_transfer_config.kv_checksum_fail_closed = fail_closed
        vllm_config.parallel_config.tensor_parallel_size = tp_size
        vllm_config.parallel_config.pipeline_parallel_size = pp_size
        stages = [LAYERS[i::pp_size] for i in range(pp_size)]
        has_extra = extra_group is not None
        # Like the engine's, the scheduler's config is the first worker's.
        self.scheduler = create_scheduler(
            vllm_config,
            kv_cache_config=_kv_cache_config(HEADS, stages[0], has_extra, block_size),
        )
        self.block_size = block_size
        # Per layer, [blocks, heads, tokens, cell] bytes; ranks view their heads.
        self.kv = {
            layer: torch.zeros(
                NUM_BLOCKS, HEADS, block_size, CELL_BYTES, dtype=torch.uint8
            )
            for layer in LAYERS
        }
        replicas = max(1, tp_size // HEADS)
        heads_per_rank = max(1, HEADS // tp_size)
        self.workers = []
        for pp_rank, layers in enumerate(stages):
            for tp_rank in range(tp_size):
                first = tp_rank // replicas * heads_per_rank
                caches = {
                    layer: self.kv[layer][:, first : first + heads_per_rank].view(
                        torch.bfloat16
                    )
                    for layer in layers
                }
                if extra_group == "covered":
                    caches[EXTRA_LAYER] = torch.zeros_like(caches[layers[0]])
                self.workers.append(
                    KVChecksumWorker(
                        _kv_cache_config(heads_per_rank, layers, has_extra, block_size),
                        caches,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        pp_rank=pp_rank,
                        uniform_kv_heads=HEADS,
                    )
                )

    def schedule(self) -> SchedulerOutput:
        output = self.scheduler.schedule()
        for worker in self.workers:
            worker.bind(output.kv_checksum_scheduled)
        return output

    def finish_step(
        self,
        output: SchedulerOutput,
        finished_recving: set[str] | None = None,
        received: dict[str, bytes] | None = None,
        reporting_workers: int | None = None,
    ):
        """Run the workers' post_forward and the scheduler's update.

        ``received``: checksums the connector received with the loads.
        """
        checksums = None
        for worker in self.workers[:reporting_workers]:
            worker_output = worker.post_forward(
                finished_recving, set(), output.finished_req_ids, received
            )
            if checksums is None:
                checksums = worker_output
            elif worker_output is not None:
                checksums = checksums.aggregate(worker_output)
        requests = [self.scheduler.requests[i] for i in output.num_scheduled_tokens]
        model_runner_output = create_model_runner_output(requests)
        model_runner_output.kv_connector_output = KVConnectorOutput(
            finished_recving=finished_recving, kv_checksums=checksums
        )
        return self.scheduler.update_from_output(output, model_runner_output)


def _scheduled_blocks(
    output: SchedulerOutput, req_id: str, send: bool
) -> tuple[int, list[int]]:
    scheduled = output.kv_checksum_scheduled
    assert scheduled is not None
    blocks = (scheduled.reqs_to_send if send else scheduled.reqs_to_recv)[req_id]
    return blocks.groups[0]


@dataclass
class _Transfer:
    """What a prefilled request hands its consumer."""

    params: dict[str, Any]
    """The consumer request's kv_transfer_params."""
    sent: bytes | None
    """Checksums the producer's connector sends itself, if it does."""
    block_ids: list[int]
    """The producer's blocks, by position."""

    @property
    def payload(self) -> KVChecksumPayload | None:
        """The checksums the producer sent, by either carrier."""
        data = self.sent
        if KV_TRANSFER_PARAMS_KEY in self.params:
            data = base64.b64decode(self.params[KV_TRANSFER_PARAMS_KEY])
        return None if data is None else KVChecksumPayload.from_bytes(data)


def _prefill(
    producer: _Engine, request: Request, reporting_workers: int | None = None
) -> _Transfer:
    """Prefill ``request`` on the producer with seeded KV, leaving the KV in
    its blocks."""
    producer.scheduler.add_request(request)
    output = producer.schedule()
    _, block_ids = _scheduled_blocks(output, request.request_id, send=True)
    gen = torch.Generator().manual_seed(0)
    for kv in producer.kv.values():
        for block_id in block_ids:
            kv[block_id] = torch.randint(
                0, 256, kv[block_id].shape, generator=gen, dtype=torch.uint8
            )
    [engine_output] = producer.finish_step(output, reporting_workers=reporting_workers)[
        0
    ].outputs
    assert engine_output.finish_reason == FinishReason.LENGTH
    params = engine_output.kv_transfer_params
    assert params is not None
    # A connector carrier ships the checksums in the next step.
    meta = producer.schedule().kv_connector_metadata
    sent = (
        getattr(meta, "kv_checksums", None)
        or getattr(meta, "push_finished_kv_checksums", {})
    ).get(request.request_id)
    return _Transfer(params, sent, block_ids)


def _copy_blocks(
    producer: _Engine,
    consumer: _Engine,
    transfer: _Transfer,
    start: int,
    block_ids: list[int],
) -> None:
    """The connector's transfer: the producer's tokens into the consumer's
    blocks at the same token positions."""
    p_size, c_size = producer.block_size, consumer.block_size
    for layer, kv in consumer.kv.items():
        for position, block_id in enumerate(block_ids, start):
            for token in range(position * c_size, (position + 1) * c_size):
                if token // p_size < len(transfer.block_ids):
                    remote_block_id = transfer.block_ids[token // p_size]
                    kv[block_id, :, token % c_size] = producer.kv[layer][
                        remote_block_id, :, token % p_size
                    ]


def _finish_load(consumer: _Engine, output, request: Request, transfer: _Transfer):
    """Report the load of ``request`` finished, with the checksums the
    producer's connector sent, if any."""
    received = {request.request_id: transfer.sent} if transfer.sent else None
    return consumer.finish_step(output, {request.request_id}, received)


def _load(
    producer: _Engine,
    consumer: _Engine,
    request: Request,
    transfer: _Transfer,
    corrupt: bool = False,
):
    """Start the load of ``request`` on the consumer, copy the producer's
    blocks into the consumer's, and report the load finished."""
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    start, block_ids = _scheduled_blocks(output, request.request_id, send=False)
    _copy_blocks(producer, consumer, transfer, start, block_ids)
    if corrupt:
        consumer.kv[LAYERS[1]][block_ids[-1], 0, 0, 0] ^= 1
    return _finish_load(consumer, output, request, transfer)


CONNECTORS = pytest.mark.parametrize(
    "connector",
    [
        "KVTransferParamsCarrierConnector",
        "ConnectorCarrierConnector",
        "NixlConnector",
        "NixlPushConnector",
    ],
    ids=["kv-transfer-params", "connector-carrier", "nixl-pull", "nixl-push"],
)


def _error_ids(engine_outputs) -> set[str]:
    return {
        o.request_id
        for outputs in engine_outputs.values()
        for o in outputs.outputs
        if o.finish_reason == FinishReason.ERROR
    }


@pytest.mark.parametrize(
    "tp_size,pp_size", [(2, 1), (1, 2), (2, 2)], ids=["tp2", "pp2", "tp2pp2"]
)
@CONNECTORS
def test_sent_checksums_do_not_depend_on_tp_and_pp(connector, tp_size, pp_size):
    """Checksums and group fingerprints sum over TP shards and PP stages, so
    any producer layout sends what a single worker holding the whole model
    sends, through either carrier."""
    payloads = []
    for producer in (
        _Engine("kv_producer", connector),
        _Engine("kv_producer", connector, tp_size=tp_size, pp_size=pp_size),
    ):
        request = create_request(1, NUM_TOKENS, do_remote_decode=True)
        transfer = _prefill(producer, request)
        # Exactly one carrier is used.
        assert (transfer.sent is None) == (KV_TRANSFER_PARAMS_KEY in transfer.params)
        payloads.append(transfer.payload)
    expected, sent = payloads
    assert expected is not None and sent is not None
    assert sent.num_tokens == expected.num_tokens == NUM_TOKENS
    assert sent.tp_size == tp_size
    [(fingerprint, block_size, start, values)] = sent.groups.values()
    [(fingerprint_1, _, _, values_1)] = expected.groups.values()
    assert (block_size, start, len(values)) == (BLOCK_SIZE, 0, 3)
    assert fingerprint == fingerprint_1
    assert (values == values_1).all() and values.any()


def test_only_requests_the_connector_carries_are_checksummed():
    """A request that is not a producer request of a P/D connector is not
    checksummed and gets no checksums."""
    producer = _Engine("kv_producer")
    request = create_request(1, NUM_TOKENS, max_tokens=1)
    producer.scheduler.add_request(request)
    output = producer.schedule()
    assert output.kv_checksum_scheduled is None
    [engine_output] = producer.finish_step(output)[0].outputs
    assert engine_output.kv_transfer_params is None


def test_no_checksums_sent_unless_every_worker_reported():
    """Without every worker's part the sum would be wrong, so none is sent."""
    transfer = _prefill(
        _Engine("kv_producer", tp_size=2),
        create_request(1, NUM_TOKENS, do_remote_decode=True),
        reporting_workers=1,
    )
    assert transfer.payload is None


@pytest.mark.parametrize(
    "producer_tp_pp,consumer_tp_pp",
    [((1, 1), (1, 1)), ((2, 1), (1, 1)), ((1, 1), (2, 1)), ((1, 1), (4, 1))]
    + [((1, 2), (1, 1)), ((1, 1), (2, 2)), ((4, 1), (1, 1))],
    ids=lambda tp_pp: f"tp{tp_pp[0]}pp{tp_pp[1]}",
)
@CONNECTORS
def test_correct_load_is_verified(connector, producer_tp_pp, consumer_tp_pp):
    """Checksums and group fingerprints sum over TP shards and PP stages, a
    producer with replicated heads (TP 4, 2 heads) sends each head once, and
    a consumer with replicated heads checks every replica, so any TP and PP
    pairing verifies."""
    (p_tp, p_pp), (c_tp, c_pp) = producer_tp_pp, consumer_tp_pp
    producer = _Engine("kv_producer", connector, tp_size=p_tp, pp_size=p_pp)
    consumer = _Engine(
        "kv_consumer", connector, tp_size=c_tp, pp_size=c_pp, fail_closed=True
    )
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    assert not _error_ids(_load(producer, consumer, request, transfer))
    consumer.schedule()
    assert request.status == RequestStatus.RUNNING


def _load_after_prefix_hit(
    producer: _Engine, consumer: _Engine, transfer: _Transfer, prefix: int
):
    """Cache ``prefix`` tokens of the prompt on the consumer, then load the
    rest; return the request, its first loaded position and the outputs."""
    local = create_request(2, NUM_TOKENS, prefix, max_tokens=1)
    consumer.scheduler.add_request(local)
    consumer.finish_step(consumer.schedule())
    request = create_request(1, NUM_TOKENS, prefix, do_remote_prefill=True)
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    start, block_ids = _scheduled_blocks(output, request.request_id, send=False)
    _copy_blocks(producer, consumer, transfer, start, block_ids)
    return (
        request,
        start,
        len(block_ids),
        _finish_load(consumer, output, request, transfer),
    )


def test_prefix_hit_verifies_only_loaded_blocks():
    """With a local prefix hit the consumer loads, and checks, only the
    blocks after it."""
    producer = _Engine("kv_producer")
    consumer = _Engine("kv_consumer", fail_closed=True)
    prefix = BLOCK_SIZE
    transfer = _prefill(
        producer, create_request(1, NUM_TOKENS, prefix, do_remote_decode=True)
    )
    _, start, num_blocks, engine_outputs = _load_after_prefix_hit(
        producer, consumer, transfer, prefix
    )
    assert (start, num_blocks) == (1, 2)
    assert not _error_ids(engine_outputs)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize(
    "producer_block_size,consumer_block_size", [(16, 32), (32, 16)]
)
@CONNECTORS
def test_different_block_sizes_are_compared(
    connector, producer_block_size, consumer_block_size, corrupt
):
    """With different block sizes the smaller blocks' checksums are summed
    and compared with the larger ones, including the last partial block."""
    producer = _Engine("kv_producer", connector, block_size=producer_block_size)
    consumer = _Engine(
        "kv_consumer", connector, fail_closed=True, block_size=consumer_block_size
    )
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    engine_outputs = _load(producer, consumer, request, transfer, corrupt)
    assert _error_ids(engine_outputs) == ({request.request_id} if corrupt else set())


def test_load_starting_inside_a_producer_block_is_unverified():
    """A consumer with smaller blocks whose local prefix hit ends inside a
    producer block cannot compare its first loaded block, so the load is
    unverified rather than verified without it."""
    producer = _Engine("kv_producer", block_size=32)
    consumer = _Engine("kv_consumer", fail_closed=True, block_size=16)
    prefix = 16
    transfer = _prefill(
        producer, create_request(1, NUM_TOKENS, prefix, do_remote_decode=True)
    )
    request, start, _, engine_outputs = _load_after_prefix_hit(
        producer, consumer, transfer, prefix
    )
    assert start == 1
    assert _error_ids(engine_outputs) == {request.request_id}


@pytest.mark.parametrize("fault", ["corrupt", "no-checksums", "malformed"])
@pytest.mark.parametrize(
    "fail_closed,policy", [(False, "fail"), (True, "fail"), (True, "recompute")]
)
@CONNECTORS
def test_bad_load_follows_failure_policy(
    connector, fault, fail_closed, policy, caplog_vllm
):
    """A corrupted load, or one with missing or malformed checksums, is only
    logged when fail-open; when fail-closed it is a KV load failure: the
    request fails, or recomputes its prompt locally."""
    producer = _Engine("kv_producer", connector)
    consumer = _Engine("kv_consumer", connector, fail_closed=fail_closed, policy=policy)
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    if fault == "no-checksums":
        transfer.params.pop(KV_TRANSFER_PARAMS_KEY, None)
        transfer.sent = None
    elif fault == "malformed":
        # Deeply nested msgpack, as a client could send in kv_transfer_params.
        malformed = b"\x94\x01\x01\x01" + b"\x91" * 100000 + b"\x01"
        if KV_TRANSFER_PARAMS_KEY in transfer.params:
            transfer.params[KV_TRANSFER_PARAMS_KEY] = base64.b64encode(
                malformed
            ).decode()
        else:
            transfer.sent = malformed

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    engine_outputs = _load(
        producer, consumer, request, transfer, corrupt=fault == "corrupt"
    )
    if fault == "corrupt":
        assert "KV checksum mismatch" in caplog_vllm.text
    if not fail_closed:
        assert not _error_ids(engine_outputs)
        output = consumer.schedule()
        assert output.num_scheduled_tokens[request.request_id] == 1
    elif policy == "fail":
        assert _error_ids(engine_outputs) == {request.request_id}
        assert request.status == RequestStatus.FINISHED_ERROR
    else:
        assert not _error_ids(engine_outputs)
        output = consumer.schedule()
        assert output.num_scheduled_tokens[request.request_id] == NUM_TOKENS


@pytest.mark.parametrize("producer_group", ["skipped", "covered"])
def test_groups_skipped_on_both_sides_are_ignored(producer_group):
    """A group neither side can checksum (e.g. SSM state) is left out of the
    producer's checksums and of the verdict; a group only one side
    checksums leaves the load unverified."""
    producer = _Engine("kv_producer", extra_group=producer_group)
    consumer = _Engine("kv_consumer", fail_closed=True, extra_group="skipped")
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    sent = transfer.payload
    assert sent is not None
    assert set(sent.groups) == ({0} if producer_group == "skipped" else {0, 1})

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, transfer))
    assert errors == (set() if producer_group == "skipped" else {request.request_id})


@pytest.mark.parametrize("fail_closed", [False, True])
def test_connector_without_carrier_is_refused(fail_closed):
    """With a connector that cannot carry checksums no load would be checked:
    fail-closed refuses to start, fail-open disables checksums."""
    vllm_config = create_vllm_config(kv_role="kv_consumer")
    vllm_config.kv_transfer_config.enable_kv_checksum = True
    vllm_config.kv_transfer_config.kv_checksum_fail_closed = fail_closed
    connector = object.__new__(MockKVConnector)
    kv_cache_config = _kv_cache_config(HEADS, LAYERS)
    if fail_closed:
        with pytest.raises(ValueError, match="does not carry KV checksums"):
            KVChecksumScheduler.create(vllm_config, kv_cache_config, connector)
    else:
        assert (
            KVChecksumScheduler.create(vllm_config, kv_cache_config, connector) is None
        )
