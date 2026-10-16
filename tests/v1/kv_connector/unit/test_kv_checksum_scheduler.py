# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums end to end through a producer's and a consumer's schedulers
with real checksum workers, for NIXL pull and a minimal connector carrying
them in kv_transfer_params: the consumer accepts a correct load whatever the
TP and PP sizes, and handles a corrupted or unverifiable one per
kv_checksum_fail_closed and kv_load_failure_policy."""

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

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("cuda_stream_stubs")]

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


KVConnectorFactory.register_connector(
    "KVTransferParamsCarrierConnector",
    __name__,
    KVTransferParamsCarrierConnector.__name__,
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
        heads_per_rank = HEADS // tp_size
        self.workers = []
        for pp_rank, layers in enumerate(stages):
            for tp_rank in range(tp_size):
                first = tp_rank * heads_per_rank
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
        reporting_workers: int | None = None,
    ):
        """Run the workers' post_forward and the scheduler's update."""
        checksums = None
        for worker in self.workers[:reporting_workers]:
            worker_output = worker.post_forward(
                finished_recving, set(), output.finished_req_ids
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


def _prefill(
    producer: _Engine, request: Request, reporting_workers: int | None = None
) -> tuple[dict[str, Any] | None, list[int]]:
    """Prefill ``request`` on the producer with seeded KV, and return its
    kv_transfer_params and its blocks, by position."""
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
    return engine_output.kv_transfer_params, block_ids


def _sent(params: dict[str, Any] | None) -> KVChecksumPayload | None:
    if params is None or KV_TRANSFER_PARAMS_KEY not in params:
        return None
    return KVChecksumPayload.from_bytes(
        base64.b64decode(params[KV_TRANSFER_PARAMS_KEY])
    )


def _copy_blocks(
    producer: _Engine,
    consumer: _Engine,
    producer_block_ids: list[int],
    start: int,
    block_ids: list[int],
) -> None:
    """The connector's transfer: the producer's tokens into the consumer's
    blocks at the same token positions."""
    p_size, c_size = producer.block_size, consumer.block_size
    for layer, kv in consumer.kv.items():
        for position, block_id in enumerate(block_ids, start):
            for token in range(position * c_size, (position + 1) * c_size):
                if token // p_size < len(producer_block_ids):
                    remote_block_id = producer_block_ids[token // p_size]
                    kv[block_id, :, token % c_size] = producer.kv[layer][
                        remote_block_id, :, token % p_size
                    ]


def _load(
    producer: _Engine,
    consumer: _Engine,
    request: Request,
    params: dict[str, Any] | None,
    producer_block_ids: list[int],
    corrupt: bool = False,
):
    """Start the load of ``request`` on the consumer, copy the producer's
    blocks into the consumer's, and report the load finished."""
    request.kv_transfer_params = {**(params or {}), "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    start, block_ids = _scheduled_blocks(output, request.request_id, send=False)
    _copy_blocks(producer, consumer, producer_block_ids, start, block_ids)
    if corrupt:
        consumer.kv[LAYERS[1]][block_ids[-1], 0, 0, 0] ^= 1
    return consumer.finish_step(output, {request.request_id})


CONNECTORS = pytest.mark.parametrize(
    "connector",
    ["KVTransferParamsCarrierConnector", "NixlConnector"],
    ids=["kv-transfer-params", "nixl-pull"],
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
def test_sent_checksums_do_not_depend_on_tp_and_pp(tp_size, pp_size):
    """Checksums and group fingerprints sum over TP shards and PP stages, so
    any producer layout sends what a single worker holding the whole model
    sends."""
    payloads = []
    for producer in (
        _Engine("kv_producer"),
        _Engine("kv_producer", tp_size=tp_size, pp_size=pp_size),
    ):
        request = create_request(1, NUM_TOKENS, do_remote_decode=True)
        payloads.append(_sent(_prefill(producer, request)[0]))
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
    params, _ = _prefill(
        _Engine("kv_producer", tp_size=2),
        create_request(1, NUM_TOKENS, do_remote_decode=True),
        reporting_workers=1,
    )
    assert params is not None and _sent(params) is None


@pytest.mark.parametrize(
    "producer_tp_pp,consumer_tp_pp",
    [((1, 1), (1, 1)), ((2, 1), (1, 1)), ((1, 1), (2, 1))]
    + [((1, 2), (1, 1)), ((1, 1), (2, 2))],
    ids=lambda tp_pp: f"tp{tp_pp[0]}pp{tp_pp[1]}",
)
@CONNECTORS
def test_correct_load_is_verified(connector, producer_tp_pp, consumer_tp_pp):
    """Checksums and group fingerprints sum over TP shards and PP stages, so
    any TP and PP pairing verifies."""
    (p_tp, p_pp), (c_tp, c_pp) = producer_tp_pp, consumer_tp_pp
    producer = _Engine("kv_producer", connector, tp_size=p_tp, pp_size=p_pp)
    consumer = _Engine(
        "kv_consumer", connector, tp_size=c_tp, pp_size=c_pp, fail_closed=True
    )
    params, block_ids = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    assert not _error_ids(_load(producer, consumer, request, params, block_ids))
    consumer.schedule()
    assert request.status == RequestStatus.RUNNING


def test_prefix_hit_verifies_only_loaded_blocks():
    """With a local prefix hit the consumer loads, and checks, only the
    blocks after it."""
    producer = _Engine("kv_producer")
    consumer = _Engine("kv_consumer", fail_closed=True)
    prefix = BLOCK_SIZE
    params, producer_block_ids = _prefill(
        producer, create_request(1, NUM_TOKENS, prefix, do_remote_decode=True)
    )

    local = create_request(2, NUM_TOKENS, prefix, max_tokens=1)
    consumer.scheduler.add_request(local)
    consumer.finish_step(consumer.schedule())
    request = create_request(1, NUM_TOKENS, prefix, do_remote_prefill=True)
    request.kv_transfer_params = {**(params or {}), "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    start, block_ids = _scheduled_blocks(output, request.request_id, send=False)
    assert (start, len(block_ids)) == (1, 2)
    _copy_blocks(producer, consumer, producer_block_ids, start, block_ids)
    assert not _error_ids(consumer.finish_step(output, {request.request_id}))


@pytest.mark.parametrize("corrupt", [False, True])
def test_different_block_sizes_are_unverified(corrupt):
    """Checksums of different block sizes are not compared: the load is
    unverified, corrupted or not."""
    producer = _Engine("kv_producer", block_size=32)
    consumer = _Engine("kv_consumer", fail_closed=True)
    params, block_ids = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )
    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    engine_outputs = _load(producer, consumer, request, params, block_ids, corrupt)
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
    params, block_ids = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )
    assert params is not None
    if fault == "no-checksums":
        params.pop(KV_TRANSFER_PARAMS_KEY)
    elif fault == "malformed":
        # Deeply nested msgpack, as a client could send in kv_transfer_params.
        malformed = b"\x94\x01\x01\x01" + b"\x91" * 100000 + b"\x01"
        params[KV_TRANSFER_PARAMS_KEY] = base64.b64encode(malformed).decode()

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    engine_outputs = _load(
        producer, consumer, request, params, block_ids, corrupt=fault == "corrupt"
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
    params, block_ids = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )
    sent = _sent(params)
    assert sent is not None
    assert set(sent.groups) == ({0} if producer_group == "skipped" else {0, 1})

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, params, block_ids))
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
