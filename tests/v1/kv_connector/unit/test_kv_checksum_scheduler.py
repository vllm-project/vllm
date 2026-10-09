# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums end to end through a producer's and a consumer's schedulers
with real checksum workers, for NIXL pull (kv_transfer_params carrier), NIXL
push and Mooncake (connector carriers): the consumer accepts a correct load whatever
the TP and PP sizes, and handles a corrupted or unverifiable one per
kv_checksum_fail_closed and kv_load_failure_policy."""

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum.scheduler import (
    KV_TRANSFER_PARAMS_KEY,
    KVChecksumScheduler,
)
from vllm.distributed.kv_transfer.kv_checksum.worker import KVChecksumWorker
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
        connector: str = "NixlConnector",
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
        self.connector = connector
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
    ):
        """Run the workers' post_forward and the scheduler's update.

        ``received``: checksums the connector received with the loads.
        """
        checksums = None
        for worker in self.workers:
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


def _block_ids(output: SchedulerOutput, req_id: str, send: bool) -> tuple[int, list]:
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


def _prefill(producer: _Engine, request: Request) -> _Transfer:
    """Prefill ``request`` on the producer, leaving the KV in its blocks."""
    if producer.connector == "MooncakeConnector":
        assert request.kv_transfer_params is not None
        request.kv_transfer_params["transfer_id"] = f"xfer-{request.request_id}"
    producer.scheduler.add_request(request)
    output = producer.schedule()
    _, block_ids = _block_ids(output, request.request_id, send=True)
    gen = torch.Generator().manual_seed(0)
    for kv in producer.kv.values():
        for block_id in block_ids:
            kv[block_id] = torch.randint(
                0, 256, kv[block_id].shape, generator=gen, dtype=torch.uint8
            )
    [engine_output] = producer.finish_step(output)[0].outputs
    assert engine_output.finish_reason == FinishReason.LENGTH
    # A connector carrier ships the checksums to the workers in the next step.
    meta = producer.schedule().kv_connector_metadata
    if producer.connector == "MooncakeConnector":
        # The router sends the request to both sides; the producer returns
        # no kv_transfer_params.
        assert engine_output.kv_transfer_params is None
        params = {
            "do_remote_decode": False,
            "transfer_id": f"xfer-{request.request_id}",
            "remote_engine_id": "producer",
            "remote_bootstrap_addr": "http://producer:8998",
        }
        sent = meta.kv_checksums.get(request.request_id)
    else:
        params = engine_output.kv_transfer_params
        assert params is not None
        sent = getattr(meta, "push_finished_kv_checksums", {}).get(request.request_id)
        assert (sent is not None) != (KV_TRANSFER_PARAMS_KEY in params)
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


def _load(
    producer: _Engine,
    consumer: _Engine,
    request: Request,
    transfer: _Transfer,
    corrupt: bool = False,
):
    """Start the load of ``request`` on the consumer, copy the producer's
    blocks into the consumer's, and report the load finished with the
    checksums the producer's connector sent, if any."""
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    start, block_ids = _block_ids(output, request.request_id, send=False)
    _copy_blocks(producer, consumer, transfer, start, block_ids)
    if corrupt:
        consumer.kv[LAYERS[1]][block_ids[-1], 0, 0, 0] ^= 1
    received = {request.request_id: transfer.sent} if transfer.sent else None
    return consumer.finish_step(output, {request.request_id}, received)


CONNECTORS = pytest.mark.parametrize(
    "connector",
    ["NixlConnector", "NixlPushConnector", "MooncakeConnector"],
    ids=["nixl-pull", "nixl-push", "mooncake"],
)


def _error_ids(engine_outputs) -> set[str]:
    return {
        o.request_id
        for outputs in engine_outputs.values()
        for o in outputs.outputs
        if o.finish_reason == FinishReason.ERROR
    }


@pytest.mark.parametrize(
    "producer_tp_pp,consumer_tp_pp",
    [((1, 1), (1, 1)), ((2, 1), (1, 1)), ((1, 1), (2, 1)), ((1, 1), (4, 1))]
    + [((1, 2), (1, 1)), ((1, 1), (2, 2))],
    ids=lambda tp_pp: f"tp{tp_pp[0]}pp{tp_pp[1]}",
)
@CONNECTORS
def test_correct_load_is_verified(connector, producer_tp_pp, consumer_tp_pp):
    """Checksums and group fingerprints sum over TP shards and PP stages, and a
    consumer with replicated heads (TP 4, 2 heads) checks every replica, so
    any TP and PP pairing verifies."""
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


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize(
    "producer_block_size,consumer_block_size", [(16, 32), (32, 16)]
)
@pytest.mark.parametrize(
    "connector", ["NixlConnector", "NixlPushConnector"], ids=["nixl-pull", "nixl-push"]
)
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


def test_prefix_hit_verifies_only_loaded_blocks():
    """With a local prefix hit the consumer loads, and checks, only the
    blocks after it."""
    producer = _Engine("kv_producer")
    consumer = _Engine("kv_consumer", fail_closed=True)
    prefix = BLOCK_SIZE
    transfer = _prefill(
        producer,
        create_request(1, NUM_TOKENS, prefix, do_remote_decode=True),
    )

    local = create_request(2, NUM_TOKENS, prefix, max_tokens=1)
    consumer.scheduler.add_request(local)
    consumer.finish_step(consumer.schedule())
    request = create_request(1, NUM_TOKENS, prefix, do_remote_prefill=True)
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    start, block_ids = _block_ids(output, request.request_id, send=False)
    assert (start, len(block_ids)) == (1, 2)
    _copy_blocks(producer, consumer, transfer, start, block_ids)
    engine_outputs = consumer.finish_step(output, {request.request_id})
    assert not _error_ids(engine_outputs)


def test_load_starting_inside_a_producer_block_is_unverified():
    """A consumer with smaller blocks whose local prefix hit ends inside a
    producer block cannot compare its first loaded block, so the load is
    unverified rather than verified without it."""
    producer = _Engine("kv_producer", "NixlPushConnector", block_size=32)
    consumer = _Engine(
        "kv_consumer", "NixlPushConnector", fail_closed=True, block_size=16
    )
    prefix = 16
    transfer = _prefill(
        producer, create_request(1, NUM_TOKENS, prefix, do_remote_decode=True)
    )

    local = create_request(2, NUM_TOKENS, prefix, max_tokens=1)
    consumer.scheduler.add_request(local)
    consumer.finish_step(consumer.schedule())
    request = create_request(1, NUM_TOKENS, prefix, do_remote_prefill=True)
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    start, block_ids = _block_ids(output, request.request_id, send=False)
    assert start == 1 and transfer.sent is not None
    _copy_blocks(producer, consumer, transfer, start, block_ids)
    engine_outputs = consumer.finish_step(
        output, {request.request_id}, {request.request_id: transfer.sent}
    )
    assert _error_ids(engine_outputs) == {request.request_id}


@pytest.mark.parametrize("fault", ["corrupt", "no-checksums"])
@pytest.mark.parametrize(
    "fail_closed,policy", [(False, "fail"), (True, "fail"), (True, "recompute")]
)
@CONNECTORS
def test_bad_load_follows_failure_policy(
    connector, fault, fail_closed, policy, caplog_vllm
):
    """A corrupted or unverifiable load is only logged when fail-open; when
    fail-closed it is a KV load failure: the request fails, or recomputes its
    prompt locally."""
    producer = _Engine("kv_producer", connector)
    consumer = _Engine("kv_consumer", connector, fail_closed=fail_closed, policy=policy)
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    if fault == "no-checksums":
        transfer.params.pop(KV_TRANSFER_PARAMS_KEY, None)
        transfer.sent = None

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
    verdict; a group only one side checksums leaves the load unverified."""
    producer = _Engine("kv_producer", extra_group=producer_group)
    consumer = _Engine("kv_consumer", fail_closed=True, extra_group="skipped")
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))

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
