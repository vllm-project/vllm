# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums end to end through a producer's and a consumer's schedulers
with real checksum workers, for NIXL pull (kv_transfer_params carrier) and
push (completion notif carrier): the consumer accepts a correct load whatever
the TP and PP sizes, and handles a corrupted or unverifiable one per
kv_checksum_fail_closed and kv_load_failure_policy."""

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
    heads: int, layers: list[str], extra_group: bool = False
) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=heads, head_size=8, dtype=torch.bfloat16
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
    ):
        """``extra_group``: None, or whether the workers can checksum a second
        group ("covered") or skip it for lack of a cache ("skipped")."""
        vllm_config = create_vllm_config(
            kv_connector=connector,
            kv_role=kv_role,
            block_size=BLOCK_SIZE,
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
            vllm_config, kv_cache_config=_kv_cache_config(HEADS, stages[0], has_extra)
        )
        # Per layer, [blocks, heads, tokens, cell] bytes; ranks view their heads.
        self.kv = {
            layer: torch.zeros(
                NUM_BLOCKS, HEADS, BLOCK_SIZE, CELL_BYTES, dtype=torch.uint8
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
                        _kv_cache_config(heads_per_rank, layers, has_extra),
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


def _prefill(producer: _Engine, request: Request) -> tuple[dict[str, Any], Any]:
    """Prefill ``request`` on the producer, leaving the KV in its blocks.

    Returns:
        Its kv_transfer_params, and the checksums the connector sends to the
        consumer itself (push), if any.

    """
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
    params = engine_output.kv_transfer_params
    assert params is not None
    # Push: the next step ships the checksums to the workers with the blocks.
    meta = producer.schedule().kv_connector_metadata
    sent = getattr(meta, "push_finished_kv_checksums", {}).get(request.request_id)
    assert (sent is not None) != (KV_TRANSFER_PARAMS_KEY in params)
    return params, sent


def _copy_blocks(
    producer: _Engine,
    consumer: _Engine,
    params: dict[str, Any],
    start: int,
    block_ids: list[int],
) -> None:
    """The connector's transfer: the producer's blocks at the consumer's
    block positions."""
    remote_block_ids = params["remote_block_ids"][0]
    for layer, kv in consumer.kv.items():
        for position, block_id in enumerate(block_ids, start):
            kv[block_id] = producer.kv[layer][remote_block_ids[position]]


def _load(
    producer: _Engine,
    consumer: _Engine,
    request: Request,
    params: dict[str, Any],
    sent: bytes | None = None,
    corrupt: bool = False,
):
    """Start the load of ``request`` on the consumer, copy the producer's
    blocks into the consumer's, and report the load finished with the
    checksums the producer's connector ``sent``, if any."""
    request.kv_transfer_params = {**params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    start, block_ids = _block_ids(output, request.request_id, send=False)
    _copy_blocks(producer, consumer, params, start, block_ids)
    if corrupt:
        consumer.kv[LAYERS[1]][block_ids[-1], 0, 0, 0] ^= 1
    received = {request.request_id: sent} if sent is not None else None
    return consumer.finish_step(output, {request.request_id}, received)


CONNECTORS = pytest.mark.parametrize(
    "connector", ["NixlConnector", "NixlPushConnector"], ids=["pull", "push"]
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
    params, sent = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    assert not _error_ids(_load(producer, consumer, request, params, sent))
    consumer.schedule()
    assert request.status == RequestStatus.RUNNING


def test_prefix_hit_verifies_only_loaded_blocks():
    """With a local prefix hit the consumer loads, and checks, only the
    blocks after it."""
    producer = _Engine("kv_producer")
    consumer = _Engine("kv_consumer", fail_closed=True)
    prefix = BLOCK_SIZE
    params, _ = _prefill(
        producer,
        create_request(1, NUM_TOKENS, prefix, do_remote_decode=True),
    )

    local = create_request(2, NUM_TOKENS, prefix, max_tokens=1)
    consumer.scheduler.add_request(local)
    consumer.finish_step(consumer.schedule())
    request = create_request(1, NUM_TOKENS, prefix, do_remote_prefill=True)
    request.kv_transfer_params = {**params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    start, block_ids = _block_ids(output, request.request_id, send=False)
    assert (start, len(block_ids)) == (1, 2)
    _copy_blocks(producer, consumer, params, start, block_ids)
    engine_outputs = consumer.finish_step(output, {request.request_id})
    assert not _error_ids(engine_outputs)


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
    params, sent = _prefill(
        producer, create_request(1, NUM_TOKENS, do_remote_decode=True)
    )
    if fault == "no-checksums":
        params.pop(KV_TRANSFER_PARAMS_KEY, None)
        sent = None

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    engine_outputs = _load(
        producer, consumer, request, params, sent, corrupt=fault == "corrupt"
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
    params, _ = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, params))
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
