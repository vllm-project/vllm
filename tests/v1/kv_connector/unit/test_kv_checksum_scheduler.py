# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums end to end through a producer's and a consumer's schedulers
with real checksum workers, for NIXL pull (checksums in kv_transfer_params),
NIXL push and Mooncake (checksums in the connector's messages): the consumer
accepts a correct load whatever the TP and PP sizes, and handles a corrupted or
unverifiable one per kv_checksum_fail_closed and kv_load_failure_policy."""

from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum.scheduler import (
    KV_TRANSFER_PARAMS_KEY,
)
from vllm.distributed.kv_transfer.kv_checksum.worker import KVChecksumWorker
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MambaStateSplit,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
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
# A Mamba (GDN) layer: conv 4 x 8 bf16 and ssm 2 x 8 fp32 per TP rank, 128
# bytes in a 160-byte page. With prefix caching its states align to the
# attention blocks: a request's row ends with its running state, then one
# speculative scratch block.
MAMBA_LAYER = "model.layers.3.linear_attn"
MAMBA_SPEC = MambaSpec(
    block_size=BLOCK_SIZE,
    shapes=((4, 8), (2, 8)),
    dtypes=(torch.bfloat16, torch.float32),
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    mamba_cache_mode="align",
    num_speculative_blocks=1,
)
MAMBA_STATE_BYTES, MAMBA_PAGE_BYTES = 128, 160


def _split_mamba_spec(tp_size: int) -> MambaSpec:
    """A Mamba2 layer with one group, whose TP split is known: conv [x, B, C]
    of 4, 2 and 2 rows by 8 bf16 columns, B and C held whole by every rank,
    and a temporal state of 2 heads of 8 fp32, globally."""
    x, heads = 4 // tp_size, 2 // tp_size
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((x + 4, 8), (heads, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        mamba_cache_mode="align",
        num_speculative_blocks=1,
        state_splits=(
            MambaStateSplit(0, ((x, 1), (2, tp_size), (2, tp_size))),
            MambaStateSplit(0, ((heads, 1),)),
        ),
    )


def _split_shard(conv: torch.Tensor, temporal: torch.Tensor, tp_size, rank):
    """A rank's state bytes of the global ``_split_mamba_spec`` states."""
    x, heads = 4 // tp_size, 2 // tp_size
    local_conv = torch.cat([conv[rank * x : (rank + 1) * x], conv[4:]])
    local_temporal = temporal[rank * heads : (rank + 1) * heads]
    return torch.cat(
        [
            t.contiguous().view(torch.uint8).flatten()
            for t in (local_conv, local_temporal)
        ]
    )


def _kv_cache_config(
    heads: int,
    layers: list[str],
    extra_group: bool = False,
    mamba_spec: MambaSpec | None = None,
    block_size: int = BLOCK_SIZE,
) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=8, dtype=torch.bfloat16
    )
    groups = [KVCacheGroupSpec(layers, spec)]
    if extra_group:
        groups.append(KVCacheGroupSpec([EXTRA_LAYER], spec))
    if mamba_spec is not None:
        groups.append(KVCacheGroupSpec([MAMBA_LAYER], mamba_spec))
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
        mamba: bool | str = False,
        block_size: int = BLOCK_SIZE,
    ):
        """``extra_group``: None, or whether the workers can checksum a second
        group ("covered"), skip it for lack of a cache ("skipped"), or leave
        it out as it holds a drafter layer ("draft").
        ``mamba``: add a Mamba group, as hybrid models have; "split" for one
        whose TP split is known (``_split_mamba_spec``)."""
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
        if mamba:
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
            vllm_config.cache_config.mamba_cache_mode = "align"
        stages = [LAYERS[i::pp_size] for i in range(pp_size)]
        has_extra = extra_group is not None
        self.split_mamba = mamba == "split"
        mamba_spec = (
            (_split_mamba_spec(tp_size) if self.split_mamba else MAMBA_SPEC)
            if mamba
            else None
        )
        self.tp_size = tp_size
        # Like the engine's, the scheduler's config is the first worker's.
        self.connector = connector
        scheduler_config = _kv_cache_config(
            HEADS, stages[0], has_extra, mamba_spec, block_size
        )
        self.scheduler = create_scheduler(vllm_config, kv_cache_config=scheduler_config)
        self.mamba_group = len(scheduler_config.kv_cache_groups) - 1 if mamba else None
        self.block_size = block_size
        # Per layer, [blocks, heads, tokens, cell] bytes; ranks view their heads.
        self.kv = {
            layer: torch.zeros(
                NUM_BLOCKS, HEADS, block_size, CELL_BYTES, dtype=torch.uint8
            )
            for layer in LAYERS
        }
        # Per TP rank, the Mamba layer's state pages (its own shard).
        self.states = [
            torch.zeros(NUM_BLOCKS, 2 * MAMBA_PAGE_BYTES, dtype=torch.uint8)
            for _ in range(tp_size)
        ]
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
                if extra_group in ("covered", "draft"):
                    caches[EXTRA_LAYER] = torch.zeros_like(caches[layers[0]])
                if mamba_spec is not None:
                    # The model runner's [blocks, 1, 1, state] int8 page view.
                    content = mamba_spec.state_content_size_bytes
                    state = self.states[tp_rank][:, :content]
                    caches[MAMBA_LAYER] = state.view(torch.int8)[:, None, None, :]
                self.workers.append(
                    KVChecksumWorker(
                        _kv_cache_config(
                            heads_per_rank, layers, has_extra, mamba_spec, block_size
                        ),
                        caches,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        pp_rank=pp_rank,
                        uniform_kv_heads=HEADS,
                        draft_layer_names=frozenset(
                            [EXTRA_LAYER] if extra_group == "draft" else []
                        ),
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
        checksums = []
        for worker in self.workers:
            pending = worker.post_forward(
                KVConnectorTransferResults(
                    finished_recving=finished_recving or set(),
                    kv_checksums=received or {},
                ),
                output.finished_req_ids,
            )
            if pending is not None:
                checksums.append(pending.enqueue_async_cpu_copy())
        requests = [self.scheduler.requests[i] for i in output.num_scheduled_tokens]
        model_runner_output = create_model_runner_output(requests)
        # Like the model runner, sample nothing for a prefill not yet done.
        for i, request in enumerate(requests):
            if request.is_prefill_chunk:
                model_runner_output.sampled_token_ids[i] = []
        model_runner_output.kv_connector_output = KVConnectorOutput(
            finished_recving=finished_recving, kv_checksums=checksums or None
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
    state_block: int | None = None
    """The producer's Mamba state block, if any."""
    split_state: tuple[torch.Tensor, torch.Tensor] | None = None
    """The global states of a ``_split_mamba_spec`` layer, if any."""


def _state_block(
    engine: _Engine, output: SchedulerOutput, req_id: str, send: bool
) -> int | None:
    """The request's running-state block, the one the model writes: its Mamba
    row's last block before the speculative scratch blocks. Also checks that
    the workers checksum that block."""
    if engine.mamba_group is None:
        return None
    row = engine.scheduler.kv_cache_manager.get_blocks(req_id).blocks
    state_block = row[engine.mamba_group][-1 - MAMBA_SPEC.num_speculative_blocks]
    scheduled = output.kv_checksum_scheduled
    assert scheduled is not None
    blocks = (scheduled.reqs_to_send if send else scheduled.reqs_to_recv)[req_id]
    assert blocks.groups[engine.mamba_group] == (0, [state_block.block_id])
    return state_block.block_id


def _prefill(producer: _Engine, request: Request) -> _Transfer:
    """Prefill ``request`` on the producer, leaving the KV in its blocks."""
    if producer.connector == "MooncakeConnector":
        assert request.kv_transfer_params is not None
        request.kv_transfer_params["transfer_id"] = f"xfer-{request.request_id}"
    producer.scheduler.add_request(request)
    # Prefill may take several steps (Mamba states align chunks to blocks).
    output = producer.schedule()
    for _ in range(NUM_TOKENS):
        scheduled = output.kv_checksum_scheduled
        if scheduled is not None and request.request_id in scheduled.reqs_to_send:
            break
        producer.finish_step(output)
        output = producer.schedule()
    else:
        raise AssertionError("the prefill never finished")
    _, block_ids = _block_ids(output, request.request_id, send=True)
    gen = torch.Generator().manual_seed(0)
    for kv in producer.kv.values():
        for block_id in block_ids:
            kv[block_id] = torch.randint(
                0, 256, kv[block_id].shape, generator=gen, dtype=torch.uint8
            )
    state_block = _state_block(producer, output, request.request_id, send=True)
    split_state = None
    if state_block is not None and producer.split_mamba:
        conv = torch.randint(-(2**15), 2**15, (8, 8), generator=gen)
        temporal = torch.randint(-(2**31), 2**31, (2, 8), generator=gen)
        split_state = (
            conv.to(torch.int16).view(torch.bfloat16),
            temporal.to(torch.int32).view(torch.float32),
        )
        for rank, states in enumerate(producer.states):
            shard = _split_shard(*split_state, producer.tp_size, rank)
            states[state_block, : len(shard)] = shard
    elif state_block is not None:
        for states in producer.states:
            states[state_block] = torch.randint(
                0, 256, states[state_block].shape, generator=gen, dtype=torch.uint8
            )
    [engine_output] = producer.finish_step(output)[0].outputs
    assert engine_output.finish_reason == FinishReason.LENGTH
    # Connectors that send the checksums themselves get them in the next
    # step's metadata.
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
        assert KV_TRANSFER_PARAMS_KEY in params
        sent = getattr(meta, "push_finished_kv_checksums", {}).get(request.request_id)
        assert (sent is not None) == (producer.connector == "NixlPushConnector")
        if sent is not None:
            # In push mode the proxy sends the consumer's request without the
            # producer's response, so the checksums reach it with the KV.
            params = {k: v for k, v in params.items() if k != KV_TRANSFER_PARAMS_KEY}
    return _Transfer(params, sent, block_ids, state_block, split_state)


def _copy_blocks(
    producer: _Engine,
    consumer: _Engine,
    transfer: _Transfer,
    start: int,
    block_ids: list[int],
    state_block: int | None = None,
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
    if state_block is not None and transfer.split_state is not None:
        for rank, states in enumerate(consumer.states):
            shard = _split_shard(*transfer.split_state, consumer.tp_size, rank)
            states[state_block, : len(shard)] = shard
    elif (
        state_block is not None
        and transfer.state_block is not None
        and len(producer.states) == len(consumer.states)
    ):
        for p_states, c_states in zip(producer.states, consumer.states):
            c_states[state_block] = p_states[transfer.state_block]


def _load(
    producer: _Engine,
    consumer: _Engine,
    request: Request,
    transfer: _Transfer,
    corrupt: str | None = None,
):
    """Start the load of ``request`` on the consumer, copy the producer's
    blocks into the consumer's, and report the load finished with the
    checksums the producer's connector sent, if any."""
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    start, block_ids = _block_ids(output, request.request_id, send=False)
    state_block = _state_block(consumer, output, request.request_id, send=False)
    _copy_blocks(producer, consumer, transfer, start, block_ids, state_block)
    if corrupt == "kv":
        consumer.kv[LAYERS[1]][block_ids[-1], 0, 0, 0] ^= 1
    elif corrupt == "state":
        assert state_block is not None
        consumer.states[0][state_block, MAMBA_STATE_BYTES - 1] ^= 1
    elif corrupt == "last-replica":
        # The last rank's copy of B, after its x rows of 16 bytes each.
        assert state_block is not None
        x_bytes = 4 // consumer.tp_size * 16
        consumer.states[-1][state_block, x_bytes] ^= 1
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
    engine_outputs = _load(
        producer, consumer, request, transfer, "kv" if corrupt else None
    )
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
        producer,
        consumer,
        request,
        transfer,
        corrupt="kv" if fault == "corrupt" else None,
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


@pytest.mark.parametrize("producer_group", ["draft", "covered"])
def test_drafter_groups_are_ignored(producer_group):
    """A group of drafter layers only, which no worker checksums, is left out
    of the verdict, unless the producer checksums it."""
    producer = _Engine("kv_producer", extra_group=producer_group)
    consumer = _Engine("kv_consumer", fail_closed=True, extra_group="draft")
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, transfer))
    assert errors == (set() if producer_group == "draft" else {request.request_id})


def test_load_from_a_cache_tier_is_not_checksummed(monkeypatch):
    """A consumer request whose KV is loaded from a cache tier (e.g. by an
    offloading connector in a MultiConnector) rather than from its producer is
    not checksummed: the tier's KV need not match the producer's bit for bit."""
    producer = _Engine("kv_producer")
    consumer = _Engine("kv_consumer", fail_closed=True)
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    monkeypatch.setattr(
        consumer.scheduler.connector,
        "get_external_cache_hit_sources",
        lambda request, num_tokens: {CacheHitSource.HOST: num_tokens},
    )

    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    request.kv_transfer_params = {**transfer.params, "do_remote_prefill": True}
    consumer.scheduler.add_request(request)
    output = consumer.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert output.kv_checksum_scheduled is None


def test_requests_without_pd_params_are_not_checksummed():
    """Only requests with do_remote_decode (producers) or do_remote_prefill
    (consumers) in their kv_transfer_params are checksummed."""
    producer = _Engine("kv_producer")
    request = create_request(1, NUM_TOKENS)
    producer.scheduler.add_request(request)
    for _ in range(NUM_TOKENS):
        output = producer.schedule()
        assert output.kv_checksum_scheduled is None
        if request.num_computed_tokens >= request.num_prompt_tokens:
            break
        producer.finish_step(output)
    else:
        raise AssertionError("the prefill never finished")


@CONNECTORS
@pytest.mark.parametrize("corrupt", [None, "state"])
def test_hybrid_model_verifies_mamba_state(connector, corrupt):
    """With a Mamba group, the request's running state is checksummed too: a
    correct load verifies, a flipped state bit is a mismatch."""
    producer = _Engine("kv_producer", connector, mamba=True)
    consumer = _Engine("kv_consumer", connector, fail_closed=True, mamba=True)
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, transfer, corrupt))
    assert errors == (set() if corrupt is None else {request.request_id})


@CONNECTORS
@pytest.mark.parametrize("corrupt", [None, "last-replica"])
@pytest.mark.parametrize("producer_tp, consumer_tp", [(1, 2), (2, 1)])
def test_split_mamba_state_compared_across_tp_sizes(
    connector, producer_tp, consumer_tp, corrupt
):
    """With its TP split known, Mamba state is compared between producers and
    consumers of different TP sizes: a correct load verifies, and a corrupted
    copy of a replicated sub-projection is a mismatch."""
    producer = _Engine("kv_producer", connector, tp_size=producer_tp, mamba="split")
    consumer = _Engine(
        "kv_consumer", connector, tp_size=consumer_tp, fail_closed=True, mamba="split"
    )
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    errors = _error_ids(_load(producer, consumer, request, transfer, corrupt))
    assert errors == (set() if corrupt is None else {request.request_id})


def test_mamba_state_not_compared_across_tp_sizes():
    """TP-sharded Mamba state is laid out per TP size, so between TP 1 and
    TP 2 it is left out like a group neither side checksums: the attention
    groups still verify the load, even fail-closed."""
    producer = _Engine("kv_producer", mamba=True)
    consumer = _Engine("kv_consumer", tp_size=2, fail_closed=True, mamba=True)
    transfer = _prefill(producer, create_request(1, NUM_TOKENS, do_remote_decode=True))
    request = create_request(1, NUM_TOKENS, do_remote_prefill=True)
    assert not _error_ids(_load(producer, consumer, request, transfer))
