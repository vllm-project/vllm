# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end integration tests against a fake-NIXL stack.

Exercises the real `ECCPUScheduler` + `ECCPUWorker` stack over a real ZMQ
router/dealer pair on TCP loopback, real msgspec wire codecs, and a real
mmap-backed `ECSharedRegion`. The one piece swapped out is `NixlWrapper`,
replaced by `FakeNixlWrapper` so the tests run on hosts without NIXL/UCX.

What these tests cover
-----------------------
- The full **worker round-trip**: an encoder tensor goes through the
  producer worker's `save_caches` (GPU→mmap), travels over the wire, and
  comes out of the consumer worker's `start_load_caches` (mmap→GPU).
- **Concurrency**: two simultaneous transfers for distinct mm_hashes must
  both complete with no leaked consumer blocks and no leaked producer pins.

CUDA stream/event bookkeeping in the workers is patched out so the tests
run on any host. The data motion that matters (`tensor.copy_`,
`view().reshape()`) is real CPU memory.
"""

import contextlib
import random
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from _fakes import FakeNixlWrapper, reset_fake_nixl_universe

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
    ECCPUScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker

# The worker's start_load_caches path calls the swap_blocks_batch compiled
# op; skip cleanly on hosts whose vllm extension predates that op instead of
# failing with an AttributeError deep in start_load_caches.
_requires_swap_blocks_batch = pytest.mark.skipif(
    not hasattr(torch.ops._C_cache_ops, "swap_blocks_batch"),
    reason=(
        "installed vllm C++ extension predates the swap_blocks_batch op "
        "used by ECCPUWorker.start_load_caches"
    ),
)

# ── shared layout ────────────────────────────────────────────────────────────

# block_size_bytes = hidden_dim * element_size (fp16=2) = 32. One row of
# encoder output fits in one block, which keeps the assertions readable.
_HIDDEN_DIM = 16
_DTYPE = torch.float16
_ELEMENT_SIZE = 2
_BLOCK_SIZE_BYTES = _HIDDEN_DIM * _ELEMENT_SIZE
_NUM_BLOCKS = 8


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_vllm_config(role: str) -> SimpleNamespace:
    """Minimal stand-in for `VllmConfig` shaped for `create_ec_shared_region`
    and the scheduler / worker constructors."""
    is_producer = role in ("ec_producer", "ec_both")
    is_consumer = role in ("ec_consumer", "ec_both")

    ec_transfer_config = SimpleNamespace(
        engine_id=str(uuid.uuid4()),
        is_ec_producer=is_producer,
        is_ec_consumer=is_consumer,
        ec_enable_nixl=True,
        ec_connector_extra_config={"ec_cpu_bytes": _NUM_BLOCKS * _BLOCK_SIZE_BYTES},
    )
    model_config = SimpleNamespace(
        dtype=_DTYPE,
        model="test-model",
        hf_config=None,
        get_inputs_embeds_size=lambda: _HIDDEN_DIM,
    )
    parallel_config = SimpleNamespace(rank=0, data_parallel_rank=0)
    return SimpleNamespace(
        instance_id=str(uuid.uuid4()),
        ec_transfer_config=ec_transfer_config,
        model_config=model_config,
        parallel_config=parallel_config,
        max_concurrent_batches=1,
    )


def _free_port() -> int:
    """Best-effort: pick a TCP port nobody else is on right now."""
    import socket

    for _ in range(50):
        port = random.randint(35000, 65000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("could not find a free TCP port")


def _build_info(producer: ECCPUScheduler, *, size_bytes: int) -> dict:
    """The dict an orchestrator would put in
    `request.ec_transfer_params[mm_hash]`. Carries only the side-channel
    address and size; the consumer learns the producer's NIXL metadata from
    the live XferAck."""
    return {
        "peer_host": producer._peer_host,
        "peer_port": producer._peer_port,
        "size_bytes": size_bytes,
    }


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    """Spin until predicate() is truthy or timeout. Used to wait on the
    producer router thread's asynchronous pin release."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _feature(mm_hash: str, length: int = 1) -> MagicMock:
    f = MagicMock()
    f.mm_hash = mm_hash
    f.identifier = mm_hash
    f.mm_position.length = length
    f.mm_position.offset = 0
    return f


def _request(features: list[MagicMock], *, params: dict | None = None) -> MagicMock:
    req = MagicMock()
    req.mm_features = features
    req.ec_transfer_params = params
    return req


def _drive_until(
    p: ECCPUScheduler,
    c: ECCPUScheduler,
    c_request,
    predicate,
    timeout_s: float = 10.0,
    on_meta=None,
):
    """Tick a full consumer scheduler step until ``predicate(p, c, meta)`` is
    truthy or we time out.

    Each iteration re-runs ``c.ensure_cache_available`` (which drives the
    consumer's ``_poll_step`` on the first call of the batch: XferAck drain,
    NIXL READ, completion detection) until it admits the request. Once
    admitted, mirroring the real scheduler loop, ``c.update_state_after_alloc``
    is called once per mm feature to pin the now-ready entry and queue it for
    load; the following ``c.build_connector_meta`` then promotes it into
    ``meta.loads``. The producer serves grants and notifs asynchronously on
    its own router thread. ``on_meta(meta)`` is invoked on every tick so the
    caller can accumulate state across multiple meta arrivals (used by the
    concurrent test, where the two acks may land on different ticks)."""
    deadline = time.monotonic() + timeout_s
    last_meta = None
    admitted = False
    while time.monotonic() < deadline:
        if not admitted:
            admitted = c.ensure_cache_available(c_request, num_computed_tokens=0)
            if admitted:
                for idx in range(len(c_request.mm_features)):
                    c.update_state_after_alloc(c_request, idx)
        p.build_connector_meta(SimpleNamespace(finished_req_ids=set()))
        last_meta = c.build_connector_meta(SimpleNamespace(finished_req_ids=set()))
        if on_meta is not None:
            on_meta(last_meta)
        if predicate(p, c, last_meta):
            return last_meta
        time.sleep(0.01)
    return None


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def stack(monkeypatch):
    """Build producer + consumer scheduler/worker pair against `FakeNixlWrapper`.

    Yields ``(p_sched, c_sched, p_worker, c_worker)``. The workers attach to
    the same mmap files as their respective schedulers (same engine_id).
    Tear-down: workers first (release mmap views), then schedulers (router
    thread join + NIXL state). The fake-NIXL universe is cleared on entry
    and exit so neither test sees stale agent registrations.
    """
    reset_fake_nixl_universe()

    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_PORT", str(_free_port()))

    fake_platform = MagicMock(device_type="cpu")
    p_cfg = _make_vllm_config("ec_producer")
    c_cfg = _make_vllm_config("ec_consumer")

    # The scheduler imports NixlWrapper / nixl_agent_config from
    # vllm.distributed.nixl_utils at call time (only for an availability
    # check), so we patch them at the source module. Workers do CPU<->GPU
    # copy; we patch CUDA away so byte motion runs on CPU and the tests
    # don't need a GPU.
    with (
        patch(
            "vllm.distributed.nixl_utils.NixlWrapper",
            FakeNixlWrapper,
        ),
        patch(
            "vllm.distributed.nixl_utils.nixl_agent_config",
            MagicMock(),
        ),
        # NixlEngine constructs its wrapper from the nixl_engine module's own
        # import, so the fake must be patched there too.
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl.NixlWrapper",
            FakeNixlWrapper,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl.nixl_agent_config",
            MagicMock(),
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.is_pin_memory_available",
            return_value=False,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.current_platform",
            fake_platform,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_pcp_group",
            return_value=SimpleNamespace(rank_in_group=0),
        ),
        patch("torch.cuda.Stream", return_value=MagicMock()),
        patch("torch.cuda.Event", return_value=MagicMock()),
        patch("torch.cuda.stream", return_value=contextlib.nullcontext()),
        patch("torch.cuda.current_stream", return_value=MagicMock()),
    ):
        p_sched = ECCPUScheduler(p_cfg)
        c_sched = ECCPUScheduler(c_cfg)
        p_worker = ECCPUWorker(p_cfg)
        c_worker = ECCPUWorker(c_cfg)

        try:
            yield p_sched, c_sched, p_worker, c_worker
        finally:
            for thing in (p_worker, c_worker, c_sched, p_sched):
                with contextlib.suppress(Exception):
                    thing.shutdown()
            reset_fake_nixl_universe()


# ── tests ────────────────────────────────────────────────────────────────────


@_requires_swap_blocks_batch
def test_end_to_end_single_xfer(stack):
    """Full producer → consumer round-trip exercising every layer:

      producer worker save_caches (encoder_cache → producer mmap)
        → next build_connector_meta promotes _pending_save → _local_encodings
        → consumer ensure_cache_available (XferReq on DEALER → ROUTER)
        → producer router thread: pin source, reply XferAck (grant + metadata)
        → consumer drain: register producer, NIXL READ (fake memmove
          producer→consumer mmap), poll to DONE → meta.loads
        → producer router thread: completion notif → unpin source
        → consumer worker start_load_caches (consumer mmap → encoder_cache)

    The encoder tensor must come out of the consumer worker bit-identical
    to what the producer worker put in.
    """
    p_sched, c_sched, p_worker, c_worker = stack

    mm_hash = "img_42"
    feature = _feature(mm_hash)
    p_request = _request([feature])

    # 1. Producer scheduler reserves CPU blocks for the new encoding.
    p_sched.update_state_after_alloc(p_request, index=0)
    p_meta_save = p_sched.build_connector_meta(SimpleNamespace(finished_req_ids=set()))
    assert mm_hash in p_meta_save.saves, (
        "producer scheduler must allocate blocks for newly-scheduled encoding"
    )

    # 2. Producer worker writes the encoding into its mmap.
    src = torch.arange(_HIDDEN_DIM, dtype=_DTYPE).reshape(1, _HIDDEN_DIM)
    encoder_cache_src: dict[str, torch.Tensor] = {mm_hash: src}
    p_worker.save_caches(encoder_cache_src, mm_hash, connector_metadata=p_meta_save)

    # 3. mark_ready is delayed by StepTracker until the GPU->mmap DMA is
    # guaranteed complete; one extra build_connector_meta tick advances the
    # tracker so the producer router thread can serve reads for mm_hash.
    p_sched.build_connector_meta(SimpleNamespace(finished_req_ids=set()))

    # 4. Consumer arrives with ec_transfer_params pointing at the producer.
    info = _build_info(p_sched, size_bytes=_HIDDEN_DIM * _ELEMENT_SIZE)
    c_request = _request([feature], params={mm_hash: info})
    accepted = c_sched.ensure_cache_available(c_request, num_computed_tokens=0)
    assert accepted is False, (
        "consumer must defer the request while waiting for the transfer"
    )
    assert mm_hash in c_sched._in_flight, "consumer must record the in-flight transfer"

    # 5. Tick both sides until the consumer's meta carries the load.
    final_meta = _drive_until(
        p_sched,
        c_sched,
        c_request,
        predicate=lambda p, c, m: m is not None and mm_hash in m.loads,
    )
    assert final_meta is not None, "transfer did not complete within timeout"

    # 6. Consumer worker reads the bytes back into its encoder_cache.
    encoder_cache_dst: dict[str, torch.Tensor] = {}
    c_worker.start_load_caches(encoder_cache_dst, connector_metadata=final_meta)

    out = encoder_cache_dst[mm_hash]
    assert out.shape == (1, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    assert torch.equal(out, src), (
        f"round-trip altered the bytes: src={src.tolist()} got={out.tolist()}"
    )

    # 7. Bookkeeping is clean on both sides.
    assert c_sched.has_cache_item(mm_hash), (
        "consumer must report cache hit for subsequent requests"
    )
    assert c_sched._in_flight == set(), "no remaining in-flight on consumer"
    # The producer's router thread releases the source pin on the READ
    # completion notif; that is asynchronous, so wait for it to drain.
    assert _wait_until(lambda: not p_sched._producer_session._active_xfers), (
        "producer source pin not released after read"
    )
    p_entry = p_sched._cache.get(mm_hash)
    assert p_entry is not None and p_entry.evictable, "producer leaked a block pin"


@_requires_swap_blocks_batch
def test_concurrent_xfers_different_mm_hashes(stack):
    """Two simultaneous transfers (distinct mm_hashes, single producer ↔
    single consumer) must both complete and leave clean state.

    Catches: leaked source pins when two completion notifs are drained in
    the same producer poll, leaked consumer blocks when two acks land in the
    same drain pass, and mm_hash crosstalk between the two reads.
    """
    p_sched, c_sched, p_worker, c_worker = stack

    f_a, f_b = _feature("img_a"), _feature("img_b")
    p_request = _request([f_a, f_b])

    # Producer reserves blocks for both encodings in one step.
    p_sched.update_state_after_alloc(p_request, index=0)
    p_sched.update_state_after_alloc(p_request, index=1)
    p_meta_save = p_sched.build_connector_meta(SimpleNamespace(finished_req_ids=set()))
    assert {"img_a", "img_b"} <= p_meta_save.saves.keys()

    # Producer worker writes both encodings; distinct value ranges so a
    # crosstalk bug would surface as a clear byte mismatch.
    src_a = torch.arange(_HIDDEN_DIM, dtype=_DTYPE).reshape(1, _HIDDEN_DIM)
    src_b = (torch.arange(_HIDDEN_DIM, dtype=_DTYPE) + 100).reshape(1, _HIDDEN_DIM)
    encoder_cache_src = {"img_a": src_a, "img_b": src_b}
    p_worker.save_caches(encoder_cache_src, "img_a", connector_metadata=p_meta_save)
    p_worker.save_caches(encoder_cache_src, "img_b", connector_metadata=p_meta_save)

    # Advance the ready-tracker one more step so both entries become ready
    # and the producer router thread can serve reads for either mm_hash.
    p_sched.build_connector_meta(SimpleNamespace(finished_req_ids=set()))

    # Consumer requests both at once.
    info = _build_info(p_sched, size_bytes=_HIDDEN_DIM * _ELEMENT_SIZE)
    c_request = _request([f_a, f_b], params={"img_a": info, "img_b": info})
    c_sched.ensure_cache_available(c_request, num_computed_tokens=0)
    assert {"img_a", "img_b"} <= c_sched._in_flight

    # The two acks may arrive on different ticks, so accumulate
    # encoder_cache_dst across iterations as `meta.loads` reports each.
    encoder_cache_dst: dict[str, torch.Tensor] = {}

    def _absorb(meta):
        if meta is not None and meta.loads:
            c_worker.start_load_caches(encoder_cache_dst, connector_metadata=meta)

    final_meta = _drive_until(
        p_sched,
        c_sched,
        c_request,
        predicate=lambda p, c, m: {"img_a", "img_b"} <= encoder_cache_dst.keys(),
        on_meta=_absorb,
    )
    assert final_meta is not None, "concurrent transfers did not complete"

    # Bytes preserved for both, no crosstalk.
    assert torch.equal(encoder_cache_dst["img_a"], src_a)
    assert torch.equal(encoder_cache_dst["img_b"], src_b)

    # Final state is clean: producer pins released, both arrivals cached.
    assert _wait_until(lambda: not p_sched._producer_session._active_xfers), (
        f"producer pins not released: {list(p_sched._producer_session._active_xfers)}"
    )
    for h in ("img_a", "img_b"):
        p_entry = p_sched._cache.get(h)
        assert p_entry is not None and p_entry.evictable, (
            f"producer leaked a block pin for {h}"
        )
    assert c_sched._in_flight == set(), (
        f"consumer in-flight not empty: {list(c_sched._in_flight)}"
    )
    assert c_sched.has_cache_item("img_a") and c_sched.has_cache_item("img_b"), (
        "both arrivals should be promoted to the consumer's local cache"
    )
