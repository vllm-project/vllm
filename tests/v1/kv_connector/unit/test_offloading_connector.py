# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import msgspec
import msgspec.msgpack
import prometheus_client
import pytest
import torch
import zmq
from tqdm import tqdm

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform

# The offloaded block size must stay a whole number of GPU blocks.
CPU_BLOCK_SIZES: int
if current_platform.is_xpu():
    CPU_BLOCK_SIZES = 64
elif current_platform.is_rocm():
    # ROCM_AITER_UNIFIED_ATTN prefers 64-token GPU blocks, so 192 covers both that
    # and the 16-token default.
    CPU_BLOCK_SIZES = 192
else:
    CPU_BLOCK_SIZES = 48

_ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    _ATTN_BACKENDS = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]
elif current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported

    _ATTN_BACKENDS = ["TRITON_ATTN"]
    if is_aiter_found_and_supported():
        _ATTN_BACKENDS.append("ROCM_AITER_UNIFIED_ATTN")
elif current_platform.is_xpu():
    _ATTN_BACKENDS = ["FLASH_ATTN", "TRITON_ATTN"]

# (model, attn_backend | None, block_size | None, uses_hma)
#
# - Llama: tested with each attention backend and a custom block_size.
# - Gemma-3: HMA (sliding window + full attention), default backend.
# - Mamba-130m: HMA (attention-free, acts like sliding_window=1),
#   default backend.  Prefix caching must be force-enabled.
# - Falcon-H1-0.5B-Instruct: HMA (parallel SSM/attention in every layer).
#   After page-size unification the mamba and attention groups have
#   different block sizes.
MODEL_PARAMS: list[tuple[str, str | None, int | None, bool]] = [
    ("meta-llama/Llama-3.2-1B-Instruct", backend, CPU_BLOCK_SIZES, False)
    for backend in _ATTN_BACKENDS
]
# HMA / Mamba models are only tested on CUDA (not ROCm).
if current_platform.is_cuda():
    MODEL_PARAMS += [
        ("google/gemma-3-1b-it", None, CPU_BLOCK_SIZES, True),
        ("state-spaces/mamba-130m-hf", None, CPU_BLOCK_SIZES, True),
        # Falcon-H1: parallel hybrid (every layer has both attention and SSM).
        # The mamba and attention groups end up with different GPU block sizes
        # after page-size unification, so we leave cpu_block_size=None
        # (blocks_per_chunk stays 1).
        ("tiiuae/Falcon-H1-0.5B-Instruct", None, None, True),
    ]

# Maximum time (seconds) to wait for the async CPU offload transfer
# to complete before giving up.
_RESET_CACHE_TIMEOUT = 30 if current_platform.is_rocm() else 10

# ZMQ poll timeout (ms) for the first event.
_FIRST_EVENT_POLL_MS = 10_000 if current_platform.is_rocm() else 1000

# Hard ceiling (seconds) on how long get_new_cpu_stored_events may loop,
# to prevent hangs if non-CPU events keep arriving indefinitely.
_EVENT_DRAIN_TIMEOUT = 60


def _assert_gdn_native_state(actual, shards, tp, rank):
    """Compare native channels/heads, independently of canonical copy offsets."""
    assert 0 <= rank < tp
    assert shards and all(set(shard) == set(actual) for shard in shards)
    checked_bytes = 0
    for name, value in actual.items():
        parts = [shard[name] for shard in shards]
        assert all(part.dtype == value.dtype for part in parts), name
        full = torch.cat(parts, dim=0)
        assert full.shape[0] % tp == 0, name
        expected = full.chunk(tp, dim=0)[rank]
        assert value.shape == expected.shape, name
        left = value.contiguous().view(torch.uint8)
        right = expected.contiguous().view(torch.uint8)
        assert torch.equal(left, right), name
        checked_bytes += left.numel()
    return checked_bytes


class _GDNStateOracle:
    """Test-only native pre-store snapshots and post-completion byte checks."""

    vllm_config: VllmConfig

    def gdn_oracle_records(self):
        return self._gdn_records

    def init_worker_extension(self):
        from vllm.config import get_layers_from_vllm_config
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
            OffloadingConnectorWorker,
        )
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

        directory = Path(os.environ["VLLM_TEST_GDN_ORACLE_DIR"])
        source_tp = int(os.environ["VLLM_TEST_GDN_SOURCE_TP"])
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        rank = self.vllm_config.parallel_config.rank
        self._gdn_records: list[dict] = []
        modules, pending = {}, {}
        groups: list[list[str]] = []
        original_init = OffloadingConnectorWorker._init_worker
        original_submit = SingleDirectionOffloadingHandler.transfer_async
        original_finish = SingleDirectionOffloadingHandler.get_finished

        def initialize(connector, caches):
            original_init(connector, caches)
            assert connector.spec.config.recurrent_schema.format == "gdn-ds-v1"
            modules.update(get_layers_from_vllm_config(self.vllm_config, MambaBase))
            modules.update(get_layers_from_vllm_config(self.vllm_config, Attention))
            groups.extend(
                g.layer_names for g in connector.kv_cache_config.kv_cache_groups
            )

        def snapshot(group, block):
            values = {}
            for name in groups[group]:
                layer = modules[name]
                if isinstance(layer, MambaBase):
                    conv, temporal = layer.kv_cache
                    q, v = layer.key_dim // tp, layer.value_dim // tp
                    assert conv.shape[1:] == (2 * q + v, 3)
                    assert temporal.shape[1:] == (
                        layer.num_v_heads // tp,
                        layer.head_v_dim,
                        layer.head_k_dim,
                    )
                    pieces = dict(zip(("Q", "K", "V"), conv[block].split((q, q, v))))
                    pieces["S"] = temporal[block]
                else:
                    cache = layer.kv_cache
                    assert cache.ndim == 4 and cache.shape[2] == 1024
                    key, value = cache[block].chunk(2, dim=-1)
                    pieces = {"AK": key, "AV": value}
                # Only diagnostic host reads are exempt, never production calls.
                with gpu_sync_allowed():
                    values.update(
                        {
                            f"{name}::{k}": v.detach().cpu().contiguous()
                            for k, v in pieces.items()
                        }
                    )
            return values

        def entries(handler, src, dst):
            gpu, cpu = (src, dst) if handler.gpu_to_cpu else (dst, src)
            assert handler.src_blocks_per_chunk == handler.dst_blocks_per_chunk == 1
            assert sum(gpu.group_sizes) == len(gpu.block_ids) == len(cpu.block_ids)
            offset = 0
            for group, count in enumerate(gpu.group_sizes):
                for i in range(count):
                    yield (
                        group,
                        int(gpu.block_indices[group]) + i,
                        int(gpu.block_ids[offset + i]),
                        int(cpu.block_ids[offset + i]),
                    )
                offset += count

        def submit(handler, job_id, src, dst):
            if handler.gpu_to_cpu and not source_tp:
                for group, logical, block, slot in entries(handler, src, dst):
                    filename = f"g{group}-b{logical}-j{job_id}-r{rank}.pt"
                    torch.save(snapshot(group, block), directory / filename)
                    self._gdn_records.append(
                        dict(
                            rank=rank,
                            group=group,
                            logical=logical,
                            slot=slot,
                            file=filename,
                        )
                    )
            success = original_submit(handler, job_id, src, dst)
            if success and not handler.gpu_to_cpu:
                pending[(id(handler), job_id)] = (src, dst)
            return success

        def finish(handler):
            completed = original_finish(handler)
            for result in completed:
                specs = pending.pop((id(handler), result.job_id), None)
                if specs is None:
                    continue
                assert result.success and source_tp
                provider = json.loads((directory / "provider.json").read_text())
                for group, logical, block, slot in entries(handler, *specs):
                    key = (directory / f"load-slot{slot}.key").read_text()
                    reference = provider[key]
                    assert (reference["group"], reference["logical"]) == (
                        group,
                        logical,
                    )
                    assert len(reference["files"]) == source_tp
                    shards = [
                        torch.load(directory / f, weights_only=True)
                        for f in reference["files"]
                    ]
                    actual = snapshot(group, block)
                    size = _assert_gdn_native_state(actual, shards, tp, rank)
                    self._gdn_records.append(
                        dict(
                            key=key,
                            group=group,
                            bytes=size,
                            components=sorted(
                                {name.rsplit("::", 1)[1] for name in actual}
                            ),
                        )
                    )
            return completed

        OffloadingConnectorWorker._init_worker = initialize
        SingleDirectionOffloadingHandler.transfer_async = submit
        SingleDirectionOffloadingHandler.get_finished = finish


@pytest.mark.cpu_test
@pytest.mark.parametrize("source_tp,target_tp", [(1, 2), (2, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float16])
def test_gdn_canonical_fs_cross_tp_oracle(source_tp, target_tp, conv_dtype):
    """Distinct native markers catch matching-but-wrong canonical permutations."""
    full = {}
    for i, (name, shape) in enumerate(
        [
            ("Q", (8, 3)),
            ("K", (8, 3)),
            ("V", (16, 3)),
            ("S", (8, 4, 2)),
            ("AK", (4, 11, 6)),
            ("AV", (4, 11, 6)),
        ]
    ):
        dtype = torch.float32 if name == "S" else conv_dtype
        full[name] = (
            torch.arange(torch.tensor(shape).prod()).reshape(shape) % 29 + 32 * i
        ).to(dtype)
    shards = [
        {k: v.chunk(source_tp)[r] for k, v in full.items()} for r in range(source_tp)
    ]
    for rank in range(target_tp):
        actual = {k: v.chunk(target_tp)[rank].clone() for k, v in full.items()}
        expected_bytes = sum(v.numel() * v.element_size() for v in actual.values())
        assert (
            _assert_gdn_native_state(actual, shards, target_tp, rank) == expected_bytes
        )
        for mutation in (
            dict(actual, Q=actual["K"], K=actual["Q"]),
            dict(actual, S=actual["S"].roll(1, 0)),
            dict(actual, AK=actual["AV"], AV=actual["AK"]),
        ):
            with pytest.raises(AssertionError):
                _assert_gdn_native_state(mutation, shards, target_tp, rank)
    if source_tp > 1:
        packed = torch.cat([torch.cat([s[k] for k in ("Q", "K", "V")]) for s in shards])
        wrong = dict(full, **dict(zip(("Q", "K", "V"), packed.split((8, 8, 16)))))
        with pytest.raises(AssertionError):
            _assert_gdn_native_state(wrong, shards, 1, 0)


def _gdn_fs_model_case(model, tp, root, result_path, oracle_root, source_tp):
    """Run in a fresh process so no producer GPU/primary cache can survive."""
    from vllm.v1.kv_offload.base import get_offload_group_idx
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

    directory = Path(oracle_root)
    jobs = {}
    fs_records: dict[str, list[dict]] = {"store": [], "load": []}
    original_store = FileSystemTierManager.submit_store
    original_load = FileSystemTierManager.submit_load
    original_finish = FileSystemTierManager.get_finished_jobs

    def submit(tier, job, kind):
        jobs[job.job_id] = (kind, job)
        return (original_store if kind == "store" else original_load)(tier, job)

    def finish(tier):
        completed = list(original_finish(tier))
        for result in completed:
            kind, job = jobs.pop(result.job_id)
            assert result.success
            for key, slot in zip(job.keys, job.chunk_ids):
                fs_records[kind].append(
                    dict(
                        key=key.hex(), slot=int(slot), group=get_offload_group_idx(key)
                    )
                )
                if kind == "load":
                    (directory / f"load-slot{int(slot)}.key").write_text(key.hex())
        return completed

    patches = pytest.MonkeyPatch()
    patches.setattr(
        FileSystemTierManager, "submit_store", lambda s, j: submit(s, j, "store")
    )
    patches.setattr(
        FileSystemTierManager, "submit_load", lambda s, j: submit(s, j, "load")
    )
    patches.setattr(FileSystemTierManager, "get_finished_jobs", finish)
    os.environ["VLLM_TEST_GDN_ORACLE_DIR"] = oracle_root
    os.environ["VLLM_TEST_GDN_SOURCE_TP"] = str(source_tp)
    revision = (
        "2fc06364715b967f1860aea9cf38778875588b17"
        if model == "Qwen/Qwen3.5-0.8B"
        else None
    )
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        language_model_only=True,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        distributed_executor_backend="mp",
        worker_extension_cls=(
            "tests.v1.kv_connector.unit.test_offloading_connector._GDNStateOracle"
        ),
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        num_gpu_blocks_override=32,
        gpu_memory_utilization=0.25,
        block_size=1024,
        prefix_match_unit=16,
        enable_prefix_caching=True,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        attention_config={"backend": "FLASH_ATTN"},
        additional_config={"gdn_prefill_backend": "triton"},
        enforce_eager=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "canonical_layout": True,
                "cpu_bytes_to_use": 1 << 30,
                "blocks_per_chunk": 1,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": root,
                        "n_read_threads": 2,
                        "n_write_threads": 2,
                    }
                ],
            },
        ),
    )
    try:
        result = llm.generate(
            [TokensPrompt(prompt_token_ids=[785 + i % 97 for i in range(2049)])],
            SamplingParams(
                temperature=0,
                max_tokens=64,
                ignore_eos=True,
                extra_args={
                    "kv_transfer_params": {"max_offload_tokens": 0}
                    if source_tp
                    else {"max_load_tokens": 0}
                },
            ),
            use_tqdm=False,
        )[0]
        # Drive delayed worker acks using the existing integration helper.
        _wait_for_prefix_cache_reset(llm)
        connector = llm.llm_engine.engine_core.engine_core.scheduler.connector.scheduler
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with connector.manager.lock:
                pending = connector.manager.has_pending_work() or bool(connector._jobs)
            if not pending:
                break
            llm.llm_engine.step()
            time.sleep(0.001)
        else:
            raise TimeoutError("GDN filesystem stores did not complete")
        tier = connector.manager.secondary_tiers[0]
        oracle = llm.collective_rpc("gdn_oracle_records")
        if not source_tp:
            provider = {}
            for event in fs_records["store"]:
                records = [
                    [
                        r
                        for r in per_rank
                        if (r["slot"], r["group"]) == (event["slot"], event["group"])
                    ]
                    for per_rank in oracle
                ]
                assert all(len(r) == 1 for r in records)
                assert [r[0]["rank"] for r in records] == list(range(tp))
                assert len({r[0]["logical"] for r in records}) == 1
                provider[event["key"]] = dict(
                    group=event["group"],
                    logical=records[0][0]["logical"],
                    files=[r[0]["file"] for r in records],
                )
            (directory / "provider.json").write_text(json.dumps(provider))
        llm.llm_engine.do_log_stats()
        metrics = [
            metric
            for metric in llm.get_metrics()
            if metric.name == "vllm:kv_offload_tiering_read_bytes"
        ]
        Path(result_path).write_text(
            json.dumps(
                {
                    "tokens": result.outputs[0].token_ids,
                    "cached_tokens": result.num_cached_tokens,
                    "fs_read_bytes": sum(metric.value for metric in metrics),
                    "row_size": connector.manager.secondary_tiers[0]._block_size,
                    "identity": tier.file_mapper.get_run_config(),
                    "oracle": oracle,
                    "fs_loads": fs_records["load"],
                }
            )
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
        patches.undo()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("producer_tp,consumer_tp", [(1, 2), (2, 1)])
def test_gdn_canonical_fs_cross_tp(tmp_path, producer_tp, consumer_tp):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two GPUs")
    model = os.environ.get("VLLM_TEST_GDN_MODEL", "Qwen/Qwen3.5-0.8B")
    environment = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_SSM_CONV_STATE_LAYOUT="DS",
        VLLM_KV_CACHE_LAYOUT="LBNHC",
        PYTHONHASHSEED="0",
    )

    def run(name, tp, cache_root, source_tp=0):
        result = tmp_path / f"{name}.json"
        oracle_root = tmp_path / "oracle"
        oracle_root.mkdir(exist_ok=True)
        command = (
            "import json,sys; "
            "from tests.v1.kv_connector.unit.test_offloading_connector "
            "import _gdn_fs_model_case; _gdn_fs_model_case(*json.loads(sys.argv[1]))"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                command,
                json.dumps(
                    [
                        model,
                        tp,
                        str(cache_root),
                        str(result),
                        str(oracle_root),
                        source_tp,
                    ]
                ),
            ],
            env=environment,
            check=True,
            timeout=300,
        )
        return json.loads(result.read_text())

    producer = run("producer", producer_tp, tmp_path / "shared")
    consumer = run("consumer", consumer_tp, tmp_path / "shared", producer_tp)
    assert producer["cached_tokens"] == 0
    assert consumer["cached_tokens"] == 2048
    assert consumer["fs_read_bytes"] >= consumer["row_size"] * 5
    assert consumer["identity"] == producer["identity"]
    assert consumer["identity"]["recurrent_schema"]["format"] == "gdn-ds-v1"
    assert len(consumer["oracle"]) == consumer_tp
    loaded_keys = {event["key"] for event in consumer["fs_loads"]}
    assert {event["group"] for event in consumer["fs_loads"]} == {0, 1, 2, 3}
    for records in consumer["oracle"]:
        assert {r["key"] for r in records} == loaded_keys
        assert sum(r["bytes"] for r in records) > 0
        assert {c for r in records for c in r["components"]} == {
            "Q",
            "K",
            "V",
            "S",
            "AK",
            "AV",
        }
    # Native-state bytes, not cross-TP token parity, are the correctness oracle.
    assert len(producer["tokens"]) == len(consumer["tokens"]) == 64


class MockSubscriber:
    """Helper class to receive and verify published events."""

    def __init__(
        self,
        endpoint: str,
        topic: str,
    ):
        self.ctx = zmq.Context.instance()
        self.topic_bytes = topic.encode("utf-8")

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)
        self.sub.connect(endpoint)

        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_cpu_stored_events(self) -> list[BlockStored]:
        cpu_stored_events: list[BlockStored] = []

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        poll_ms = _FIRST_EVENT_POLL_MS
        deadline = time.monotonic() + _EVENT_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            events = dict(poller.poll(poll_ms))

            if events.get(self.sub) != zmq.POLLIN:
                return cpu_stored_events

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored) and event.medium == "CPU":
                    cpu_stored_events.append(event)
                    poll_ms = 100

        return cpu_stored_events

    def close(self):
        """Clean up resources."""
        self.sub.close()


def _wait_for_prefix_cache_reset(llm: LLM) -> None:
    """Wait for async offload transfers to finish so prefix cache can reset.

    The GPU-to-CPU offload runs on a CUDA stream asynchronously. While blocks
    are still held by the offload worker, ``reset_prefix_cache`` returns
    ``False``. Between retries we send a dummy single-token prefill to force
    the engine to step, which polls the worker for completed transfers and
    frees GPU blocks.
    """
    _dummy_params = SamplingParams(max_tokens=1)
    deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck"
            )
        # Force an engine step so the scheduler polls get_finished()
        # and releases GPU blocks held by in-flight async stores.
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            _dummy_params,
            use_tqdm=False,
        )


def _latency_test(llm: LLM, subscriber: MockSubscriber | None):
    # TODO: Reintroduce latency test on ROCm once MRV2 supports cross
    # layer KV Cache. See https://github.com/vllm-project/vllm/pull/45947
    if current_platform.is_rocm():
        return
    sampling_params = SamplingParams(max_tokens=1)

    num_times_cpu_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_cpu_hit_time = 0.0
    max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len
    # Use a long prompt that fits within the model's context window.
    prompt_len = min(10001, max_model_len - 1)
    prompt_token_ids = [0] * prompt_len
    for i in tqdm(range(num_tests), desc="Running tests"):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # run generation - this should trigger saving KV cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.time() - start_time
        total_gpu_hit_time += gpu_hit_time

        # Wait for the async CPU offload to finish, then reset prefix cache
        # so the next generate() must reload from CPU rather than GPU.
        _wait_for_prefix_cache_reset(llm)

        # Verify CPU stored events arrived (offload is done before we
        # attempt to load from CPU).
        if subscriber is not None:
            assert subscriber.get_new_cpu_stored_events(), (
                f"No CPU stored events received on iteration {i}; "
                "async offload may not have completed in time"
            )

        # run generation again - this should trigger loading from CPU
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_hit_time = time.time() - start_time
        total_cpu_hit_time += cpu_hit_time

        if cpu_hit_time < cold_time:
            num_times_cpu_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    CPU hit: {total_cpu_hit_time * 1000 / num_tests:.2f}ms")

    assert num_times_cpu_better_than_cold >= 0.8 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber | None):
    sampling_params = SamplingParams(max_tokens=1)
    extra_config = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config
    )
    cpu_block_size = extra_config.get("block_size")
    if cpu_block_size is None:
        # No custom offloaded block_size: offloaded blocks match GPU blocks.
        # Use the hash block_size (cache_config.block_size) for alignment.
        cpu_block_size = llm.llm_engine.vllm_config.cache_config.block_size

    if subscriber is not None:
        subscriber.get_new_cpu_stored_events()

    # Align the reusable prefix; the final token must be recomputed for logits.
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode("Let's count to 10. One, two, three, four,")
    reusable_prefix = prompt_token_ids[:-1]
    final_token = prompt_token_ids[-1:]

    [padding_token_id] = tokenizer.encode(".", add_special_tokens=False)
    padding = [padding_token_id] * (-len(reusable_prefix) % cpu_block_size)
    insert_at = int(reusable_prefix[0] == tokenizer.bos_token_id)
    reusable_prefix[insert_at:insert_at] = padding
    assert len(reusable_prefix) % cpu_block_size == 0

    prompt = TokensPrompt(
        prompt_token_ids=reusable_prefix + final_token,
    )

    # Seed the CPU cache with the prompt.
    llm.generate(prompt, sampling_params, use_tqdm=False)

    if subscriber is not None:
        assert subscriber.get_new_cpu_stored_events()

    test_count = 20
    results = llm.generate([prompt] * test_count, sampling_params, use_tqdm=False)
    success_count = sum(1 for r in results if r.outputs[0].text == " five")
    assert success_count >= 0.5 * test_count


@pytest.mark.parametrize("model, attn_backend, cpu_block_size, uses_hma", MODEL_PARAMS)
def test_cpu_offloading(
    model: str,
    attn_backend: str | None,
    cpu_block_size: int | None,
    uses_hma: bool,
) -> None:
    """Tests OffloadingConnector with CPUOffloadingSpec."""
    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    extra_config: dict = {"cpu_bytes_to_use": 500 << 20}
    if cpu_block_size is not None:
        extra_config["block_size"] = cpu_block_size
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    subscriber: MockSubscriber | None = None
    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]

    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    # Attention-free / hybrid models disable prefix caching by default
    # (ModelConfig.is_prefix_caching_supported returns False).  Without it,
    # mamba_block_size falls back to max_model_len, making GPU blocks too
    # large for any reasonable offloaded block_size.  Force-enable it.
    force_prefix_caching = uses_hma

    llm = LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
        **({"attention_config": {"backend": attn_backend}} if attn_backend else {}),
        # Keep HMA explicitly enabled for HMA model coverage.
        **({"disable_hybrid_kv_cache_manager": False} if uses_hma else {}),
        **({"enable_prefix_caching": True} if force_prefix_caching else {}),
        # ROCm: batch size 1 to reduce variability
        **({"max_num_seqs": 1} if current_platform.is_rocm() else {}),
    )

    events_endpoint = events_endpoint.replace("*", "127.0.0.1")
    subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        if subscriber is not None:
            subscriber.close()
        del llm


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_cpu_offloading_metrics() -> None:
    """Verify that offloading Prometheus metrics (new flat and deprecated
    labeled) are emitted after stores and loads."""
    extra_config: dict = {
        "cpu_bytes_to_use": 500 << 20,
        "block_size": CPU_BLOCK_SIZES,
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        disable_log_stats=False,
    )

    try:
        prompt_token_ids = list(range(500))

        # First generate: cold run, triggers a store to CPU.
        # Use max_tokens>1 so the request is still producing output
        # tokens when the async store completes and stats get drained.
        # (The LLMEngine only records stats on steps with request outputs.)
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            SamplingParams(max_tokens=10),
            use_tqdm=False,
        )

        # Wait for the async offload to finish, then reset GPU prefix cache
        # so the next generate must load from CPU.
        _wait_for_prefix_cache_reset(llm)

        # Second generate: triggers a load from CPU.
        # Send a short filler alongside the load prompt so there's always
        # a request producing output tokens when the load stats get drained
        # (the LLMEngine only records stats on steps with request outputs).
        filler = TokensPrompt(prompt_token_ids=[0])
        llm.generate(
            [filler, TokensPrompt(prompt_token_ids=prompt_token_ids)],
            SamplingParams(max_tokens=50),
            use_tqdm=False,
        )

        # Metric helpers.
        registry = prometheus_client.REGISTRY

        def _get_counter_value(
            name: str, labels: dict[str, str] | None = None
        ) -> float:
            total = 0.0
            for metric in registry.collect():
                if metric.name == name:
                    for sample in metric.samples:
                        if sample.name != name + "_total":
                            continue
                        if labels and not all(
                            sample.labels.get(k) == v for k, v in labels.items()
                        ):
                            continue
                        total += sample.value
            return total

        def _get_histogram_count(
            name: str, labels: dict[str, str] | None = None
        ) -> float:
            total = 0.0
            for metric in registry.collect():
                if metric.name == name:
                    for sample in metric.samples:
                        if sample.name != name + "_count":
                            continue
                        if labels and not all(
                            sample.labels.get(k) == v for k, v in labels.items()
                        ):
                            continue
                        total += sample.value
            return total

        # New flat counter metrics
        store_bytes = _get_counter_value("vllm:kv_offload_store_bytes")
        assert store_bytes > 0, f"Expected store_bytes > 0, got {store_bytes}"
        load_bytes = _get_counter_value("vllm:kv_offload_load_bytes")
        assert load_bytes > 0, f"Expected load_bytes > 0, got {load_bytes}"
        store_time = _get_counter_value("vllm:kv_offload_store_time")
        assert store_time > 0, f"Expected store_time > 0, got {store_time}"
        load_time = _get_counter_value("vllm:kv_offload_load_time")
        assert load_time > 0, f"Expected load_time > 0, got {load_time}"

        # New flat histogram metrics
        store_size_count = _get_histogram_count("vllm:kv_offload_store_size")
        assert store_size_count > 0, (
            f"Expected store_size histogram observations > 0, got {store_size_count}"
        )
        load_size_count = _get_histogram_count("vllm:kv_offload_load_size")
        assert load_size_count > 0, (
            f"Expected load_size histogram observations > 0, got {load_size_count}"
        )

        # Deprecated labeled metrics — verify per transfer_type label.
        load_label = {"transfer_type": "CPU_to_GPU"}
        store_label = {"transfer_type": "GPU_to_CPU"}

        dep_load_bytes = _get_counter_value("vllm:kv_offload_total_bytes", load_label)
        assert dep_load_bytes > 0, (
            f"Expected deprecated load bytes > 0, got {dep_load_bytes}"
        )
        dep_store_bytes = _get_counter_value("vllm:kv_offload_total_bytes", store_label)
        assert dep_store_bytes > 0, (
            f"Expected deprecated store bytes > 0, got {dep_store_bytes}"
        )
        dep_load_time = _get_counter_value("vllm:kv_offload_total_time", load_label)
        assert dep_load_time > 0, (
            f"Expected deprecated load time > 0, got {dep_load_time}"
        )
        dep_store_time = _get_counter_value("vllm:kv_offload_total_time", store_label)
        assert dep_store_time > 0, (
            f"Expected deprecated store time > 0, got {dep_store_time}"
        )
        dep_load_size = _get_histogram_count("vllm:kv_offload_size", load_label)
        assert dep_load_size > 0, (
            f"Expected deprecated load size observations > 0, got {dep_load_size}"
        )
        dep_store_size = _get_histogram_count("vllm:kv_offload_size", store_label)
        assert dep_store_size > 0, (
            f"Expected deprecated store size observations > 0, got {dep_store_size}"
        )

        # Flat and deprecated metrics must be consistent (dual-write).
        assert store_bytes == dep_store_bytes
        assert load_bytes == dep_load_bytes
        assert store_time == dep_store_time
        assert load_time == dep_load_time
        assert store_size_count == dep_store_size
        assert load_size_count == dep_load_size
    finally:
        del llm


def test_tiering_offloading() -> None:
    """Tests OffloadingConnector with TieringOffloadingSpec."""
    extra_config: dict = {
        "cpu_bytes_to_use": 500 << 20,
        "block_size": CPU_BLOCK_SIZES,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "example"}],
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"),
        topic=kv_events_config.topic,
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm


def test_fs_tiering_offloading(tmp_path) -> None:
    """Tests OffloadingConnector with TieringOffloadingSpec
    + fs secondary tier."""
    extra_config: dict = {
        "cpu_bytes_to_use": 1 << 30,
        "block_size": CPU_BLOCK_SIZES,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "fs", "root_dir": str(tmp_path)}],
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"),
        topic=kv_events_config.topic,
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="HMA mamba-align CPU offload test is CUDA-only",
)
@pytest.mark.parametrize(
    "model,block_size,tp_size",
    [
        # ("Qwen/Qwen3.6-35B-A3B", 1056, 2),
        # ("tiiuae/falcon-mamba-7b", 16, 1),
        ("state-spaces/mamba-1.4b-hf", 16, 1),
    ],
)
def test_mamba_cpu_offload_boundary(model: str, block_size: int, tp_size: int):
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 4 << 30,
            "block_size": block_size,
        },
    )
    llm = LLM(
        model=model,
        max_model_len=block_size * 10,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tp_size,
        kv_transfer_config=kv_transfer_config,
        language_model_only=True,
        enable_prefix_caching=True,
        # Use lossless state storage so this exact-equality regression isolates
        # offload boundary selection from low-precision Mamba checkpointing.
        mamba_cache_dtype="float32",
        mamba_ssm_cache_dtype="float32",
        disable_hybrid_kv_cache_manager=False,
    )

    _PROMPT_SIZE: int = block_size * 2
    # Cold prefill and offload-resume are not bit-exact (selective_scan_fn over
    # the whole prompt vs selective_state_update for the recomputed token), so
    # exact-text equality only holds while the rounding difference never flips a
    # greedy argmax. Use a long, highly predictable prompt: it needs no "...."
    # padding and keeps top-1 ahead of top-2 by >2.8 logprobs at every step.
    _PROMPT_TEXT = (
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. "
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. "
    )

    # build prompt ids to match prompt_size
    tokenizer = llm.get_tokenizer()
    raw_ids: list[int] = tokenizer.encode(_PROMPT_TEXT)
    while len(raw_ids) < _PROMPT_SIZE:
        raw_ids = tokenizer.encode("....") + raw_ids
    initial_ids: list[int] = raw_ids[:_PROMPT_SIZE]

    sampling_params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)

    failures: list[str] = []

    def _get_output_str(outputs):
        return outputs[0].outputs[0].text

    def _verify(llm, prompt, label: str):
        cold_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        _wait_for_prefix_cache_reset(llm)
        cpu_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

        cold_text = _get_output_str(cold_outputs)
        cpu_text = _get_output_str(cpu_outputs)
        print(f"{label} : cold outputs\n{cold_text}")
        print(f"{label} : cpu outputs\n{cpu_text}")

        if cold_text != cpu_text:
            failures.append(
                f"{label}: mismatch\n  cold: {cold_text!r}\n  cpu:  {cpu_text!r}"
            )

    try:
        # A cached Mamba entry is a point state. CPU cache stores are triggered
        # at offload block boundaries. When the prompt is exactly at a boundary,
        # CPU offload must not load that boundary's cached state because it already
        # includes the last token that must be recomputed.
        # This is fine for other attention types as we have all the necessary
        # token KV values in the hit blocks.
        prompt = TokensPrompt(prompt_token_ids=initial_ids)
        _verify(llm, prompt, "block-boundary-prompt")

        # Test for prompt token ids at non-block boundaries.
        # Reuse is okay for this case.
        prompt = TokensPrompt(prompt_token_ids=[0] + initial_ids)
        _verify(llm, prompt, "block-mid-prompt")

        assert not failures, "\n\n".join(failures)

    finally:
        del llm
