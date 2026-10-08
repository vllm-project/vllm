# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError
from unittest.mock import patch

import pytest
import torch

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )
    assert vllm_config.cache_config.prefix_cache_retention_interval == 0

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])

    args = parser.parse_args(["--prefix-cache-retention-interval", "64"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_cache_retention_interval == 64


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_mm_prefix_lm_raises_batched_tokens_floor():
    """Verify that prefix-LM multimodal models auto-raise
    max_num_batched_tokens to fit at least one multimodal item.

    Regression test for https://github.com/vllm-project/vllm/issues/42687
    """
    from unittest.mock import patch

    # Simulate a prefix-LM multimodal model whose largest modality
    # (video) requires 2496 tokens — more than the 2048 default.
    fake_mm_min = (2496, "video")

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_model_len=2048,
        enforce_eager=True,
    )

    with (
        patch.object(
            type(engine_args),
            "_get_min_mm_batched_tokens",
            staticmethod(lambda _mc: fake_mm_min),
        ),
        patch(
            "vllm.config.ModelConfig.is_multimodal_model",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "vllm.config.ModelConfig.is_mm_prefix_lm",
            new_callable=lambda: property(lambda self: True),
        ),
    ):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.scheduler_config.max_num_batched_tokens >= 2496


def test_custom_histogram_buckets_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # Unset: byte-identical default behavior.
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.observability_config.custom_histogram_buckets is None

    # A JSON mapping round-trips into the config verbatim.
    args = parser.parse_args(
        [
            "--custom-histogram-buckets",
            '{"request_latency": [0.01, 0.05, 0.1, 0.5]}',
        ]
    )
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.observability_config.custom_histogram_buckets == {
        "request_latency": [0.01, 0.05, 0.1, 0.5]
    }

    # Malformed JSON is rejected at argument-parsing time.
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        parser.parse_args(["--custom-histogram-buckets", "{not json"])

    # An unknown family key parses as JSON but fails config validation.
    args = parser.parse_args(["--custom-histogram-buckets", '{"bogus": [1.0, 2.0]}'])
    with pytest.raises(ValueError, match="unknown bucket family"):
        EngineArgs.from_cli_args(args=args).create_engine_config()


def test_data_parallel_start_rank_zero_infers_hybrid_lb():
    """An explicit --data-parallel-start-rank 0 must be treated the same as
    any other explicit start rank when inferring hybrid LB mode, not as
    "unset" (regression test for a truthiness-vs-`is not None` bug).
    """
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_start_rank=0,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_hybrid_lb is True
    assert vllm_config.parallel_config.data_parallel_rank == 0


def test_external_lb_preserves_explicit_rank_when_dp_exceeds_nodes():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=4,
        data_parallel_rank=3,
        data_parallel_external_lb=True,
        nnodes=2,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 3


@pytest.mark.parametrize(
    ("data_parallel_size", "nnodes", "tensor_parallel_size"),
    [(4, 2, 1), (2, 3, 3)],
)
def test_external_lb_requires_explicit_rank_when_nodes_are_not_evenly_partitioned(
    data_parallel_size, nnodes, tensor_parallel_size
):
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=data_parallel_size,
        data_parallel_external_lb=True,
        tensor_parallel_size=tensor_parallel_size,
        nnodes=nnodes,
        node_rank=1,
    )

    with (
        patch.object(ModelConfig, "is_moe", new=property(lambda self: True)),
        pytest.raises(
            ValueError,
            match="Set a unique `--data-parallel-rank`",
        ),
    ):
        engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)


def test_external_lb_infers_rank_when_dp_does_not_exceed_nodes():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=2,
        data_parallel_external_lb=True,
        nnodes=2,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 1


def test_external_lb_infers_rank_for_multinode_replicas():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=2,
        data_parallel_external_lb=True,
        tensor_parallel_size=12,
        nnodes=4,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 0


def test_extensible_kv_cache_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_extensible_kv_cache is None
    assert engine_args.gpu_memory_utilization is None

    args = parser.parse_args(["--enable-extensible-kv-cache"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_extensible_kv_cache

    args = parser.parse_args(["--no-enable-extensible-kv-cache"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_extensible_kv_cache is False


# Off CUDA the platform check trips first, before the behavior under test.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
def test_extensible_kv_cache_defaults():
    """Unset, the extensible KV cache is on for the V2 runner on CUDA and the
    budget is the whole device; turned off, or where unsupported, the standard
    0.92 default returns. An explicit utilization is always kept."""
    config = EngineArgs(model="facebook/opt-125m").create_engine_config(
        UsageContext.OPENAI_API_SERVER
    )
    assert config.cache_config.enable_extensible_kv_cache
    assert config.cache_config.gpu_memory_utilization is None
    assert config.cache_config.resolved_gpu_memory_utilization == 1.0

    config.cache_config.enable_extensible_kv_cache = False
    assert config.cache_config.resolved_gpu_memory_utilization == 0.92

    config = EngineArgs(
        model="facebook/opt-125m", enable_extensible_kv_cache=False
    ).create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert config.cache_config.resolved_gpu_memory_utilization == 0.92

    config = EngineArgs(
        model="facebook/opt-125m", gpu_memory_utilization=0.5
    ).create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert config.cache_config.enable_extensible_kv_cache
    assert config.cache_config.resolved_gpu_memory_utilization == 0.5
    config.cache_config.enable_extensible_kv_cache = False
    assert config.cache_config.resolved_gpu_memory_utilization == 0.5

    # Manual KV sizing leaves the feature off by default but rejects an
    # explicit request.
    config = EngineArgs(
        model="facebook/opt-125m", kv_cache_memory_bytes=1 << 30
    ).create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert not config.cache_config.enable_extensible_kv_cache
    assert config.cache_config.resolved_gpu_memory_utilization == 0.92
    config = EngineArgs(
        model="facebook/opt-125m", num_gpu_blocks_override=64
    ).create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert not config.cache_config.enable_extensible_kv_cache


@pytest.mark.parametrize(
    "manual_size",
    [dict(kv_cache_memory_bytes=1 << 30), dict(num_gpu_blocks_override=64)],
)
@requires_cuda
def test_extensible_kv_cache_rejects_manual_kv_cache_size(manual_size):
    """Measured sizing and a manual size conflict: silently clamping the manual
    one would misreport the cache, so an explicit request is an error."""
    (arg_name,) = manual_size
    engine_args = EngineArgs(
        model="facebook/opt-125m", enable_extensible_kv_cache=True, **manual_size
    )
    with pytest.raises(ValueError, match=arg_name):
        engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)


def _fake_executor(vllm_config, collective_rpc):
    from types import MethodType, SimpleNamespace

    from vllm.v1.executor.abstract import Executor

    fake = SimpleNamespace(vllm_config=vllm_config, collective_rpc=collective_rpc)
    fake._extensible_kv_cache_unsupported_reason = MethodType(
        Executor._extensible_kv_cache_unsupported_reason, fake
    )
    return fake


@requires_cuda
def test_extensible_kv_cache_falls_back_when_driver_unsupported():
    from vllm.v1.executor.abstract import Executor

    calls: list[str] = []

    def collective_rpc(method: str):
        calls.append(method)
        if method == "extensible_kv_cache_unsupported_reason":
            return [None, "no VMM support"]
        return [None, None]

    engine_args = EngineArgs(model="facebook/opt-125m", enable_extensible_kv_cache=True)
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.cache_config.enable_extensible_kv_cache
    assert vllm_config.cache_config.resolved_gpu_memory_utilization == 1.0
    specs = [{"layer": object()}]
    fake = _fake_executor(vllm_config, collective_rpc)
    Executor.resolve_extensible_kv_cache(fake, specs)
    assert not vllm_config.cache_config.enable_extensible_kv_cache
    assert vllm_config.cache_config.resolved_gpu_memory_utilization == 0.92
    assert calls == [
        "extensible_kv_cache_unsupported_reason",
        "disable_extensible_kv_cache",
    ]

    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    fake = _fake_executor(vllm_config, lambda method: [None, None])
    Executor.resolve_extensible_kv_cache(fake, specs)
    assert vllm_config.cache_config.enable_extensible_kv_cache

    # No KV cache at all: nothing to size, so the feature is turned off.
    Executor.resolve_extensible_kv_cache(fake, [{}])
    assert not vllm_config.cache_config.enable_extensible_kv_cache


@requires_cuda
def test_external_launcher_ranks_agree_on_extensible_kv_cache(monkeypatch):
    """Under torchrun each rank probes only its own driver; a rank whose probe
    passes must still follow one whose probe fails, or it hangs waiting for
    the others in the block-count all-reduce."""
    import torch.distributed as dist

    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

    class FakeExecutor(ExecutorWithExternalLauncher):
        def __init__(self, vllm_config, collective_rpc):
            self.vllm_config = vllm_config
            self.collective_rpc = collective_rpc

    calls: list[str] = []

    def collective_rpc(method: str):
        calls.append(method)
        return [None]

    reduced: list[tuple[int, object]] = []

    def all_reduce(value: int, op):
        reduced.append((value, op))
        return 1  # Some other rank reported "unsupported".

    monkeypatch.setattr(
        ExecutorWithExternalLauncher, "_all_reduce", staticmethod(all_reduce)
    )
    engine_args = EngineArgs(model="facebook/opt-125m", enable_extensible_kv_cache=True)
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    FakeExecutor(vllm_config, collective_rpc).resolve_extensible_kv_cache(
        [{"layer": object()}]
    )
    assert reduced == [(0, dist.ReduceOp.MAX)]
    assert not vllm_config.cache_config.enable_extensible_kv_cache
    assert calls == [
        "extensible_kv_cache_unsupported_reason",
        "disable_extensible_kv_cache",
    ]


def test_executor_agrees_committable_blocks_across_workers():
    """Workers report what warmup may commit from `initialize_from_config`;
    the executor passes the minimum to `compile_or_warm_up_model`. Workers
    without an extensible cache report None and get no argument."""
    from types import SimpleNamespace

    from vllm.v1.executor.abstract import Executor

    calls: list[tuple[str, tuple]] = []
    reported: list[int | None] = [120, 96]

    def collective_rpc(method: str, args=()):
        calls.append((method, args))
        return reported if method == "initialize_from_config" else []

    fake = SimpleNamespace(collective_rpc=collective_rpc)
    Executor.compile_or_warm_up_model(
        fake, Executor.initialize_from_config(fake, ["cfg"])
    )
    reported = [None, None]
    Executor.compile_or_warm_up_model(
        fake, Executor.initialize_from_config(fake, ["cfg"])
    )
    assert calls == [
        ("initialize_from_config", (["cfg"],)),
        ("compile_or_warm_up_model", (96,)),
        ("initialize_from_config", (["cfg"],)),
        ("compile_or_warm_up_model", ()),
    ]


@pytest.mark.parametrize("requested", [None, True])
def test_extensible_kv_cache_off_with_elastic_ep(monkeypatch, requested):
    """Elastic EP scale-up skips warmup and reuses the profiled KV cache size,
    so nothing would measure an extensible cache: it stays off by default and an
    explicit request is an error."""
    from types import SimpleNamespace

    import vllm.platforms
    from vllm.config.vllm import VllmConfig

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            _extensible_kv_cache_resolved=False,
            enable_extensible_kv_cache=requested,
            kv_cache_memory_bytes=None,
            num_gpu_blocks_override=None,
        ),
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(enable_elastic_ep=True),
        kv_transfer_config=None,
    )
    if requested:
        with pytest.raises(ValueError, match="elastic EP"):
            VllmConfig._resolve_extensible_kv_cache(config)
    else:
        VllmConfig._resolve_extensible_kv_cache(config)
        assert config.cache_config.enable_extensible_kv_cache is False


@requires_cuda
@pytest.mark.parametrize("multi", [False, True])
def test_extensible_kv_cache_rejects_connector_memory_pool(multi):
    """A connector's custom memory pool (Mooncake NVLink/BAREX) must own the KV
    allocation, which the driver-mapped extensible cache bypasses: the feature
    stays off by default and an explicit request is an error."""
    from vllm.config.kv_transfer import KVTransferConfig

    mooncake = {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"custom_mem_pool": "NVLINK"},
    }
    if multi:
        transfer = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"connectors": [mooncake]},
        )
    else:
        transfer = KVTransferConfig(**mooncake)

    def make_config(**kwargs):
        return EngineArgs(
            model="facebook/opt-125m", kv_transfer_config=transfer, **kwargs
        ).create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert not make_config().cache_config.enable_extensible_kv_cache
    with pytest.raises(ValueError, match="custom_mem_pool"):
        make_config(enable_extensible_kv_cache=True)


@requires_cuda
def test_extensible_kv_cache_connector_needs_block_compact_layout():
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    def make_config():
        engine_args = EngineArgs(
            model="facebook/opt-125m",
            enable_extensible_kv_cache=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector", kv_role="kv_both"
            ),
        )
        return engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    layout = resolve_kv_cache_layout(make_config(), [["LHBNC", "LBNHC"]])
    assert layout.name == "LBNHC"
    with pytest.raises(ValueError, match="block-compact"):
        resolve_kv_cache_layout(make_config(), [["LHBNC"]])
