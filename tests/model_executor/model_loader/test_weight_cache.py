# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest
import regex as re

from tests.utils import create_new_process_for_each_test
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


class WeightCacheDaemon:
    """Context manager running the real weight cache daemon as a subprocess."""

    def __init__(
        self,
        model: str,
        tp_size: int,
        extra_args: list[str] | None = None,
    ):
        # Short base path: Unix socket paths are limited to ~107 characters.
        self.socket_dir = tempfile.mkdtemp(prefix="vllm_ipc_")
        self.health_port = get_open_port()
        self._cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "preload",
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--weight-cache-socket-dir",
            self.socket_dir,
            "--enforce-eager",
            "--weight-cache-health-host",
            "127.0.0.1",
            "--weight-cache-health-port",
            str(self.health_port),
            *(extra_args or []),
        ]
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "WeightCacheDaemon":
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        self._proc.stderr.read()

    def check_health(self) -> None:
        url = f"http://127.0.0.1:{self.health_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                assert response.status == 200
        except urllib.error.HTTPError as error:
            raise AssertionError(
                f"Weight cache daemon is not ready: HTTP {error.code}"
            ) from error

    def wait_until_ready(self, timeout_s: float = 900) -> None:
        """Poll until READY (which covers the autotune tuners) or the preload
        process dies."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.check_health()
                return
            except (AssertionError, urllib.error.URLError):
                if time.monotonic() > deadline:
                    raise
                if self._proc is not None and self._proc.poll() is not None:
                    raise AssertionError("vllm preload exited before ready") from None
                time.sleep(2)

    def _stop(self) -> None:
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


@dataclass
class ModelCase:
    model: str
    prompts: list[str]
    images: list | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    daemon_args: list[str] = field(default_factory=list)
    # Additionally assert the daemons tune in place before serving and write
    # the on-disk FlashInfer autotune cache, and the restart engine reuses it
    # (needs SM90+/FlashInfer).
    check_flashinfer_cache: bool = False


def generate(
    vllm_runner,
    case: ModelCase,
    socket_dir: str | None,
    fallback: bool,
):
    extra_config = (
        {} if socket_dir is None else {"socket_dir": socket_dir, "fallback": fallback}
    )
    with vllm_runner(
        case.model,
        load_format="auto" if socket_dir is None else "ipc_cache",
        model_loader_extra_config=extra_config,
        # Greedy outputs are compared across engine restarts; cap the batch
        # size so scheduling cannot change the results.
        max_num_seqs=1,
        **case.llm_kwargs,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        return llm.generate(case.prompts, sampling_params, images=case.images)


# Both hybrid-recurrent models need chunked prefill for their mamba-style
# cache mode.
QWEN_CASE = ModelCase(
    model="Qwen/Qwen3.5-0.8B",
    prompts=[
        "Hello, my name is",
        "The capital of France is",
    ],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_chunked_prefill=True,
    ),
)

# Kimi K3 adds the mxfp4-pack MoE path (post-load kernel setup runs in
# pre-processed mode) and a vision tower, exercised with an image request.
K3_CASE = ModelCase(
    model="tiny-random/kimi-k3",
    prompts=[
        "The capital of France is",
        "def quicksort(arr):",
        "<|kimi_image_placeholder|>What is shown in the image?",
    ],
    images=[None, None, ImageAsset("stop_sign").pil_image],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        # The image placeholder expands to >1k tokens for the stop_sign asset.
        max_model_len=4096,
    ),
    daemon_args=["--trust-remote-code"],
    # Kimi K3 has tunable FlashInfer ops (a dense model like Qwen2.5 tunes
    # nothing and upstream never writes the cache file for it).
    check_flashinfer_cache=True,
)

# Qwen3.5-0.8B ships one MTP layer in the target checkpoint, so method="mtp"
# loads the draft from the same model. The daemon must cache it in its draft
# group for the warm runs (fallback=False) to succeed.
QWEN_MTP_CASE = ModelCase(
    model="Qwen/Qwen3.5-0.8B",
    prompts=[
        "Hello, my name is",
        "The capital of France is",
    ],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_chunked_prefill=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    ),
    daemon_args=[
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 1}',
    ],
)


DSV2_EP_CASE = ModelCase(
    model="deepseek-ai/DeepSeek-V2-Lite-Chat",
    prompts=["The capital of France is"],
    llm_kwargs=dict(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=1024,
    ),
    daemon_args=["--enable-expert-parallel", "--dtype", "bfloat16"],
)


@pytest.mark.parametrize(
    "case",
    [QWEN_CASE, K3_CASE, QWEN_MTP_CASE],
    ids=["qwen3.5", "kimi-k3", "qwen3.5-mtp"],
)
def test_ipc_cache_cold_start_and_warm_restart(
    vllm_runner, case: ModelCase, tmp_path, monkeypatch, capfd
):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon — for the MTP case, both the target's and the
    draft's daemon groups.

    Cases with ``check_flashinfer_cache`` additionally cover in-daemon
    autotune (default-on with `--enable-flashinfer-autotune`): each daemon
    tunes before binding its socket and must write the on-disk FlashInfer
    autotune cache, and the restart engine must load tactics from it instead
    of re-profiling. The daemon's cache file is keyed with the
    OPENAI_API_SERVER batch defaults while these engines run under LLM_CLASS,
    so they never hit the daemon's file by design; reuse is proven against
    the file the earlier in-process engines wrote.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("Weight cache IPC sharing requires CUDA or ROCm")
    if case is K3_CASE and not current_platform.is_device_capability_family(100):
        pytest.skip("Kimi K3 IPC weight cache requires an SM100 MXFP4 backend")
    if case.check_flashinfer_cache:
        from vllm.utils.flashinfer import has_flashinfer

        if not (has_flashinfer() and current_platform.has_device_capability(90)):
            pytest.skip("FlashInfer autotune requires FlashInfer and SM90+")
        # Isolate the on-disk autotune cache. The engines are in-process
        # (envs read lazily); the daemon subprocesses spawn later and
        # inherit it.
        monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    run_cold_start_and_warm_restart(vllm_runner, case, tmp_path, capfd)


@create_new_process_for_each_test()
@pytest.mark.distributed(num_gpus=2)
def test_ipc_cache_cold_start_and_warm_restart_tp2_ep(vllm_runner, tmp_path):
    """Same as above, with expert parallelism enabled at tp=2."""
    # No capfd: a CaptureFixture does not survive the cloudpickle into the
    # child, and this case never reaches the check_flashinfer_cache branch.
    run_cold_start_and_warm_restart(vllm_runner, DSV2_EP_CASE, tmp_path)


def run_cold_start_and_warm_restart(vllm_runner, case: ModelCase, tmp_path, capfd=None):
    # Baseline: plain disk loading with the default loader.
    baseline_outputs = generate(
        vllm_runner,
        case,
        None,
        fallback=True,
    )
    assert all(text for _, texts in baseline_outputs for text in texts)

    # Cold start: no daemon is serving, so the loader falls back to disk.
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_empty_") as empty_socket_dir:
        cold_outputs = generate(vllm_runner, case, empty_socket_dir, fallback=True)

    if case.check_flashinfer_cache:
        # The baseline engine already wrote its own (LLM_CLASS-keyed) table.
        engine_cache_files = set(tmp_path.rglob("autotune_configs*.json"))

    with WeightCacheDaemon(
        case.model,
        tp_size=case.llm_kwargs.get("tensor_parallel_size", 1),
        extra_args=case.daemon_args,
    ) as d:
        warm_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        d.wait_until_ready()
        if case.check_flashinfer_cache:
            # Each daemon tunes before its ready report; exactly one new file
            # (their OPENAI_API_SERVER-keyed table) must appear.
            tuner_files = set(tmp_path.rglob("autotune_configs*.json"))
            tuner_files -= engine_cache_files
            assert len(tuner_files) == 1, (
                f"expected the tuner's tuned table, got {tuner_files}"
            )
        # Warm restart: a second engine lifetime against the same daemon.
        if case.check_flashinfer_cache:
            capfd.readouterr()  # drain: the assertion reads the increment
        restart_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        if case.check_flashinfer_cache:
            # The restart's EngineCore subprocess inherits pytest's fds; the
            # flashinfer autotuner logs "[Autotuner]: Loaded N configs from
            # <path>" (INFO, not once-deduplicated) on a disk cache hit.
            out = capfd.readouterr()
            m = re.search(
                r"\[Autotuner\]: Loaded (\d+) configs from", out.out + out.err
            )
            assert m and int(m.group(1)) > 0, (
                "restart engine did not load the on-disk autotune cache"
            )

    assert cold_outputs == baseline_outputs
    assert warm_outputs == baseline_outputs
    assert restart_outputs == baseline_outputs


def _parallel(**kw):
    from types import SimpleNamespace

    base = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        nnodes=1,
        node_rank=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_daemon_places_tp_and_dp_ranks_on_local_gpus():
    """Each node serves a contiguous global-rank block (DP-major), so without
    DP local GPU i is TP rank r*local+i, and single-node DP launchers offset
    by --data-parallel-start-rank."""
    from vllm.model_executor.model_loader.weight_cache.daemon import plan_local_ranks

    tp_second_node = _parallel(tensor_parallel_size=8, nnodes=2, node_rank=1)
    assert plan_local_ranks(tp_second_node) == [(i, 0, 0, 4 + i) for i in range(4)]

    dp_third_node = _parallel(
        data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=8
    )
    assert plan_local_ranks(dp_third_node) == [(i, 8 + i, 0, 0) for i in range(4)]

    dp_tp = _parallel(
        tensor_parallel_size=2,
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=2,
    )
    assert plan_local_ranks(dp_tp) == [
        (0, 2, 0, 0),
        (1, 2, 0, 1),
        (2, 3, 0, 0),
        (3, 3, 0, 1),
    ]

    # DP + nnodes: node-local TP replicas
    dp_tp_node0 = _parallel(tensor_parallel_size=8, data_parallel_size=2, nnodes=2)
    assert plan_local_ranks(dp_tp_node0) == [(i, 0, 0, i) for i in range(8)]
    dp_tp_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=2, node_rank=1
    )
    assert plan_local_ranks(dp_tp_node1) == [(i, 1, 0, i) for i in range(8)]

    # DP + nnodes: TP group spans nodes (TP8 x DP2 on 4 nodes)
    tp_span_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=1
    )
    assert plan_local_ranks(tp_span_node1) == [(i, 0, 0, 4 + i) for i in range(4)]
    tp_span_node2 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=2
    )
    assert plan_local_ranks(tp_span_node2) == [(i, 1, 0, i) for i in range(4)]

    # DP + nnodes: several DP replicas per node (TP4 x DP4 on 2 nodes)
    dp_multi_node0 = _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    assert plan_local_ranks(dp_multi_node0) == [(i, i // 4, 0, i % 4) for i in range(8)]

    # PP adds one daemon placement per stage while preserving TP rank order.
    pp_tp = _parallel(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert plan_local_ranks(pp_tp) == [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (2, 0, 1, 0),
        (3, 0, 1, 1),
    ]


def test_daemon_rejects_unmappable_parallelism():
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        _reject_unsupported_parallelism,
    )

    _reject_unsupported_parallelism(
        _parallel(
            data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=12
        )
    )
    # DP combines with --nnodes when the world size divides evenly.
    _reject_unsupported_parallelism(
        _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    )
    _reject_unsupported_parallelism(_parallel(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(_parallel(tensor_parallel_size=3, nnodes=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(
            _parallel(tensor_parallel_size=2, data_parallel_size=3, nnodes=4)
        )
    with pytest.raises(ValueError, match="exceeds"):
        _reject_unsupported_parallelism(
            _parallel(
                data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=13
            )
        )


def test_daemon_keys_dense_dp_ranks_like_the_engine():
    """The engine runs non-MoE DP ranks as independent DP=1 replicas, so the
    daemon must fingerprint them the same way or no engine can map them."""
    from types import SimpleNamespace

    from vllm.config import ParallelConfig
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        get_engine_dp_placement,
    )

    def daemon_config(is_moe: bool):
        return SimpleNamespace(
            model_config=SimpleNamespace(is_moe=is_moe),
            parallel_config=ParallelConfig(data_parallel_size=4),
        )

    dense_engine_rank = ParallelConfig(data_parallel_size=4, data_parallel_rank=3)
    dense_engine_rank.reconfigure_for_independent_dp_rank()
    assert get_engine_dp_placement(daemon_config(is_moe=False), dp_rank=3) == (
        dense_engine_rank.data_parallel_size,
        dense_engine_rank.data_parallel_rank,
    )
    # MoE ranks stay in one DP group and keep their placement.
    assert get_engine_dp_placement(daemon_config(is_moe=True), dp_rank=3) == (4, 3)


def test_daemon_sets_its_rank_like_engine_workers(monkeypatch):
    """Per-rank state such as the torch.compile cache is keyed by
    parallel_config.rank, so daemons left at rank 0 would share rank 0's
    compiled artifact and run every rank's kernels on rank 0's device."""
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache import daemon

    monkeypatch.setattr(
        daemon.WeightCacheKey, "from_model_config", lambda *args, **kwargs: None
    )
    for global_rank in range(4):
        vllm_config = SimpleNamespace(
            parallel_config=_parallel(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                enable_eplb=False,
                enable_expert_parallel=False,
                expert_placement_strategy="linear",
                rank=0,
            ),
            model_config=SimpleNamespace(is_moe=False),
            cache_config=None,
        )
        server = daemon.WeightCacheDaemon(
            vllm_config, global_rank, global_rank, "tcp://127.0.0.1:0"
        )
        assert server.vllm_config.parallel_config.rank == global_rank


def test_weight_cache_key_distinguishes_dp_ranks():
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    key = WeightCacheKey(
        checkpoint="ckpt",
        model_arch="Arch",
        tp_size=1,
        tp_rank=0,
        dtype="bf16",
        quantization=None,
        quant_config_hash="h",
        rope_parameters_hash="initial",
        kv_cache_dtype="auto",
        kv_cache_dtype_skip_layers=(),
        revision=None,
        vllm_version="v",
        dp_size=16,
        dp_rank=3,
        pp_size=2,
        pp_rank=1,
    )
    assert key.mismatched_fields(replace(key, dp_rank=4)) == ["dp_rank"]
    assert key.mismatched_fields(replace(key, dp_size=8, dp_rank=3)) == ["dp_size"]
    assert key.mismatched_fields(replace(key, pp_rank=0)) == ["pp_rank"]


@pytest.mark.parametrize("hf_quant_config", [None, {"quant_algo": "NVFP4"}])
def test_weight_cache_key_distinguishes_nvfp4_activation_override(
    tmp_path, hf_quant_config
):
    """Cache keys isolate activation modes and reject ambiguous legacy hashes."""
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from vllm.config import CacheConfig
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey
    from vllm.utils.hashing import safe_hash

    model_config = SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        quantization="modelopt_fp4",
        quantization_config=None,
        revision=None,
        hf_text_config=SimpleNamespace(),
        hf_config=SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
            quantization_config=hf_quant_config,
        ),
    )
    static_key = WeightCacheKey.from_model_config(
        model_config, tp_size=1, tp_rank=0, cache_config=CacheConfig()
    )
    model_config.quantization_config = QuantizationConfigArgs(
        moe={"activation": "nvfp4_per_token"}
    )
    per_token_key = WeightCacheKey.from_model_config(
        model_config, tp_size=1, tp_rank=0, cache_config=CacheConfig()
    )
    assert static_key.mismatched_fields(per_token_key) == ["quant_config_hash"]

    legacy_hash = (
        ""
        if hf_quant_config is None
        else safe_hash(
            json.dumps(hf_quant_config, sort_keys=True).encode(),
            usedforsecurity=False,
        ).hexdigest()
    )
    assert static_key.mismatched_fields(
        replace(static_key, quant_config_hash=legacy_hash)
    ) == ["quant_config_hash"]


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("nested", [False, True], ids=["text", "multimodal"])
@pytest.mark.parametrize(
    "case",
    [
        (False, "yarn", 10000.0, 2.0, 512, 0, None),
        (False, "deepseek_yarn", 10000, 2, 512, 0, "rope_parameters_hash"),
        (True, "deepseek_yarn", 10000, 2, 512, 0, None),
        (True, "yarn", 10000, 2, 512, 0, None),
        (True, "deepseek_yarn", 1000000, 2, 512, 0, "rope_parameters_hash"),
        (True, "deepseek_yarn", 10000, 4, 512, 0, "rope_parameters_hash"),
        (True, "deepseek_yarn", 10000, 2, 512, 1, "tp_rank"),
        (False, "yarn", 10000, 2, 1024, 0, "rope_parameters_hash"),
        (True, "deepseek_yarn", 10000, 2, 1024, 0, "rope_parameters_hash"),
    ],
)
def test_ipc_rope_matching(
    tmp_path, monkeypatch, default_vllm_config, fallback, nested, case
):
    """Reloads accept constructor-normalized RoPE without weakening other checks."""
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch
    from transformers import DeepseekV2Config, Qwen3VLConfig

    from vllm.config import LoadConfig, ModelConfig
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        CacheConfigMismatchError,
        WeightCacheKey,
        hash_rope_parameters,
    )

    is_reload, rope_type, theta, factor, max_position, rank, mismatch = case
    rope = dict(rope_type="yarn", rope_theta=10000, factor=2)
    config = (
        Qwen3VLConfig(text_config={"rope_parameters": rope})
        if nested
        else DeepseekV2Config(rope_parameters=rope)
    )
    config.architectures = [
        "Qwen3VLForConditionalGeneration" if nested else "DeepseekV2ForCausalLM"
    ]
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
    )
    model_config.hf_text_config.max_position_embeddings = 512
    server = object.__new__(daemon.WeightCacheDaemon)
    server.cache_key = WeightCacheKey.from_model_config(
        model_config, 1, 0, cache_config=default_vllm_config.cache_config
    )
    rope = model_config.hf_text_config.rope_parameters
    # DeepseekV2 construction normalizes yarn to deepseek_yarn.
    rope["rope_type"] = "deepseek_yarn"
    server.loaded_rope_parameters_hash = hash_rope_parameters(model_config)
    rope.update(rope_type=rope_type, rope_theta=theta, factor=factor)
    model_config.hf_text_config.max_position_embeddings = max_position
    server.model = torch.nn.Module()
    server.model.register_buffer("rotary_cache", torch.ones(2), persistent=False)
    server.role, server.global_rank = "target", 0
    for module in (daemon, ipc_loader):
        monkeypatch.setattr(module, "get_current_device_uuid", lambda: "test-device")
    for name in ("get_dp_group", "get_pp_group"):
        monkeypatch.setattr(
            ipc_loader, name, lambda: SimpleNamespace(world_size=1, rank_in_group=0)
        )
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    loader.fallback = fallback
    loader.state_timeout_s = 5
    model = torch.nn.Module()
    model.register_buffer("rotary_cache", torch.zeros(2), persistent=False)
    client, connection = socket.socketpair()
    with client, connection, ThreadPoolExecutor(max_workers=1) as executor:
        client.settimeout(5)
        connection.settimeout(5)
        monkeypatch.setattr(loader, "_connect", lambda timeout: client)
        server_task = executor.submit(server._handle_connection, connection)
        try:
            with (
                pytest.raises(CacheConfigMismatchError, match=mismatch)
                if mismatch
                else nullcontext()
            ):
                if is_reload:
                    loader.load_weights(model, model_config)
                else:
                    loader._fetch_entries(
                        model_config,
                        default_vllm_config.cache_config,
                        default_vllm_config.parallel_config,
                    )
            if is_reload:
                expected = torch.zeros(2) if mismatch else server.model.rotary_cache
                torch.testing.assert_close(model.rotary_cache, expected)
        finally:
            server_task.result(timeout=5)


def test_hash_checkpoint_distinguishes_weight_values(tmp_path):
    # Same layout, different values: a base checkpoint and its fine-tune must
    # not share a key, or an ipc_cache engine silently maps the daemon's stale
    # weights (#59647).
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        hash_checkpoint,
    )

    for name, value in (("base", 0.0), ("finetuned", 1.0)):
        (tmp_path / name).mkdir()
        save_file(
            {"w": torch.full((4, 4), value)},
            str(tmp_path / name / "model.safetensors"),
        )
    assert hash_checkpoint(str(tmp_path / "base")) != hash_checkpoint(
        str(tmp_path / "finetuned")
    )


def test_hash_checkpoint_stable_across_directory_copies(tmp_path):
    # The #54921 property: byte-identical checkpoints under different paths
    # share a key, so a daemon preloaded from one directory can serve an
    # engine started on the copy.
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        hash_checkpoint,
    )

    (tmp_path / "a").mkdir()
    save_file(
        {"w": torch.arange(4096, dtype=torch.float32).reshape(64, 64)},
        str(tmp_path / "a" / "model.safetensors"),
    )
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    assert hash_checkpoint(str(tmp_path / "a")) == hash_checkpoint(str(tmp_path / "b"))


def test_hash_checkpoint_catches_single_tensor_change_among_many(tmp_path):
    # Every tensor's head contributes to the key, so changing one tensor out
    # of 64 flips it. The sampled reads run through a thread pool, and 64
    # tensors exceed the worker count, so this also pins down that the digest
    # does not depend on read scheduling.
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        hash_checkpoint,
    )

    tensors = {f"w{i:03d}": torch.full((8, 8), float(i)) for i in range(64)}
    (tmp_path / "base").mkdir()
    save_file(tensors, str(tmp_path / "base" / "model.safetensors"))
    changed = dict(tensors)
    changed["w037"] = torch.full((8, 8), -1.0)
    (tmp_path / "changed").mkdir()
    save_file(changed, str(tmp_path / "changed" / "model.safetensors"))
    assert hash_checkpoint(str(tmp_path / "base")) != hash_checkpoint(
        str(tmp_path / "changed")
    )


def test_hash_checkpoint_without_safetensors_returns_none(tmp_path):
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        hash_checkpoint,
    )

    assert hash_checkpoint(str(tmp_path)) is None
    assert hash_checkpoint(str(tmp_path / "not-a-dir")) is None


def test_ipc_loader_copy_mode_reports_no_external_weight_memory():
    """Copy mode clones the weights into the engine, so nothing is external
    (vllm_config is never touched)."""
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import (
        IpcModelLoader,
    )

    copy_mode = LoadConfig(
        load_format="ipc_cache", model_loader_extra_config={"mode": "copy"}
    )
    loader = IpcModelLoader(copy_mode)
    assert loader.get_external_weight_memory(None) == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_daemon_reports_exported_weight_bytes():
    """The engine's budget excludes exactly the storages it maps from the
    daemon, not daemon-only allocations or tensors it does not export."""
    import socket

    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        recv_msg,
        send_msg,
    )

    def query_memory(server: daemon.WeightCacheDaemon) -> int:
        client, connection = socket.socketpair()
        with client, connection:
            send_msg(client, {"cmd": "get_memory"})
            server._handle_connection(connection)
            response = recv_msg(client)
        assert response["status"] == "ok"
        return response["memory_bytes"]

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        torch.empty(256, 1024, device="cuda"), requires_grad=False
    )
    # A tied parameter and a view share the weight's storage.
    model.lm_head = model.weight
    model.register_buffer("weight_row", model.weight[0], persistent=False)
    model.register_buffer("bias", torch.empty(1024, device="cuda"))
    # Shipped by value rather than mapped.
    model.register_buffer("cpu_table", torch.empty(1024))
    # Kept by the daemon but not exported (e.g. warmup workspace).
    model.scratch = torch.empty(1 << 20, device="cuda")
    leftover = torch.empty(1 << 20, device="cuda")

    server = object.__new__(daemon.WeightCacheDaemon)
    server.model = model
    assert query_memory(server) == (256 * 1024 + 1024) * 4
    server.model = None
    assert query_memory(server) == 0
    del leftover


@pytest.mark.parametrize(
    "cache_dtype,skip_layers,mismatched_fields",
    [
        ("auto", ["1", "2"], ["kv_cache_dtype"]),
        ("fp8", ["1"], ["kv_cache_dtype_skip_layers"]),
        ("fp8", ["2", "1", "2"], []),
    ],
)
def test_ipc_cache_key_tracks_kv_quantization(
    tmp_path, monkeypatch, cache_dtype, skip_layers, mismatched_fields
):
    """Initial loads and standalone reloads use the same KV compatibility key."""
    from types import SimpleNamespace

    import torch

    from vllm.config import CacheConfig, LoadConfig, ParallelConfig
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        WeightCacheKey,
        WeightCacheState,
    )

    model_config = SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        quantization="fp8",
        quantization_config=None,
        revision=None,
        hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
        hf_text_config=SimpleNamespace(),
        is_moe=False,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(),
        speculative_config=None,
        cache_config=CacheConfig(
            cache_dtype="fp8", kv_cache_dtype_skip_layers=["1", "2"]
        ),
    )
    server = daemon.WeightCacheDaemon(config, 0, 0, "unused")
    config.cache_config = CacheConfig(
        cache_dtype=cache_dtype, kv_cache_dtype_skip_layers=skip_layers
    )
    for name in ("get_dp_group", "get_pp_group"):
        monkeypatch.setattr(
            ipc_loader, name, lambda: SimpleNamespace(world_size=1, rank_in_group=0)
        )
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_rank", lambda: 0)
    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    requested_keys: list[WeightCacheKey] = []

    def request_state(key, *, is_reload=False):
        requested_keys.append(key)
        return WeightCacheState(entries={}, aliases={}, attrs={})

    monkeypatch.setattr(loader, "_request_state", request_state)
    monkeypatch.setattr(ipc_loader, "check_ipc_platform_support", lambda: None)
    monkeypatch.setattr(loader, "_build_model", lambda *args: torch.nn.Module())
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    model = loader.load_model(config, model_config)
    config.cache_config.cache_dtype = "bfloat16"
    config.cache_config.kv_cache_dtype_skip_layers.clear()
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.expert_placement_strategy = "round_robin"
    loader.load_weights(model, model_config)
    assert len(requested_keys) == 2
    for key in requested_keys:
        assert server.cache_key.mismatched_fields(key) == mismatched_fields
