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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest

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


@pytest.mark.parametrize(
    "case",
    [QWEN_CASE, K3_CASE, QWEN_MTP_CASE],
    ids=["qwen3.5", "kimi-k3", "qwen3.5-mtp"],
)
def test_ipc_cache_cold_start_and_warm_restart(vllm_runner, case: ModelCase):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon — for the MTP case, both the target's and the
    draft's daemon groups.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("Weight cache IPC sharing requires CUDA or ROCm")
    if case is K3_CASE and not current_platform.is_device_capability_family(100):
        pytest.skip("Kimi K3 IPC weight cache requires an SM100 MXFP4 backend")

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

    with WeightCacheDaemon(
        case.model,
        tp_size=1,
        extra_args=case.daemon_args,
    ) as d:
        warm_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        d.check_health()
        # Warm restart: a second engine lifetime against the same daemon.
        restart_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)

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

    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey
    from vllm.utils.hashing import safe_hash

    model_config = SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        quantization="modelopt_fp4",
        quantization_config=None,
        revision=None,
        hf_config=SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
            quantization_config=hf_quant_config,
        ),
    )
    static_key = WeightCacheKey.from_model_config(model_config, tp_size=1, tp_rank=0)
    model_config.quantization_config = QuantizationConfigArgs(
        moe={"activation": "nvfp4_per_token"}
    )
    per_token_key = WeightCacheKey.from_model_config(model_config, tp_size=1, tp_rank=0)
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
