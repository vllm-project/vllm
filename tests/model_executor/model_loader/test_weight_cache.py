# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import contextlib
import shutil
import socket
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
def test_ipc_cache_cold_start_and_warm_restart_tp2_ep(vllm_runner, tmp_path, capfd):
    """Same as above, with expert parallelism enabled at tp=2."""
    run_cold_start_and_warm_restart(vllm_runner, DSV2_EP_CASE, tmp_path, capfd)


def run_cold_start_and_warm_restart(vllm_runner, case: ModelCase, tmp_path, capfd):
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


def _mla_like_tensors():
    """kv_b_proj plus the transposed W_UV / permuted W_UK_T views that MLA's
    post-processing leaves on the same storage."""
    import torch

    rank, heads, nope, v = 16, 2, 4, 4
    kv_b = (
        torch.arange(heads * (nope + v) * rank, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(heads * (nope + v), rank)
    )
    w_uk, w_uv = kv_b.T.view(rank, heads, nope + v).split([nope, v], dim=-1)
    tensors = {
        "kv_b_proj.weight": kv_b,
        "W_UV": w_uv.transpose(0, 1),
        "W_UK_T": w_uk.permute(1, 2, 0),
        "scale": torch.ones(3, dtype=torch.float32),
    }
    kinds = {name: "param" for name in tensors} | {"scale": "buffer"}
    return tensors, kinds


class _FakeMooncake:
    """TransferEngine stand-in that "reads" host memory with memmove."""

    def __init__(self):
        self.registered: list[int] = []
        self.unregistered: list[int] = []

    def batch_register_memory(self, pointers, lengths):
        if set(pointers) & set(self.registered):
            return -1  # Mooncake refuses to register memory twice
        self.registered += pointers
        return 0

    def batch_unregister_memory(self, pointers):
        self.unregistered += pointers
        self.registered = [p for p in self.registered if p not in pointers]
        return 0

    def batch_transfer_sync_read(self, session, local, remote, lengths):
        import ctypes

        for dst, src, nbytes in zip(local, remote, lengths):
            ctypes.memmove(dst, src, nbytes)
        return 0


def _fake_rdma_source(engine):
    from vllm.model_executor.model_loader.weight_cache.seed import RdmaSeedSource

    source = RdmaSeedSource()
    source._initialize_engine = lambda: (engine, "127.0.0.1:1")
    return source


def test_seed_manifest_describes_tensors_as_views_over_storages():
    """Views that share an allocation must collapse onto one storage entry
    with their own offset and strides, or the mirror could neither size its
    buffers nor rebuild the layout."""
    import torch

    from vllm.model_executor.model_loader.weight_cache.seed import (
        build_manifest,
        manifest_dtype,
        manifest_nbytes,
    )

    tensors, kinds = _mla_like_tensors()
    manifest, storages = build_manifest(tensors, kinds)
    assert len(storages) == len(manifest["storages"]) == 2
    views = manifest["tensors"]
    assert {views[n]["storage"] for n in ("kv_b_proj.weight", "W_UV", "W_UK_T")} == {0}
    assert views["W_UV"]["offset"] == tensors["W_UV"].storage_offset()
    assert views["W_UV"]["stride"] == list(tensors["W_UV"].stride())
    assert views["scale"] == {
        "storage": 1,
        "offset": 0,
        "shape": [3],
        "stride": [1],
        "dtype": "float32",
        "is_param": False,
    }
    assert manifest_nbytes(manifest) == tensors["kv_b_proj.weight"].nbytes + 3 * 4
    assert manifest_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        manifest_dtype("not_a_torch_dtype")


@pytest.mark.parametrize("backend", ["peer_ipc", "rdma"])
def test_seed_rebuilds_strided_views_with_their_aliasing(backend):
    """MLA's W_UV/W_UK_T are non-contiguous views of kv_b_proj. A mirror must
    rebuild them with the source's values, strides and shared storage; the
    rdma plane carries the manifest and seed as JSON."""
    import json
    from unittest.mock import patch

    import torch

    from vllm.model_executor.model_loader.weight_cache.seed import (
        PeerIpcSeedSource,
        build_manifest,
    )

    tensors, kinds = _mla_like_tensors()
    manifest, storages = build_manifest(tensors, kinds)
    if backend == "rdma":
        source = mirror = _fake_rdma_source(_FakeMooncake())
        seed = source.prepare_seed(storages, None)
        manifest, seed = json.loads(json.dumps([manifest, seed]))
    else:
        source = mirror = PeerIpcSeedSource()
        with patch.object(torch.accelerator, "current_device_index", return_value=0):
            seed = source.prepare_seed(storages, None)
    with patch.object(torch.accelerator, "synchronize"):
        result = mirror.fill(manifest, seed, torch.device("cpu"))

    for name, tensor in tensors.items():
        assert torch.equal(result[name], tensor), name
        assert result[name].stride() == tensor.stride(), name
        assert result[name].data_ptr() != tensor.data_ptr(), name
    kv_b_storage = result["kv_b_proj.weight"].untyped_storage().data_ptr()
    assert result["W_UV"].untyped_storage().data_ptr() == kv_b_storage
    assert result["W_UK_T"].untyped_storage().data_ptr() == kv_b_storage


def test_seed_rejects_a_view_that_overruns_its_storage():
    """``set_`` would silently grow a too-small buffer, detaching it from the
    other views on that storage, so a peer's manifest is bounds-checked."""
    import torch

    from vllm.model_executor.model_loader.weight_cache.seed import (
        build_manifest,
        rebuild_tensors,
    )

    manifest, _ = build_manifest(
        {"weight": torch.zeros(4, dtype=torch.float32)}, {"weight": "param"}
    )
    manifest["tensors"]["weight"]["shape"] = [8]
    with pytest.raises(RuntimeError, match="overruns its storage"):
        rebuild_tensors(manifest, [torch.empty(16, dtype=torch.uint8)])


def test_rdma_seed_registers_each_storage_once():
    """Mooncake refuses to register memory twice, so a source that registered
    per request could seed only one mirror; releasing the weights must hand
    the memory back to the NIC."""
    from vllm.model_executor.model_loader.weight_cache.seed import build_manifest

    tensors, kinds = _mla_like_tensors()
    _, storages = build_manifest(tensors, kinds)
    engine = _FakeMooncake()
    source = _fake_rdma_source(engine)
    first = source.prepare_seed(storages, None)
    second = source.prepare_seed(storages, None)
    assert first["regions"] == second["regions"]
    assert sorted(engine.registered) == sorted(s.data_ptr() for s in storages)
    source.release()
    assert engine.registered == []


def test_peer_seed_rejects_a_remapped_source_device():
    """A CUDA IPC handle names its device by index, so a mirror that sees a
    different physical GPU at that index must refuse rather than open a
    handle on the wrong device."""
    from unittest.mock import patch

    import torch

    from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry
    from vllm.model_executor.model_loader.weight_cache.seed import (
        PeerIpcSeedSource,
        build_manifest,
    )

    manifest, storages = build_manifest({"weight": torch.zeros(2)}, {"weight": "param"})
    seed = {
        "source_device_index": 0,
        "source_gpu_uuid": "GPU-source",
        "storages": [
            TensorEntry.from_tensor(torch.zeros(8, dtype=torch.uint8), "buffer")
        ],
    }
    with patch(
        "vllm.model_executor.model_loader.weight_cache.seed.current_platform"
    ) as platform:
        platform.get_device_uuid.return_value = "GPU-somewhere-else"
        with pytest.raises(RuntimeError, match="different physical GPU"):
            PeerIpcSeedSource().fill(manifest, seed, torch.device("cpu"))


@contextlib.contextmanager
def _remote_seed_daemon(token):
    """Serve the daemon's remote seed plane on a loopback port.

    Only that thread is exercised, so __init__ (which loads a model onto a
    GPU) is skipped and just the state it touches is provided.
    """
    from unittest.mock import patch

    import torch

    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon

    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.seed_token = token
    daemon.model = None
    daemon.mirror = None
    daemon.role = "target"
    daemon.global_rank = 0
    daemon.device_index = 0
    daemon._state_lock = threading.Lock()

    server = socket.create_server(("127.0.0.1", 0))
    with patch.object(torch.accelerator, "set_device_index"):
        thread = threading.Thread(
            target=daemon._serve_remote, args=(server,), daemon=True
        )
        thread.start()
        try:
            yield server.getsockname()[1]
        finally:
            # close() alone does not wake an accept() blocked on another thread.
            with contextlib.suppress(OSError):
                server.shutdown(socket.SHUT_RDWR)
            server.close()
            thread.join(timeout=5)


def _ask_remote(port, message):
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        recv_json,
        send_json,
    )

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        send_json(sock, message)
        return recv_json(sock)


def test_remote_seed_plane_requires_the_shared_token():
    """The listener authenticates before serving anything, and it speaks JSON
    rather than pickle so an unauthenticated peer's bytes are never
    deserialized into objects."""
    with _remote_seed_daemon("s3cret") as port:
        assert _ask_remote(port, {"cmd": "fetch_manifest"})["status"] == "error"
        wrong = _ask_remote(port, {"cmd": "fetch_manifest", "token": "guess"})
        assert wrong["status"] == "error"
        # Authenticated, the request is dispatched; this daemon holds no
        # weights, so the handler itself refuses.
        served = _ask_remote(port, {"cmd": "fetch_manifest", "token": "s3cret"})
        assert served["status"] == "error"
        assert "released" in served["message"]


def test_remote_seed_plane_refuses_local_only_commands():
    """Exporting IPC handles and releasing weights stay on the
    owner-verified Unix socket even for an authenticated peer; commands the
    daemon no longer serves, like the artifact cache's fetch_artifacts, get
    the same refusal."""
    with _remote_seed_daemon("s3cret") as port:
        for cmd in (
            "get_state",
            "release",
            "put_artifact",
            "get_memory",
            "fetch_artifacts",
        ):
            response = _ask_remote(port, {"cmd": cmd, "token": "s3cret"})
            assert response["status"] == "error"
            assert "served locally" in response["message"]


def test_remote_seed_plane_drops_a_stalled_peer(monkeypatch):
    """A peer that connects and never sends its request must not hold the
    listener: it gets a deadline, and the next peer is still served."""
    from vllm.model_executor.model_loader.weight_cache import daemon

    monkeypatch.setattr(daemon, "_REMOTE_REQUEST_TIMEOUT_S", 0.5)
    with (
        _remote_seed_daemon("s3cret") as port,
        socket.create_connection(("127.0.0.1", port)),
    ):
        served = _ask_remote(port, {"cmd": "fetch_manifest", "token": "s3cret"})
    # No weights here, but the stalled peer did not hold up the listener.
    assert served["status"] == "error"
    assert "released" in served["message"]


def _bare_daemon(tmp_path, monkeypatch):
    """A daemon skipped past __init__'s model load, with its per-rank
    autotune cache path and cache root pointed under tmp_path."""
    from vllm.model_executor.model_loader.weight_cache import daemon as daemon_module
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon

    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.role = "target"
    daemon.global_rank = 0
    monkeypatch.setattr(
        daemon_module, "resolve_flashinfer_autotune_root", lambda: tmp_path
    )
    daemon._autotune_cache_path = (
        lambda: tmp_path / "hash0" / "autotune_configs_dp0_rank0.json"
    )
    return daemon


def test_read_autotune_cache_is_none_without_a_file(tmp_path, monkeypatch):
    """A daemon that never wrote a table (autotune disabled, or the
    single-GPU managed store) seeds None, and the mirror's first engine
    tunes for itself, as without seeding."""
    assert _bare_daemon(tmp_path, monkeypatch)._read_autotune_cache() is None


def test_read_autotune_cache_encodes_this_ranks_file(tmp_path, monkeypatch):
    """The tuned table rides base64-encoded inside the seed response, with
    its cache-root-relative location: the mirror cannot recompute the
    directory, because warmup finalizes the config the hash is keyed on and
    a mirror never warms up."""
    import pybase64 as base64

    daemon = _bare_daemon(tmp_path, monkeypatch)
    path = daemon._autotune_cache_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"op": 1}')
    payload = daemon._read_autotune_cache()
    assert payload is not None
    assert payload["rel"] == "hash0/autotune_configs_dp0_rank0.json"
    assert base64.b64decode(payload["data"]) == b'{"op": 1}'


def test_read_autotune_cache_drops_an_over_budget_file(tmp_path, monkeypatch, caplog):
    """The table shares the seed response's size budget, so an oversized file
    is skipped with a warning rather than failing the seed."""
    import logging

    from vllm.model_executor.model_loader.weight_cache import daemon as daemon_module

    monkeypatch.setattr(daemon_module, "_MAX_SEED_AUTOTUNE_BYTES", 4)
    daemon = _bare_daemon(tmp_path, monkeypatch)
    path = daemon._autotune_cache_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 8)
    with caplog.at_level(logging.WARNING):
        assert daemon._read_autotune_cache() is None
    assert "seed budget" in caplog.text


def test_adopt_seeded_autotune_cache_writes_the_decoded_file(tmp_path, monkeypatch):
    """The mirror lands the source's table at the source's root-relative
    location under its own cache root, where its engines hit it through the
    ordinary autotune cache lookup."""
    import pybase64 as base64

    daemon = _bare_daemon(tmp_path, monkeypatch)
    daemon._adopt_seeded_autotune_cache(
        {
            "rel": "hash0/autotune_configs_dp0_rank0.json",
            "data": base64.b64encode(b"tuned").decode(),
        }
    )
    assert (tmp_path / "hash0" / "autotune_configs_dp0_rank0.json").read_bytes() == (
        b"tuned"
    )


def test_adopt_seeded_autotune_cache_ignores_a_missing_payload(tmp_path, monkeypatch):
    """A source without a table seeds None; the seed must not fail and no
    file is written."""
    daemon = _bare_daemon(tmp_path, monkeypatch)
    daemon._adopt_seeded_autotune_cache(None)
    assert not list(tmp_path.rglob("autotune_configs*"))


def test_adopt_seeded_autotune_cache_rejects_a_malformed_location(
    tmp_path, monkeypatch, caplog
):
    """The source's cache location is authenticated but still validated
    before it touches the filesystem."""
    import logging

    import pybase64 as base64

    daemon = _bare_daemon(tmp_path, monkeypatch)
    data = base64.b64encode(b"x").decode()
    with caplog.at_level(logging.WARNING):
        for rel in (
            "../escape/autotune_configs.json",
            "/abs/path/autotune_configs.json",
            "autotune_configs.json",
            "hash0/not_the_table.json",
        ):
            daemon._adopt_seeded_autotune_cache({"rel": rel, "data": data})
    assert "malformed" in caplog.text
    assert not list(tmp_path.rglob("*.json"))


def test_adopt_seeded_autotune_cache_ignores_an_unreadable_payload(
    tmp_path, monkeypatch, caplog
):
    """Garbage from a peer costs the mirror's engines a full autotune pass;
    it must not fail the seed nor write a file."""
    import logging

    daemon = _bare_daemon(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        daemon._adopt_seeded_autotune_cache(
            {"rel": "hash0/autotune_configs_dp0_rank0.json", "data": "not b64!!!"}
        )
    assert "unreadable" in caplog.text
    assert not list(tmp_path.rglob("autotune_configs*"))


@contextlib.contextmanager
def _seed_daemon(tmp_path, monkeypatch):
    """Serve the daemon's local fetch_manifest handler on a throwaway socket.

    Only that handler is exercised, so __init__ (which loads a model onto a
    GPU) is skipped and just the state it touches is provided.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon as daemon_module
    from vllm.model_executor.model_loader.weight_cache.daemon import MirrorState

    daemon = _bare_daemon(tmp_path, monkeypatch)
    daemon.model = None
    daemon.mirror = MirrorState(
        tensors={"weight": torch.zeros(2)}, kinds={"weight": "param"}
    )
    daemon.model_config = SimpleNamespace(model="m")
    daemon.cache_config = None
    daemon._seed_sources = {}

    socket_path = str(tmp_path / "daemon.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen()

    def _serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                daemon._handle_connection(conn)

    with (
        patch.object(torch.accelerator, "current_device_index", return_value=0),
        patch.object(daemon_module, "get_current_device_uuid", return_value=None),
    ):
        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        try:
            yield daemon, socket_path
        finally:
            server.close()
            thread.join(timeout=5)


@pytest.mark.parametrize("contents", [None, b"tuned"])
def test_fetch_manifest_response_carries_the_autotune_cache(
    tmp_path, monkeypatch, contents
):
    """Every seed response carries the source rank's on-disk autotune cache
    (or None when it never wrote one), so a seeded mirror's engines skip the
    autotune pass."""
    import pybase64 as base64

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        recv_msg,
        send_msg,
    )

    with _seed_daemon(tmp_path, monkeypatch) as (daemon, socket_path):
        if contents is not None:
            path = daemon._autotune_cache_path()
            path.parent.mkdir(parents=True)
            path.write_bytes(contents)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(socket_path)
            send_msg(sock, {"cmd": "fetch_manifest"})
            response = recv_msg(sock)
    assert response["status"] == "ok"
    payload = response["autotune_cache"]
    if contents is None:
        assert payload is None
    else:
        assert payload["rel"] == "hash0/autotune_configs_dp0_rank0.json"
        assert base64.b64decode(payload["data"]) == contents


def test_draft_seed_dials_the_draft_listeners():
    """One host:base_port serves both daemon groups, so a draft mirror rank
    must dial the source's draft listener, past the target group's block,
    rather than the target daemon of the same rank."""
    from vllm.entrypoints.cli.preload import _listen_address, _seed_address

    world_size, rank = 2, 1
    _, draft_port = _listen_address("0.0.0.0:29700", rank, port_offset=world_size)
    dialed = _seed_address("10.0.0.1:29700", rank, rank, port_offset=world_size)
    assert dialed == f"10.0.0.1:{draft_port}"
    assert _seed_address("10.0.0.1:29700", rank, rank) == "10.0.0.1:29701"


def test_seed_addresses_tell_socket_paths_from_host_ports():
    """A relative socket path must stay local, and only a real host:port is
    dialed over the network."""
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        parse_seed_address,
    )

    for path in ("run/sock0", "../sock0", "sock0"):
        assert parse_seed_address(path) == path
    assert parse_seed_address("10.0.0.1:29700") == ("10.0.0.1", 29700)
    assert parse_seed_address("[fe80::1]:29700") == ("fe80::1", 29700)
    with pytest.raises(ValueError, match="host:port"):
        parse_seed_address("host:29x00")


def test_seed_matches_an_unhashed_mirror_by_model_path():
    """A mirror without the safetensors fingerprints its checkpoint by model
    path; it must accept a source serving that same --model and still refuse
    one serving another."""
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache.daemon import seed_mismatches
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    source = WeightCacheKey(
        checkpoint="3f2a9c",
        model_arch="KimiK3ForConditionalGeneration",
        tp_size=1,
        tp_rank=0,
        dtype="torch.bfloat16",
        quantization=None,
        quant_config_hash="",
        revision=None,
        vllm_version="0",
    )
    mirror = replace(source, checkpoint="/models/k3")
    assert seed_mismatches(mirror, source, "/models/k3", "/models/k3") == []
    assert seed_mismatches(mirror, source, "/models/k3", "/models/x") == ["checkpoint"]
    hashed_elsewhere = replace(source, checkpoint="77b1e0")
    assert seed_mismatches(hashed_elsewhere, source, "/models/k3", "/models/k3") == [
        "checkpoint"
    ]
