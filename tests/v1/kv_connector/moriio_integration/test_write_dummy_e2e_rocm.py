# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node MoRIIO P/D checks on dummy weights, in WRITE and READ mode.

Each case serves a real model architecture with ``--load-format dummy``, widened
by ``moriio_wide_dummy`` (see ``DUMMY_WEIGHT_RANGE``): the
config keeps the real hidden size, head geometry, MLA ranks and recurrent-state
shapes, so the connector registers and transfers exactly the production cache
layout, but the layer stack is cut to a few layers that still contain every
cache-group kind and the MoE is cut to a handful of experts. Dummy weights are
seeded, so the prefill, decode and reference servers hold identical weights.
Cases marked ``HMA_off`` run every server with
``--disable-hybrid-kv-cache-manager``, so all layers, sliding-window ones
included, share a single cache group.

Prefill runs on one GPU, decode on another, and MoRI moves the KV over xGMI:
prefill pushes it in WRITE mode, and decode pulls it in READ mode (cases
marked ``READ``). The proxy runs a READ request's prefill to completion before
it sends the request to decode.
Either way decode computes the last prompt token itself. For hybrid models
prefill computes only the first ``n - 1`` tokens and transfers the KV and the
recurrent state after them. For attention-only models prefill computes and
transfers all ``n``, and decode's scheduler, seeing the whole prompt cached,
recomputes the last token. A direct serve splits the same prompt the same way
when the first ``n - 1`` tokens end on a chunk boundary that both servers share,
so the prompts are chosen to make that happen: one is a single chunk plus one
token, the others are a whole number of hybrid cache blocks plus one token
(prefix caching ends chunks on block boundaries) or, for attention-only models,
a whole number of chunks plus one token. With the split matched, a correct
transfer reproduces the direct serve's first-token log-probabilities exactly,
so the comparison needs no noise allowance. The
servers compile with Inductor's deterministic mode and without pointwise
autotuning, since otherwise each one benchmarks its own launch configs and
separately started servers can drift apart by a few bf16 logit steps. Guards:

* decode must report an external prefix-cache hit for every transferred token
  of every prompt, so the match is not a local recompute;
* changing the first half of a prompt must move the direct serve's log-probs by
  at least ``MIN_CONTEXT_EFFECT``, so a wrong prefix would be visible;
* ``test_corrupted_prefill_state_is_detected`` makes prefill compute its KV with
  a different norm epsilon (same shapes, different values) and requires every
  prompt to fail the comparison.
"""

import atexit
import contextlib
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

_ROOT = Path(__file__).resolve().parents[4]
_PROXY = (
    _ROOT / "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py"
)

# Max allowed difference between centered first-token log-probabilities of the
# P/D and direct outputs, over the direct serve's top-k tokens. The split is
# matched, so correct transfers produce identical logits; this only absorbs the
# fp32 rounding of the log-softmax (~1e-7), far below one bf16 logit step.
TOLERANCE = float(os.getenv("MORIIO_DUMMY_E2E_TOL", "1e-5"))
# Several bf16 steps of the logits at DUMMY_WEIGHT_RANGE.
MIN_CONTEXT_EFFECT = 0.05
# Half-width of the uniform dummy init for 16/32-bit float weights, applied in
# every server worker by moriio_wide_dummy.py. The loader default (1e-3) shrinks
# every activation so far that a context change moves the first-token log-probs
# by ~1e-4; at 0.1 it moves them by >0.1 and a corrupted prefill changes the
# top-k set entirely.
DUMMY_WEIGHT_RANGE = 0.1
_WIDE_DUMMY_DIR = Path(__file__).resolve().parent
# MoRI transport between the single-node prefill and decode GPUs.
BACKEND = os.getenv("MORIIO_DUMMY_E2E_BACKEND", "xgmi")
TOP_K = 20
# Well under the 60 s deferred-send reaper, which frees unreleased blocks.
PREFILL_FREE_TIMEOUT_S = 20
MAX_TOKENS = 4
MAX_MODEL_LEN = 4096

# Prefill token budget of every server. Prompts longer than it are chunked, so
# the final-chunk save path runs; the short prompt fills exactly one chunk.
PREFILL_CHUNK = 299
SHORT_PROMPT_LEN = PREFILL_CHUNK + 1
# Hybrid blocks per aligned prompt, before the trailing token.
ALIGNED_BLOCKS = (1, 2)
# Prefill chunks per long attention-only prompt, before the trailing token.
ALIGNED_CHUNKS = (2, 4)


def _mori_available() -> bool:
    try:
        import mori.io  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm() or not _mori_available(),
        reason="MoRIIO P/D needs ROCm and MoRI",
    ),
    pytest.mark.skipif(
        current_platform.is_rocm() and current_platform.device_count() < 2,
        reason="needs one GPU for prefill and one for decode",
    ),
]


# ---------------------------------------------------------------------------
# Model architectures


def _kimi_linear_overrides(cfg) -> dict[str, Any]:
    # 3 KDA layers then 1 MLA layer; layer 0 stays dense (first_k_dense_replace).
    linear = dict(cfg.linear_attn_config)
    linear["kda_layers"] = [1, 2, 3]
    linear["full_attn_layers"] = [4]
    groups = max(1, cfg.num_expert_group)
    return {
        "num_hidden_layers": 4,
        "linear_attn_config": linear,
        "num_experts": 2 * groups,
        "num_experts_per_token": 2,
        "topk_group": min(cfg.topk_group, groups),
    }


def _qwen3_5_moe_overrides(cfg) -> dict[str, Any]:
    # 3 Gated DeltaNet layers then 1 full-attention layer.
    return {
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
    }


def _qwen3_dense_overrides(cfg) -> dict[str, Any]:
    # Attention-only: decode gets the KV for the whole prompt.
    return {"num_hidden_layers": 4}


def _deepseek_v3_overrides(cfg) -> dict[str, Any]:
    # MLA attention-only: layer 0 dense, layer 1 MoE, no MTP layer.
    return {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 0,
    }


def _gemma3_overrides(cfg) -> dict[str, Any]:
    # Five sliding-window layers then one global layer, as in the real
    # 5:1 pattern. Without the hybrid KV cache manager vLLM gives the sliding
    # layers full-attention cache specs, so all six share one group.
    return {
        "num_hidden_layers": 6,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    }


@dataclass(frozen=True)
class ArchCase:
    model: str
    overrides: Callable[[Any], dict[str, Any]]
    # Norm-epsilon override that perturbs every computed KV/state value
    # without changing a single shape. Used by the corruption control.
    corrupt_overrides: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    # Has recurrent-state (mamba-style) layers, so prefill stops at n - 1.
    hybrid: bool = True


ARCHS: dict[str, ArchCase] = {
    "kimi_linear": ArchCase(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        overrides=_kimi_linear_overrides,
        corrupt_overrides={"rms_norm_eps": 1.0},
        env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--trust-remote-code", "--mamba-ssm-cache-dtype", "float32"),
    ),
    "qwen3_5_moe": ArchCase(
        model="Qwen/Qwen3.5-35B-A3B",
        overrides=_qwen3_5_moe_overrides,
        corrupt_overrides={"text_config": {"rms_norm_eps": 1.0}},
        # AITER's GDN conv-update kernel rejects the DS conv-state layout that
        # MoRIIO's recurrent-state transfer requires.
        env={"VLLM_ROCM_USE_AITER": "0"},
        args=("--language-model-only",),
    ),
    "qwen3_dense": ArchCase(
        model="Qwen/Qwen3-32B",
        overrides=_qwen3_dense_overrides,
        corrupt_overrides={"rms_norm_eps": 1.0},
        env={"VLLM_ROCM_USE_AITER": "0"},
        hybrid=False,
    ),
    "deepseek_v3": ArchCase(
        model="deepseek-ai/DeepSeek-V3",
        overrides=_deepseek_v3_overrides,
        corrupt_overrides={"rms_norm_eps": 1.0},
        env={"VLLM_ROCM_USE_AITER": "1"},
        hybrid=False,
    ),
    # Text-only Gemma 3. The multimodal Gemma3ForConditionalGeneration takes
    # the inputs_embeds path on KV consumers (enable_mm_embeds is forced on
    # there) and computes differently from a direct serve without any transfer.
    "gemma3": ArchCase(
        model="google/gemma-3-1b-it",
        overrides=_gemma3_overrides,
        corrupt_overrides={"rms_norm_eps": 1.0},
        env={"VLLM_ROCM_USE_AITER": "0"},
        hybrid=False,
    ),
}


def _arch_overrides(arch: ArchCase) -> dict[str, Any]:
    from vllm.transformers_utils.config import get_config

    return arch.overrides(get_config(arch.model, trust_remote_code=True))


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Serving


@dataclass(frozen=True)
class Case:
    arch: str
    attention_backend: str
    prefill_graph: str
    decode_graph: str
    # Run every server of the case without the hybrid KV cache manager.
    hma_off: bool = False
    prefill_tp: int = 1
    decode_tp: int = 1
    # Extra arguments for every server of the case.
    server_args: tuple[str, ...] = ()
    # Decode pulls the KV (READ) instead of prefill pushing it (WRITE).
    read_mode: bool = False

    @property
    def extra_args(self) -> tuple[str, ...]:
        hma = ("--disable-hybrid-kv-cache-manager",) if self.hma_off else ()
        return hma + self.server_args

    @property
    def heterogeneous_tp(self) -> bool:
        return self.prefill_tp != self.decode_tp

    @property
    def id(self) -> str:
        return (
            f"{self.arch}-{self.attention_backend}-"
            f"P_{self.prefill_graph}-D_{self.decode_graph}"
            + ("-HMA_off" if self.hma_off else "")
            + (
                f"-TP{self.prefill_tp}to{self.decode_tp}"
                if self.heterogeneous_tp
                else ""
            )
            + ("-READ" if self.read_mode else "")
        )


_DEFAULT = "FULL_AND_PIECEWISE"
CASES = [
    # Kimi-Linear: KDA + MLA groups.
    Case("kimi_linear", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT),
    Case("kimi_linear", "ROCM_AITER_MLA", "NONE", "NONE"),
    Case("kimi_linear", "ROCM_AITER_MLA", "PIECEWISE", "PIECEWISE"),
    Case("kimi_linear", "ROCM_AITER_MLA", _DEFAULT, "FULL_DECODE_ONLY"),
    Case("kimi_linear", "TRITON_MLA", _DEFAULT, _DEFAULT),
    # Qwen3.5: Gated DeltaNet + full-attention groups.
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT),
    Case("qwen3_5_moe", "TRITON_ATTN", "NONE", "NONE"),
    Case("qwen3_5_moe", "TRITON_ATTN", "PIECEWISE", "PIECEWISE"),
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, "FULL_DECODE_ONLY"),
    # ROCM_ATTN keeps a (2, num_blocks, ...) KV layout and opts out of KV
    # connectors, so the second ROCm attention backend is the unified one.
    Case("qwen3_5_moe", "ROCM_AITER_UNIFIED_ATTN", _DEFAULT, _DEFAULT),
    # Attention-only: prefill transfers the whole prompt.
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT),
    Case("qwen3_dense", "ROCM_AITER_UNIFIED_ATTN", _DEFAULT, _DEFAULT),
    # Without the hybrid KV cache manager: one cache group per model.
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, hma_off=True),
    Case("deepseek_v3", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT, hma_off=True),
    Case("gemma3", "TRITON_ATTN", _DEFAULT, _DEFAULT, hma_off=True),
]


# Prefill TP2 writing into decode TP1: split GQA heads gathered into one
# block, MLA latents and replicated heads written by one prefill rank, and
# recurrent state gathered from both. Decode TP larger than prefill TP is not
# here: on one node MoRI's xGMI backend cannot open a session to a peer's second
# GPU from a prefill rank that sees fewer GPUs (see the MoRIIO usage guide).
HETEROGENEOUS_TP_CASES = [
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, prefill_tp=2),
    Case("qwen3_dense", "ROCM_AITER_UNIFIED_ATTN", _DEFAULT, _DEFAULT, prefill_tp=2),
    Case("deepseek_v3", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT, True, prefill_tp=2),
    Case("gemma3", "TRITON_ATTN", _DEFAULT, _DEFAULT, True, prefill_tp=2),
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT, prefill_tp=2),
    # Hybrid models size attention blocks to the per-rank recurrent state, and
    # an MLA page does not shrink with TP like KDA state does: pin one block
    # size on both sides (TP2 alone picks 1024 tokens, TP1 1920).
    Case(
        "kimi_linear",
        "ROCM_AITER_MLA",
        _DEFAULT,
        _DEFAULT,
        prefill_tp=2,
        server_args=("--block-size", "2048"),
    ),
]
# READ mode, equal TP: one case per cache-group kind (KDA + MLA, Gated DeltaNet
# + full attention, full attention, and single-group MLA and sliding window).
READ_CASES = [
    Case("kimi_linear", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT, read_mode=True),
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT, read_mode=True),
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, read_mode=True),
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, True, read_mode=True),
    Case("deepseek_v3", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT, True, read_mode=True),
    Case("gemma3", "TRITON_ATTN", _DEFAULT, _DEFAULT, True, read_mode=True),
]
# READ mode, decode TP2 pulling from prefill TP1 (each decode rank reads its
# heads and state slice) and decode TP1 pulling from prefill TP2. A fan-in read
# of split heads or recurrent state needs a session to the second prefill GPU,
# which MoRI's xGMI backend cannot open from a decode rank that sees one GPU;
# those cases only run over RDMA (MORIIO_DUMMY_E2E_BACKEND=rdma).
HETEROGENEOUS_TP_READ_CASES = [
    Case("qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, decode_tp=2, read_mode=True),
    Case(
        "deepseek_v3",
        "ROCM_AITER_MLA",
        _DEFAULT,
        _DEFAULT,
        True,
        decode_tp=2,
        read_mode=True,
    ),
    Case(
        "gemma3", "TRITON_ATTN", _DEFAULT, _DEFAULT, True, decode_tp=2, read_mode=True
    ),
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT, decode_tp=2, read_mode=True),
    # MLA fan-in reads the replicated latent from one prefill rank and
    # releases both.
    Case(
        "deepseek_v3",
        "ROCM_AITER_MLA",
        _DEFAULT,
        _DEFAULT,
        True,
        prefill_tp=2,
        read_mode=True,
    ),
    Case(
        "qwen3_dense", "TRITON_ATTN", _DEFAULT, _DEFAULT, prefill_tp=2, read_mode=True
    ),
    Case(
        "qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT, prefill_tp=2, read_mode=True
    ),
]


def _reads_every_prefill_gpu(case: Case) -> bool:
    """A READ fan-in that must reach a prefill GPU beyond the first."""
    return (
        case.read_mode
        and case.prefill_tp > case.decode_tp
        and (ARCHS[case.arch].hybrid or case.arch != "deepseek_v3")
    )


def _graph_args(mode: str) -> list[str]:
    if mode == "NONE":
        return ["--enforce-eager"]
    # Inductor otherwise picks launch configs by timing them, so separately
    # compiled servers may drift by a few bf16 logit steps: reductions sum in a
    # different order, and on ROCm the pointwise block size changes which
    # multiply-adds are fused into FMAs. Deterministic mode fixes only the
    # reduction configs, so pointwise autotuning is off as well.
    config = {
        "cudagraph_mode": mode,
        "inductor_compile_config": {
            "deterministic": True,
            "triton.autotune_pointwise": False,
        },
    }
    return ["--compilation-config", json.dumps(config)]


def _visible_devices() -> tuple[str, list[str]]:
    selectors = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    selector = next((k for k in selectors if k in os.environ), selectors[0])
    visible = os.getenv(selector)
    devices = (
        visible.split(",")
        if visible
        else [str(i) for i in range(current_platform.device_count())]
    )
    return selector, devices


def _device_env(selector: str, devices: list[str]) -> dict[str, str]:
    # vLLM mirrors HIP_VISIBLE_DEVICES into CUDA_VISIBLE_DEVICES at import, so
    # the test process carries both; pin every selector or the server refuses
    # the mismatch. ROCR narrows the set first, so HIP/CUDA then index into it.
    env = {selector: ",".join(devices)}
    indexed = ",".join(str(i) for i in range(len(devices)))
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if key != selector:
            env[key] = indexed if selector == "ROCR_VISIBLE_DEVICES" else env[selector]
    return env


# KV consumers force enable_mm_embeds on, which moves multimodal-capable models
# (Qwen3.5) from input_ids to inputs_embeds, but the AOT compile-cache key does
# not cover it, so a consumer can load a producer's graph and crash in
# profile_run. Keep each role's compile cache apart until
# vllm-project/vllm#54642 lands (vllm-project/vllm#50891).
_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="moriio_dummy_e2e_cache_"))
atexit.register(shutil.rmtree, _CACHE_ROOT, ignore_errors=True)


def _server(
    arch: ArchCase,
    overrides: dict[str, Any],
    backend: str,
    graph: str,
    devices: list[str],
    selector: str,
    *,
    port: int | None = None,
    extra_args: tuple[str, ...] = (),
    kv_transfer_config: dict[str, Any] | None = None,
) -> RemoteOpenAIServer:
    port = port or _open_port()
    args = [
        *arch.args,
        *extra_args,
        "--port",
        str(port),
        "--load-format",
        "dummy",
        "--worker-extension-cls",
        "moriio_wide_dummy.WideDummyWeights",
        "--hf-overrides",
        json.dumps(overrides),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        backend,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(PREFILL_CHUNK),
        "--max-num-seqs",
        "16",
        "--kv-cache-memory-bytes",
        str(2 * 1024**3),
        "--gpu-memory-utilization",
        "0.5",
        "--enable-prefix-caching",
        "--tensor-parallel-size",
        str(len(devices)),
        *_graph_args(graph),
    ]
    role = "direct"
    if kv_transfer_config is not None:
        args += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
        role = kv_transfer_config["kv_role"]
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        **_device_env(selector, devices),
        "VLLM_HOST_IP": "127.0.0.1",
        # vLLM scans upward from VLLM_PORT for the server internal sockets.
        "VLLM_PORT": str(_open_port_block(32)),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_CACHE_ROOT": str(_CACHE_ROOT / role),
        # Only this directory, so the source tree cannot shadow installed vllm.
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(_WIDE_DUMMY_DIR), os.environ.get("PYTHONPATH")])
        ),
        "MORIIO_DUMMY_WEIGHT_RANGE": str(DUMMY_WEIGHT_RANGE),
        **arch.env,
    }
    return RemoteOpenAIServer(
        arch.model, args, env_dict=env, auto_port=False, max_wait_seconds=900
    )


# Ports handed to a server of this test session, so later picks avoid them.
_HANDED_OUT: set[int] = set()
# MoRIIO binds ``port + tp_rank`` (times the DP rank) on every rank, so a
# handshake or notify port needs a free block, not one free port.
_PORT_BLOCK = 8


def _bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def _open_port() -> int:
    while True:
        port = get_open_port()
        if port not in _HANDED_OUT:
            _HANDED_OUT.add(port)
            return port


def _open_port_block(size: int = _PORT_BLOCK) -> int:
    """First port of ``size`` consecutive free ports no other server got."""
    while True:
        base = get_open_port()
        block = range(base, base + size)
        if _HANDED_OUT.isdisjoint(block) and all(_bindable(p) for p in block):
            _HANDED_OUT.update(block)
            return base


def _moriio_config(
    role: str, http_port: int, ping_port: int, read_mode: bool = False
) -> dict[str, Any]:
    return {
        "kv_connector": "MoRIIOConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "backend": BACKEND,
            "read_mode": read_mode,
            "host_ip": "127.0.0.1",
            "proxy_ip": "127.0.0.1",
            "proxy_ping_port": str(ping_port),
            "http_port": str(http_port),
            # Every TP rank binds its port plus its rank offset.
            "handshake_port": str(_open_port_block()),
            "notify_port": str(_open_port_block()),
        },
    }


@contextlib.contextmanager
def _proxy(ping_port: int) -> Iterator[str]:
    # Free ports for both sockets, so cases from parallel shards on one host
    # do not collide.
    port = _open_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(_PROXY),
            "--port",
            str(port),
            "--service-discovery-port",
            str(ping_port),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert process.poll() is None, "MoRIIO proxy exited during startup"
            try:
                requests.get(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.2)
        else:
            raise TimeoutError("MoRIIO proxy did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


@contextlib.contextmanager
def _pd_pair(
    arch: ArchCase,
    prefill_overrides: dict[str, Any],
    decode_overrides: dict[str, Any],
    case: Case,
) -> Iterator[tuple[str, RemoteOpenAIServer, RemoteOpenAIServer]]:
    selector, devices = _visible_devices()
    prefill_devices = devices[: case.prefill_tp]
    decode_devices = devices[case.prefill_tp : case.prefill_tp + case.decode_tp]
    assert len(decode_devices) == case.decode_tp, (
        f"needs {case.prefill_tp} GPUs for prefill and {case.decode_tp} for decode"
    )
    prefill_port, decode_port, ping_port = (_open_port() for _ in range(3))
    with _proxy(ping_port) as proxy_url, contextlib.ExitStack() as stack:
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(
                    _server,
                    arch,
                    prefill_overrides,
                    case.attention_backend,
                    case.prefill_graph,
                    prefill_devices,
                    selector,
                    port=prefill_port,
                    extra_args=case.extra_args,
                    kv_transfer_config=_moriio_config(
                        "kv_producer", prefill_port, ping_port, case.read_mode
                    ),
                ),
                pool.submit(
                    _server,
                    arch,
                    decode_overrides,
                    case.attention_backend,
                    case.decode_graph,
                    decode_devices,
                    selector,
                    port=decode_port,
                    extra_args=case.extra_args,
                    kv_transfer_config=_moriio_config(
                        "kv_consumer", decode_port, ping_port, case.read_mode
                    ),
                ),
            ]
            for future in futures:
                servers.append(future.result())
        _wait_for_registration(proxy_url, arch.model)
        yield proxy_url, servers[0], servers[1]


def _wait_for_registration(proxy_url: str, model: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        # The proxy hands decode max_tokens - 1 (prefill emits the first
        # token), so the probe needs at least 2 for decode to accept it.
        try:
            response = requests.post(
                f"{proxy_url}/v1/completions",
                json={"model": model, "prompt": [1000, 1001], "max_tokens": 2},
                timeout=300,
            )
            # The proxy streams the decode body, so a decode-side error can
            # surface as a truncated response rather than a status code.
            if response.status_code == 200 and response.json()["choices"]:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("prefill/decode never registered with the MoRIIO proxy")


# ---------------------------------------------------------------------------
# Measurement


def _metric(server: RemoteOpenAIServer, name: str) -> list[Any]:
    from prometheus_client.parser import text_string_to_metric_families

    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == name
    ]


def _block_size(server: RemoteOpenAIServer) -> int:
    # Hybrid models pad the attention block to the recurrent-state page, so the
    # effective size is only known once the server has started.
    sizes = {
        int(sample.labels["block_size"])
        for sample in _metric(server, "vllm:cache_config_info")
    }
    assert len(sizes) == 1, f"expected one cache block size, got {sizes}"
    return sizes.pop()


def _prompt_lengths(arch: ArchCase, server: RemoteOpenAIServer) -> list[int]:
    # Hybrid chunks end on cache-block boundaries; attention-only chunks end on
    # the prefill budget, so the last token is computed on its own either way.
    if arch.hybrid:
        unit, counts = _block_size(server), ALIGNED_BLOCKS
    else:
        unit, counts = PREFILL_CHUNK, ALIGNED_CHUNKS
    aligned = [
        k * unit + 1 for k in counts if k * unit + 1 + MAX_TOKENS <= MAX_MODEL_LEN
    ]
    assert aligned, f"alignment unit {unit} leaves no aligned prompt"
    return [SHORT_PROMPT_LEN, *aligned]


def _prompts(lengths: list[int]) -> list[list[int]]:
    rng = random.Random(0)
    return [[rng.randrange(1000, 30000) for _ in range(n)] for n in lengths]


def _first_step(url: str, model: str, prompt: list[int]) -> dict[str, float]:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "logprobs": TOP_K,
            "return_tokens_as_token_ids": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    top = response.json()["choices"][0]["logprobs"]["top_logprobs"]
    assert top, "server returned no logprobs"
    return top[0]


def _distance(ref: dict[str, float], got: dict[str, float]) -> float:
    """Max difference of centered log-probs over the shared top-k tokens.

    Centering removes the log-sum-exp offset, leaving the logit differences the
    KV actually determines.
    """
    common = sorted(ref.keys() & got.keys())
    if len(common) < len(ref) // 2:
        return float("inf")
    ref_mean = sum(ref[t] for t in common) / len(common)
    got_mean = sum(got[t] for t in common) / len(common)
    return max(abs((ref[t] - ref_mean) - (got[t] - got_mean)) for t in common)


def _wait_for_prefill_blocks_freed(prefill: RemoteOpenAIServer) -> None:
    """Prefill frees a finished request's blocks once every decode rank that
    holds a share of it has released it. A missed release leaves them in use
    until the deferred-send reaper frees them after 60 s."""
    deadline = time.monotonic() + PREFILL_FREE_TIMEOUT_S
    while True:
        usage = sum(s.value for s in _metric(prefill, "vllm:kv_cache_usage_perc"))
        if usage == 0:
            return
        assert time.monotonic() < deadline, (
            f"prefill still holds {usage:.2%} of its KV cache "
            f"{PREFILL_FREE_TIMEOUT_S} s after the last request finished"
        )
        time.sleep(0.5)


def _external_hits(server: RemoteOpenAIServer) -> float:
    samples = _metric(server, "vllm:external_prefix_cache_hits_total")
    return sum(sample.value for sample in samples)


@dataclass
class Reference:
    prompts: list[list[int]]
    logprobs: list[dict[str, float]]
    hybrid: bool
    read_mode: bool = False

    @property
    def expected_external_hits(self) -> int:
        return _expected_external_hits(self.prompts, self.hybrid, self.read_mode)


def _expected_external_hits(
    prompts: list[list[int]], hybrid: bool, read_mode: bool
) -> int:
    # Hybrid prefill stops one token short, and a READ decode always computes
    # the last prompt token itself; an attention-only WRITE hands decode the
    # whole prompt.
    return sum(len(p) - int(hybrid or read_mode) for p in prompts)


def _reference(
    arch: ArchCase, overrides: dict[str, Any], case: Case, tp: int | None = None
) -> Reference:
    """Direct serve; also checks that the comparison can see a wrong prefix.

    It runs at the decode TP size unless ``tp`` is given.
    """
    selector, devices = _visible_devices()
    devices = devices[: tp or case.decode_tp]
    # Prefill graphs run the chunks and the decode-only last step runs the same
    # captured graph in every decode mode used here, so the prefill mode
    # reproduces both halves of the P/D computation.
    with _server(
        arch,
        overrides,
        case.attention_backend,
        case.prefill_graph,
        devices,
        selector,
        extra_args=case.extra_args,
    ) as server:
        url = server.url_root
        lengths = _prompt_lengths(arch, server)
        prompts = _prompts(lengths)
        refs = [_first_step(url, arch.model, p) for p in prompts]
        # Same tail, different early context: the comparison has to see it.
        for prompt, ref in zip(prompts, refs):
            altered = [t + 1 for t in prompt[: len(prompt) // 2]]
            altered += prompt[len(prompt) // 2 :]
            moved = _distance(ref, _first_step(url, arch.model, altered))
            print(f"[sensitivity] len={len(prompt)} context-change distance={moved}")
            assert moved >= MIN_CONTEXT_EFFECT, (
                f"first-token log-probs barely depend on the context (moved "
                f"{moved:.4g} for len {len(prompt)}); a wrong prefix would go "
                "unnoticed"
            )
        # Determinism of the reference itself (served from its prefix cache).
        again = _first_step(url, arch.model, prompts[-1])
        print(f"[noise] repeat distance={_distance(refs[-1], again)}")
    return Reference(prompts, refs, arch.hybrid, case.read_mode)


def _pd_distances(
    arch: ArchCase,
    prefill_overrides: dict[str, Any],
    decode_overrides: dict[str, Any],
    case: Case,
    ref: Reference,
) -> tuple[list[float], float]:
    with _pd_pair(arch, prefill_overrides, decode_overrides, case) as (
        proxy_url,
        _,
        decode,
    ):
        hits_before = _external_hits(decode)
        distances = [
            _distance(expected, _first_step(proxy_url, arch.model, prompt))
            for prompt, expected in zip(ref.prompts, ref.logprobs)
        ]
        hits = _external_hits(decode) - hits_before
    lengths = [len(p) for p in ref.prompts]
    print(f"[pd] {case.id} lens={lengths} distances={distances} external_hits={hits}")
    return distances, hits


# ---------------------------------------------------------------------------
# Tests


def _check_matches_direct_serve(case: Case) -> None:
    arch = ARCHS[case.arch]
    overrides = _arch_overrides(arch)
    ref = _reference(arch, overrides, case)
    distances, hits = _pd_distances(arch, overrides, overrides, case, ref)

    # Every transferred token of every prompt must come from prefill.
    assert hits == ref.expected_external_hits, (
        f"decode reported {hits} external prefix-cache hits, expected "
        f"{ref.expected_external_hits}; it did not use all the transferred KV"
    )
    lengths = [len(p) for p in ref.prompts]
    bad = {n: d for n, d in zip(lengths, distances) if not d <= TOLERANCE}
    assert not bad, f"P/D first-token log-probs differ from direct serve: {bad}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_write_matches_direct_serve(case: Case):
    _check_matches_direct_serve(case)


@pytest.mark.parametrize("case", READ_CASES, ids=lambda c: c.id)
def test_read_matches_direct_serve(case: Case):
    _check_matches_direct_serve(case)


@pytest.mark.parametrize(
    "case", HETEROGENEOUS_TP_CASES + HETEROGENEOUS_TP_READ_CASES, ids=lambda c: c.id
)
def test_heterogeneous_tp_hands_decode_every_prompt_token(case: Case):
    """Prefill and decode of different TP sizes complete WRITE and READ.

    Unlike the equal-TP cases this cannot compare against a direct serve:
    a different TP size sums in a different order, and on dummy weights that
    changes the top tokens entirely, so log-probs only show the transfer did
    not hang and decode consumed the transferred KV. That every head and state
    slice lands at the right offset is pinned byte for byte by
    tests/v1/kv_connector/unit/test_moriio_hetero_tp_write.py and
    tests/v1/kv_connector/unit/test_moriio_read_tp.py.
    """
    _, devices = _visible_devices()
    if len(devices) < case.prefill_tp + case.decode_tp:
        pytest.skip(f"needs {case.prefill_tp + case.decode_tp} GPUs")
    if BACKEND == "xgmi" and _reads_every_prefill_gpu(case):
        pytest.skip("MoRI xGMI cannot open a session to a GPU decode does not see")
    arch = ARCHS[case.arch]
    overrides = _arch_overrides(arch)
    with _pd_pair(arch, overrides, overrides, case) as (proxy_url, prefill, decode):
        lengths = _prompt_lengths(arch, decode)
        prompts = _prompts(lengths)
        hits_before = _external_hits(decode)
        tops = [_first_step(proxy_url, arch.model, prompt) for prompt in prompts]
        hits = _external_hits(decode) - hits_before
        _wait_for_prefill_blocks_freed(prefill)
    print(f"[pd] {case.id} lens={lengths} external_hits={hits}")

    expected = _expected_external_hits(prompts, arch.hybrid, case.read_mode)
    assert hits == expected, (
        f"decode reported {hits} external prefix-cache hits, expected {expected}"
    )
    assert all(
        len(top) >= TOP_K and all(math.isfinite(v) for v in top.values())
        for top in tops
    ), f"decode returned non-finite log-probs: {tops}"


@pytest.mark.parametrize("arch_name", sorted(ARCHS))
def test_corrupted_prefill_state_is_detected(arch_name: str):
    """Negative control: KV computed with different values must fail."""
    _check_corrupted_prefill_state_is_detected(arch_name, read_mode=False)


@pytest.mark.parametrize("arch_name", ["kimi_linear", "qwen3_dense"])
def test_corrupted_prefill_state_is_detected_read(arch_name: str):
    """The same negative control with decode pulling the KV."""
    _check_corrupted_prefill_state_is_detected(arch_name, read_mode=True)


def _check_corrupted_prefill_state_is_detected(arch_name: str, read_mode: bool) -> None:
    arch = ARCHS[arch_name]
    default = next(c for c in CASES if c.arch == arch_name)
    case = Case(
        arch_name,
        default.attention_backend,
        _DEFAULT,
        _DEFAULT,
        default.hma_off,
        read_mode=read_mode,
    )
    overrides = _arch_overrides(arch)
    ref = _reference(arch, overrides, case)
    corrupted = _merge(overrides, arch.corrupt_overrides)
    distances, hits = _pd_distances(arch, corrupted, overrides, case, ref)

    assert hits == ref.expected_external_hits, (
        "decode did not use the transferred KV; the control proves nothing"
    )
    assert all(d > TOLERANCE for d in distances), (
        f"P/D outputs with wrongly computed prefill KV still matched the "
        f"reference (distances {distances}); the comparison cannot detect "
        "transfer corruption"
    )
