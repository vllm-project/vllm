#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the CPU EC connector over NIXL.

Spins up two `vllm serve` instances on a single pod (producer + consumer)
and drives them via the disaggregated ``/inference/v1/generate`` endpoint
to assert that an image's encoder cache flows producer → consumer via
NIXL without re-encoding on the consumer.

Driving sequence:

  1. Render an image+prompt once on the consumer to get token_ids+features.
  2. Control request to consumer (no ``ec_transfer_params``) — proves the
     consumer's encoder path is wired up; encoder MUST fire.
  3. POST ``/reset_mm_cache`` and ``/reset_prefix_cache`` on the consumer.
  4. Producer encode: ``/inference/v1/generate`` with ``max_tokens=1``,
     producing the encoding and returning ``ec_transfer_params``.
  5. Consumer EC request: ``/inference/v1/generate`` with the producer's
     ``ec_transfer_params``. Consumer encoder MUST NOT fire; XferReq /
     NIXL WRITE / XferAck / start_load_caches MUST appear in logs.

Verification is log-based: each server is launched with a sitecustomize
patch that wraps the EC connector and the model runner's encoder with
INFO log lines. The driver snapshots each log file before each step and
asserts the expected lines appear (or don't) in the appended slice.

Run from the repo root::

    python scripts/test_cpu_ec_e2e.py

Requires two CUDA devices visible to the host (defaults to GPUs 0/1).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pybase64
import regex as re
import requests

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_DIR = Path(__file__).resolve().parent  # contains sitecustomize.py
DEFAULT_IMAGE = REPO_ROOT / "tests/v1/ec_connector/integration/hato.jpg"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_PRODUCER_GPU = 0
DEFAULT_CONSUMER_GPU = 1
DEFAULT_PRODUCER_PORT = 8001
DEFAULT_CONSUMER_PORT = 8002
DEFAULT_PRODUCER_SIDE_PORT = 5601
DEFAULT_CONSUMER_SIDE_PORT = 5602  # consumer doesn't bind, but kept distinct.

HEALTH_TIMEOUT_S = 600  # cold-start of Qwen3-VL-2B on H100 is comfortably under this.
REQUEST_TIMEOUT_S = 120


@dataclass
class ServerSpec:
    role: str  # "producer" or "consumer"
    gpu: int
    http_port: int
    side_channel_port: int
    engine_id: str
    gpu_memory_utilization: float
    log_path: Path
    num_ec_blocks: int = 80000


# -----------------------------------------------------------------------------
# Server lifecycle
# -----------------------------------------------------------------------------


def build_vllm_argv(spec: ServerSpec, model: str) -> list[str]:
    """Argv for a single `vllm serve` invocation.

    Both sides load the full model. The producer's KV / prefix-cache
    footprint is intentionally small — only the encoder needs to run for
    EC saves to populate.
    """
    ec_role = "ec_producer" if spec.role == "producer" else "ec_consumer"
    ec_cfg = {
        "ec_connector": "ECCPUConnector",
        "ec_role": ec_role,
        "engine_id": spec.engine_id,
        "ec_connector_extra_config": {"num_ec_blocks": spec.num_ec_blocks},
    }
    argv = [
        "vllm",
        "serve",
        model,
        "--port",
        str(spec.http_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
        "--ec-transfer-config",
        json.dumps(ec_cfg),
    ]
    if spec.role == "producer":
        argv.append("--no-enable-prefix-caching")
    return argv


def spawn_server(spec: ServerSpec, model: str) -> subprocess.Popen:
    """Spawn one vllm serve instance with sitecustomize patches pre-loaded.

    The patches must apply to every Python interpreter in the server's
    process tree (the vision encoder forward runs in vLLM's EngineCore
    subprocess, not in the API-server parent). Delivering them via a
    `sitecustomize.py` on PYTHONPATH gets every interpreter — parent,
    EngineCore, and any helper subprocess.
    """
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(spec.gpu),
        "VLLM_EC_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_EC_SIDE_CHANNEL_PORT": str(spec.side_channel_port),
        "EC_TEST_ROLE": spec.role,
        "PYTHONPATH": str(PATCHES_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        # Gate for the /reset_mm_cache and /reset_prefix_cache routes (see
        # vllm/entrypoints/serve/cache/api_router.py:attach_router).
        "VLLM_SERVER_DEV_MODE": "1",
    }
    argv = build_vllm_argv(spec, model)
    # `python -m` instead of the `vllm` binary so the interpreter is the
    # one running this script, regardless of $PATH.
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", *argv[1:]]

    log_fh = spec.log_path.open("wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # Place each server in its own process group so the driver can SIGTERM
        # the whole tree (worker subprocesses included) cleanly on teardown.
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def wait_for_health(port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    """Poll /health until the server responds, the process dies, or we time out."""
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server on port {port} exited with code {proc.returncode} "
                f"before becoming healthy"
            )
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(
        f"server on port {port} did not become healthy within {timeout_s}s"
    )


def shutdown_server(proc: subprocess.Popen, name: str, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        print(
            f"[teardown] {name} did not exit on SIGTERM within "
            f"{grace_s:.0f}s; sending SIGKILL",
            file=sys.stderr,
        )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        fh.close()


# -----------------------------------------------------------------------------
# Log windowing — assertions are stated as "X must / must not appear in the
# slice of <log file> appended during the request window".
# -----------------------------------------------------------------------------


def log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def log_slice(path: Path, start: int) -> str:
    """Read the bytes appended to `path` since offset `start`."""
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(start)
        return f.read().decode("utf-8", errors="replace")


def assert_in_log(haystack: str, needle: str, *, where: str) -> None:
    if needle not in haystack:
        raise AssertionError(
            f"expected log line {needle!r} in {where} but did not find it.\n"
            f"--- log slice ---\n{haystack}\n--- end slice ---"
        )
    print(f"  ✓ found {needle!r} in {where}")


def assert_not_in_log(haystack: str, needle: str, *, where: str) -> None:
    if needle in haystack:
        raise AssertionError(
            f"did not expect log line {needle!r} in {where} but found it.\n"
            f"--- log slice ---\n{haystack}\n--- end slice ---"
        )
    print(f"  ✓ absent {needle!r} from {where}")


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    b64 = pybase64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


def synth_image_data_url(seed: int, size: tuple[int, int] = (224, 224)) -> str:
    """Generate a deterministic-but-unique JPEG and return its data URL.

    Pixel content is a per-seed gradient so each seed yields a distinct
    `mm_hash`. PIL is a vLLM dependency so the import is safe.
    """
    from PIL import Image

    rng_r = (seed * 73 + 11) & 0xFF
    rng_g = (seed * 151 + 29) & 0xFF
    rng_b = (seed * 223 + 47) & 0xFF
    w, h = size
    img = Image.new("RGB", size, (rng_r, rng_g, rng_b))
    # Add a per-seed diagonal stripe so two seeds with similar base colors
    # still differ across the encoded patches.
    px = img.load()
    for i in range(min(w, h)):
        px[i, i] = ((rng_r + 80) & 0xFF, (rng_g + 80) & 0xFF, (rng_b + 80) & 0xFF)
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = pybase64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render(
    consumer_port: int,
    model: str,
    image_urls: str | list[str],
    prompt: str,
) -> dict:
    if isinstance(image_urls, str):
        image_urls = [image_urls]
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": u}} for u in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    r = requests.post(
        f"http://127.0.0.1:{consumer_port}/v1/chat/completions/render",
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def generate(
    port: int,
    rendered: dict,
    *,
    max_tokens: int,
    ec_transfer_params: dict | None = None,
) -> dict:
    sampling_params: dict = {"max_tokens": max_tokens, "temperature": 0.0}
    if ec_transfer_params is not None:
        # The engine reads ec_transfer_params from sampling_params.extra_args
        # (see vllm/v1/request.py); the top-level GenerateRequest field is
        # response-only.
        sampling_params["extra_args"] = {"ec_transfer_params": ec_transfer_params}
    body = {
        "token_ids": rendered["token_ids"],
        "features": rendered.get("features"),
        "sampling_params": sampling_params,
    }
    r = requests.post(
        f"http://127.0.0.1:{port}/inference/v1/generate",
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def reset_mm_cache(port: int) -> None:
    r = requests.post(
        f"http://127.0.0.1:{port}/reset_mm_cache",
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()


def reset_prefix_cache(port: int) -> None:
    r = requests.post(
        f"http://127.0.0.1:{port}/reset_prefix_cache",
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()


# -----------------------------------------------------------------------------
# Decoding the consumer's reply (best-effort, just for human-readable output)
# -----------------------------------------------------------------------------


def decode_tokens(model: str, token_ids: list[int]) -> str:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return f"<{len(token_ids)} tokens; install transformers to decode>"
    tok = AutoTokenizer.from_pretrained(model)
    return tok.decode(token_ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------


def cleanup_dev_shm() -> None:
    """Wipe any stale `/dev/shm/vllm_ec_*.mmap` so the next spawn binds to a
    freshly-sized region (an existing file is reused without resize)."""
    stale = list(Path("/dev/shm").glob("vllm_ec_*.mmap"))
    for f in stale:
        try:
            f.unlink()
        except OSError as e:
            print(f"[setup] warning: could not remove {f}: {e}", file=sys.stderr)
    if stale:
        print(f"[setup] cleared {len(stale)} stale mmap file(s) from /dev/shm")


def make_specs(
    args,
    log_dir: Path,
    *,
    producer_num_ec_blocks: int = 80000,
    consumer_num_ec_blocks: int = 80000,
) -> tuple[ServerSpec, ServerSpec]:
    """Default ServerSpec pair: GPU 0 producer, GPU 1 consumer, distinct ports."""
    producer = ServerSpec(
        role="producer",
        gpu=args.producer_gpu,
        http_port=args.producer_port,
        side_channel_port=DEFAULT_PRODUCER_SIDE_PORT,
        engine_id="ec-producer-0",
        gpu_memory_utilization=0.05,
        log_path=log_dir / "producer.log",
        num_ec_blocks=producer_num_ec_blocks,
    )
    consumer = ServerSpec(
        role="consumer",
        gpu=args.consumer_gpu,
        http_port=args.consumer_port,
        side_channel_port=DEFAULT_CONSUMER_SIDE_PORT,
        engine_id="ec-consumer-0",
        gpu_memory_utilization=0.5,
        log_path=log_dir / "consumer.log",
        num_ec_blocks=consumer_num_ec_blocks,
    )
    return producer, consumer


# -----------------------------------------------------------------------------
# Harness — context manager that owns both processes and per-side log views.
# -----------------------------------------------------------------------------


class Harness:
    def __init__(
        self,
        producer: ServerSpec,
        consumer: ServerSpec,
        model: str,
        *,
        keep_on_exit: bool = False,
    ):
        self.producer = producer
        self.consumer = consumer
        self.model = model
        self.keep_on_exit = keep_on_exit
        self.producer_proc: subprocess.Popen | None = None
        self.consumer_proc: subprocess.Popen | None = None

    def __enter__(self) -> Harness:
        print(
            f"[setup] spawning producer (gpu={self.producer.gpu}, "
            f"port={self.producer.http_port}, "
            f"num_ec_blocks={self.producer.num_ec_blocks})"
        )
        self.producer_proc = spawn_server(self.producer, self.model)
        print(
            f"[setup] spawning consumer (gpu={self.consumer.gpu}, "
            f"port={self.consumer.http_port}, "
            f"num_ec_blocks={self.consumer.num_ec_blocks})"
        )
        self.consumer_proc = spawn_server(self.consumer, self.model)
        self._wait_both_healthy()
        return self

    def __exit__(self, *_exc) -> None:
        if self.keep_on_exit:
            print("\n[teardown] --keep-servers set; leaving both servers running.")
            for proc, port in [
                (self.producer_proc, self.producer.http_port),
                (self.consumer_proc, self.consumer.http_port),
            ]:
                if proc is not None:
                    print(f"  pid={proc.pid}  port={port}")
            return
        print("\n[teardown] stopping servers")
        if self.consumer_proc is not None:
            shutdown_server(self.consumer_proc, "consumer")
        if self.producer_proc is not None:
            shutdown_server(self.producer_proc, "producer")

    def _wait_both_healthy(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"[setup] waiting on /health for both (up to {HEALTH_TIMEOUT_S}s)…")
        pairs = [
            (self.producer, self.producer_proc),
            (self.consumer, self.consumer_proc),
        ]
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {
                ex.submit(wait_for_health, s.http_port, p, HEALTH_TIMEOUT_S): (
                    s.role,
                    s.http_port,
                )
                for s, p in pairs
                if p is not None
            }
            for fut in as_completed(futs):
                fut.result()
                name, port = futs[fut]
                print(f"  ✓ {name} healthy on {port}")

    def restart_producer(self) -> None:
        """SIGTERM the producer and respawn with the same spec.

        Used by `test_producer_restart` to exercise the consumer's stale-
        agent-metadata path. The mmap region is recreated fresh by the new
        producer process; consumers must detect the new `nixl_agent_metadata`
        and re-add the remote agent.
        """
        assert self.producer_proc is not None
        print("\n[restart] SIGTERM producer")
        shutdown_server(self.producer_proc, "producer")
        # Region file persists; wipe so the new instance starts clean.
        cleanup_dev_shm()
        print("[restart] spawning new producer")
        self.producer_proc = spawn_server(self.producer, self.model)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(
                wait_for_health,
                self.producer.http_port,
                self.producer_proc,
                HEALTH_TIMEOUT_S,
            ).result()
        print(f"  ✓ producer healthy on {self.producer.http_port}")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def _producer_encode(h: Harness, rendered: dict) -> tuple[dict, str]:
    """Drive a producer encode on `rendered`, return (ec_transfer_params, slice)."""
    mark = log_size(h.producer.log_path)
    resp = generate(h.producer.http_port, rendered, max_tokens=1)
    sl = log_slice(h.producer.log_path, mark)
    return resp.get("ec_transfer_params") or {}, sl


def test_baseline(h: Harness, image: Path, prompt: str) -> None:
    print("\n=== test_baseline ===")
    data_url = image_data_url(image)
    rendered = render(h.consumer.http_port, h.model, data_url, prompt)
    target_hash = rendered["features"]["mm_hashes"]["image"][0]
    print(f"  rendered: token_ids={len(rendered['token_ids'])}, mm_hash={target_hash}")

    # Control: encoder must run locally.
    cmark = log_size(h.consumer.log_path)
    ctrl = generate(h.consumer.http_port, rendered, max_tokens=80)
    assert_in_log(
        log_slice(h.consumer.log_path, cmark),
        "consumer ENCODER FORWARD",
        where="consumer.log [control]",
    )
    print(
        f"  control: {decode_tokens(h.model, ctrl['choices'][0]['token_ids'] or [])!r}"
    )

    reset_mm_cache(h.consumer.http_port)
    reset_prefix_cache(h.consumer.http_port)

    # Producer encode → ec_transfer_params.
    ec_params, prod_sl = _producer_encode(h, rendered)
    assert_in_log(prod_sl, "producer ENCODER FORWARD", where="producer.log [encode]")
    assert_in_log(
        prod_sl, f"producer save mm_hash={target_hash}", where="producer.log [encode]"
    )
    if target_hash not in ec_params:
        raise AssertionError(
            f"producer response missing ec_transfer_params for {target_hash}; "
            f"got keys={list(ec_params.keys())}"
        )
    info = ec_params[target_hash]
    for key in ("peer_host", "peer_port", "size_bytes", "nixl_agent_metadata_b64"):
        if key not in info:
            raise AssertionError(f"ec_transfer_params[{target_hash}] missing {key!r}")

    # Consumer EC: must NOT encode.
    pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
    resp = generate(
        h.consumer.http_port, rendered, max_tokens=80, ec_transfer_params=ec_params
    )
    psl, csl = (
        log_slice(h.producer.log_path, pmark),
        log_slice(h.consumer.log_path, cmark),
    )
    assert_in_log(
        psl, f"producer XferReq mm_hash={target_hash}", where="producer.log [ec]"
    )
    assert_in_log(
        psl, f"producer NIXL WRITE mm_hash={target_hash}", where="producer.log [ec]"
    )
    assert_in_log(
        csl,
        f"consumer XferAck ok=True mm_hash={target_hash}",
        where="consumer.log [ec]",
    )
    assert_in_log(csl, "consumer load mm_hashes=", where="consumer.log [ec]")
    assert_not_in_log(csl, "consumer ENCODER FORWARD", where="consumer.log [ec]")
    print(f"  EC: {decode_tokens(h.model, resp['choices'][0]['token_ids'] or [])!r}")
    print("  ✓ test_baseline")


def test_cache_reuse(h: Harness, prompt: str, n_repeat: int = 5) -> None:
    """Repeat consumer EC requests for the same mm_hash should hit `_loaded`
    and re-copy from local mmap, not re-fetch from the producer.

    Uses a synthetic image (seed scoped uniquely to this test) because the
    connector's `_loaded` set has no public reset; if the same mm_hash had
    already been EC-fetched by an earlier test, the first iteration here
    would silently hit `_loaded` and skip the fresh fetch we are trying
    to measure.
    """
    print(f"\n=== test_cache_reuse (n_repeat={n_repeat}) ===")
    rendered = render(
        h.consumer.http_port, h.model, synth_image_data_url(seed=50), prompt
    )
    target_hash = rendered["features"]["mm_hashes"]["image"][0]
    reset_mm_cache(h.consumer.http_port)
    reset_prefix_cache(h.consumer.http_port)
    ec_params, _ = _producer_encode(h, rendered)
    if target_hash not in ec_params:
        raise AssertionError(f"producer did not announce {target_hash}")

    pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
    for i in range(n_repeat):
        # Resetting prefix cache before each call ensures the engine actually
        # schedules the request (otherwise prefix-cache hit would short-circuit
        # the load entirely on iteration 2+).
        reset_prefix_cache(h.consumer.http_port)
        resp = generate(
            h.consumer.http_port, rendered, max_tokens=8, ec_transfer_params=ec_params
        )
        if not (resp["choices"][0]["token_ids"] or []):
            raise AssertionError(f"empty response on iter {i}")

    psl = log_slice(h.producer.log_path, pmark)
    csl = log_slice(h.consumer.log_path, cmark)
    n_xfer_req = psl.count(f"producer XferReq mm_hash={target_hash}")
    n_nixl_write = psl.count("producer NIXL WRITE")
    n_load = csl.count("consumer load mm_hashes=")
    n_encoder = csl.count("consumer ENCODER FORWARD")
    print(
        f"  XferReqs={n_xfer_req}, NIXL WRITEs={n_nixl_write}, "
        f"loads={n_load}, encoder forwards={n_encoder}"
    )
    if n_xfer_req != 1:
        raise AssertionError(
            f"expected exactly 1 XferReq across {n_repeat} repeats, got {n_xfer_req}"
        )
    if n_nixl_write != 1:
        raise AssertionError(
            f"expected exactly 1 NIXL WRITE across {n_repeat} repeats, "
            f"got {n_nixl_write}"
        )
    if n_load < n_repeat:
        raise AssertionError(f"expected ≥{n_repeat} consumer loads, got {n_load}")
    if n_encoder != 0:
        raise AssertionError(
            f"consumer should never re-encode but ENCODER FORWARD fired {n_encoder}×"
        )
    print(f"  ✓ test_cache_reuse — single fetch served {n_repeat} requests via _loaded")


def test_multi_image(h: Harness, prompt: str, n_images: int = 3) -> None:
    """One request carrying N distinct images. Producer fans out to N
    `save_caches`; consumer EC fetches all N concurrently in one request."""
    print(f"\n=== test_multi_image (n_images={n_images}) ===")
    urls = [synth_image_data_url(seed=100 + i) for i in range(n_images)]
    rendered = render(h.consumer.http_port, h.model, urls, prompt)
    hashes = rendered["features"]["mm_hashes"]["image"]
    if len(hashes) != n_images:
        raise AssertionError(f"expected {n_images} mm_hashes, got {len(hashes)}")
    print(f"  mm_hashes: {hashes}")
    reset_mm_cache(h.consumer.http_port)
    reset_prefix_cache(h.consumer.http_port)

    ec_params, prod_sl = _producer_encode(h, rendered)
    for hh in hashes:
        if hh not in ec_params:
            raise AssertionError(f"producer omitted ec_transfer_params for {hh}")
        assert_in_log(
            prod_sl,
            f"producer save mm_hash={hh}",
            where="producer.log [multi-image encode]",
        )

    pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
    _ = generate(
        h.consumer.http_port, rendered, max_tokens=20, ec_transfer_params=ec_params
    )
    psl, csl = (
        log_slice(h.producer.log_path, pmark),
        log_slice(h.consumer.log_path, cmark),
    )
    for hh in hashes:
        assert_in_log(
            psl, f"producer XferReq mm_hash={hh}", where="producer.log [multi-image ec]"
        )
        assert_in_log(
            psl,
            f"producer NIXL WRITE mm_hash={hh}",
            where="producer.log [multi-image ec]",
        )
        assert_in_log(
            csl,
            f"consumer XferAck ok=True mm_hash={hh}",
            where="consumer.log [multi-image ec]",
        )
    assert_not_in_log(
        csl, "consumer ENCODER FORWARD", where="consumer.log [multi-image ec]"
    )
    print(f"  ✓ test_multi_image — all {n_images} fetched, none re-encoded")


def test_concurrent_ec(h: Harness, prompt: str, k: int = 4) -> None:
    """K parallel consumer EC requests, each for a distinct image. Stresses
    the producer router thread (XferReq accept + WRITE post + sweep + ack)
    under concurrent load."""
    print(f"\n=== test_concurrent_ec (k={k}) ===")
    # Pre-encode each image on the producer separately so each ec_params entry
    # carries one mm_hash (mirrors the realistic per-image orchestrator flow).
    encoded: list[tuple[dict, dict, str]] = []  # (rendered, ec_params, mm_hash)
    for i in range(k):
        url = synth_image_data_url(seed=200 + i)
        rendered = render(h.consumer.http_port, h.model, url, prompt)
        target_hash = rendered["features"]["mm_hashes"]["image"][0]
        ec_params, _ = _producer_encode(h, rendered)
        if target_hash not in ec_params:
            raise AssertionError(f"producer did not announce {target_hash}")
        encoded.append((rendered, ec_params, target_hash))
    print(f"  pre-encoded {k} images on producer")

    reset_mm_cache(h.consumer.http_port)
    reset_prefix_cache(h.consumer.http_port)

    pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = [
            ex.submit(
                generate,
                h.consumer.http_port,
                rendered,
                max_tokens=8,
                ec_transfer_params=ec_params,
            )
            for rendered, ec_params, _ in encoded
        ]
        for f in futs:
            resp = f.result()
            if not (resp["choices"][0]["token_ids"] or []):
                raise AssertionError("empty response in concurrent batch")
    elapsed = time.monotonic() - t0
    psl, csl = (
        log_slice(h.producer.log_path, pmark),
        log_slice(h.consumer.log_path, cmark),
    )
    for _, _, hh in encoded:
        assert_in_log(
            psl, f"producer XferReq mm_hash={hh}", where="producer.log [concurrent]"
        )
        assert_in_log(
            psl, f"producer NIXL WRITE mm_hash={hh}", where="producer.log [concurrent]"
        )
        assert_in_log(
            csl,
            f"consumer XferAck ok=True mm_hash={hh}",
            where="consumer.log [concurrent]",
        )
    assert_not_in_log(
        csl, "consumer ENCODER FORWARD", where="consumer.log [concurrent]"
    )
    print(f"  ✓ test_concurrent_ec — k={k} parallel fetches in {elapsed:.2f}s")


def test_pool_exhaustion(args, log_dir: Path) -> None:
    """Tiny producer pool forces LRU eviction. Consumer requesting an evicted
    mm_hash gets a NACK and falls back to local encode."""
    print("\n=== test_pool_exhaustion ===")
    cleanup_dev_shm()
    pool_size = 400
    producer, consumer = make_specs(args, log_dir, producer_num_ec_blocks=pool_size)
    with Harness(producer, consumer, args.model) as h:
        # Encode A first and read its actual block count from the save log.
        rendered_a = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=300), args.prompt
        )
        hash_a = rendered_a["features"]["mm_hashes"]["image"][0]
        a_params, sl_a = _producer_encode(h, rendered_a)
        if hash_a not in a_params:
            raise AssertionError("producer did not announce A's encoding")
        m = re.search(rf"producer save mm_hash={hash_a} n_blocks=(\d+)", sl_a)
        if not m:
            raise AssertionError(
                f"could not parse n_blocks from producer save log; slice={sl_a!r}"
            )
        per_image_blocks = int(m.group(1))
        # Encode enough additional images to overflow the pool 2× over —
        # guarantees A's blocks are evicted regardless of per-image footprint.
        n_extra = (pool_size // max(per_image_blocks, 1)) * 2 + 1
        print(
            f"  per_image_blocks={per_image_blocks}, pool={pool_size}, "
            f"encoding {n_extra} more images to force eviction"
        )
        for i in range(n_extra):
            rendered = render(
                h.consumer.http_port,
                h.model,
                synth_image_data_url(seed=301 + i),
                args.prompt,
            )
            _producer_encode(h, rendered)

        # Consumer requests A using its stale ec_params. Producer should NACK
        # (mm_hash no longer in `_local_encodings`); consumer falls back to
        # local encode.
        reset_mm_cache(h.consumer.http_port)
        reset_prefix_cache(h.consumer.http_port)
        pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
        resp = generate(
            h.consumer.http_port, rendered_a, max_tokens=8, ec_transfer_params=a_params
        )
        if not (resp["choices"][0]["token_ids"] or []):
            raise AssertionError("evicted-fallback request returned empty body")
        psl = log_slice(h.producer.log_path, pmark)
        csl = log_slice(h.consumer.log_path, cmark)

        assert_in_log(
            psl,
            f"producer XferReq mm_hash={hash_a}",
            where="producer.log [evicted-fetch]",
        )
        # No NIXL WRITE for A in this window — the connector NACKs without
        # posting a transfer when the hash is not in `_local_encodings`.
        if f"producer NIXL WRITE mm_hash={hash_a}" in psl:
            raise AssertionError(
                "expected no NIXL WRITE for evicted hash; A may not have been "
                "evicted (try lowering pool_size or raising n_extra)"
            )
        assert_in_log(
            csl,
            f"consumer XferAck ok=False mm_hash={hash_a}",
            where="consumer.log [evicted-fetch]",
        )
        assert_in_log(
            csl, "consumer ENCODER FORWARD", where="consumer.log [evicted-fetch]"
        )
        print("  ✓ test_pool_exhaustion — evicted hash NACK'd and locally re-encoded")


def test_producer_restart(args, log_dir: Path) -> None:
    """Producer restart while consumer is still up. The consumer's cached
    `nixl_agent_metadata` for the producer endpoint is stale; on first
    contact after restart, `_get_or_add_peer` must detect the change and
    swap in a fresh remote agent.

    Each pass uses a distinct synthetic image — pass 2 must NOT hit the
    consumer's `_loaded` cache from pass 1, otherwise the EC fetch path
    is short-circuited and the post-restart code in `_get_or_add_peer`
    is never exercised.
    """
    print("\n=== test_producer_restart ===")
    cleanup_dev_shm()
    producer, consumer = make_specs(args, log_dir)
    with Harness(producer, consumer, args.model) as h:
        # Pass 1: fresh fetch warms the consumer's `_peer_pool` with the
        # original producer's NIXL agent metadata.
        rendered_a = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=400), args.prompt
        )
        hash_a = rendered_a["features"]["mm_hashes"]["image"][0]
        reset_mm_cache(h.consumer.http_port)
        reset_prefix_cache(h.consumer.http_port)
        ec_params_a, _ = _producer_encode(h, rendered_a)
        if hash_a not in ec_params_a:
            raise AssertionError("producer did not announce hash A on first encode")
        resp_a = generate(
            h.consumer.http_port,
            rendered_a,
            max_tokens=8,
            ec_transfer_params=ec_params_a,
        )
        if not (resp_a["choices"][0]["token_ids"] or []):
            raise AssertionError("pre-restart EC request returned empty body")
        print("  pre-restart EC fetch OK")

        h.restart_producer()

        # Pass 2: distinct image so the consumer cannot short-circuit via
        # `_loaded`; new ec_params carry a fresh nixl_agent_metadata_b64,
        # forcing `_get_or_add_peer` to invalidate the stale entry.
        rendered_b = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=401), args.prompt
        )
        hash_b = rendered_b["features"]["mm_hashes"]["image"][0]
        reset_mm_cache(h.consumer.http_port)
        reset_prefix_cache(h.consumer.http_port)
        ec_params_b, _ = _producer_encode(h, rendered_b)
        if hash_b not in ec_params_b:
            raise AssertionError("producer did not announce hash B on second encode")

        old_md = ec_params_a[hash_a]["nixl_agent_metadata_b64"]
        new_md = ec_params_b[hash_b]["nixl_agent_metadata_b64"]
        if old_md == new_md:
            raise AssertionError(
                "post-restart nixl_agent_metadata_b64 should differ from pre-restart "
                "(producer NIXL agent recreated); got identical bytes"
            )

        pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
        resp_b = generate(
            h.consumer.http_port,
            rendered_b,
            max_tokens=8,
            ec_transfer_params=ec_params_b,
        )
        if not (resp_b["choices"][0]["token_ids"] or []):
            raise AssertionError("post-restart EC request returned empty body")
        psl, csl = (
            log_slice(h.producer.log_path, pmark),
            log_slice(h.consumer.log_path, cmark),
        )
        assert_in_log(
            psl,
            f"producer XferReq mm_hash={hash_b}",
            where="producer.log [post-restart ec]",
        )
        assert_in_log(
            psl,
            f"producer NIXL WRITE mm_hash={hash_b}",
            where="producer.log [post-restart ec]",
        )
        assert_in_log(
            csl,
            f"consumer XferAck ok=True mm_hash={hash_b}",
            where="consumer.log [post-restart ec]",
        )
        assert_not_in_log(
            csl, "consumer ENCODER FORWARD", where="consumer.log [post-restart ec]"
        )
        print(
            "  ✓ test_producer_restart — post-restart EC fetch OK with fresh metadata"
        )


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


TESTS_REQUIRING_DEFAULT_HARNESS = (
    "baseline",
    "cache-reuse",
    "multi-image",
    "concurrent",
)
TESTS_REQUIRING_CUSTOM_HARNESS = ("pool-exhaustion", "producer-restart")
ALL_TESTS = TESTS_REQUIRING_DEFAULT_HARNESS + TESTS_REQUIRING_CUSTOM_HARNESS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--producer-gpu", type=int, default=DEFAULT_PRODUCER_GPU)
    parser.add_argument("--consumer-gpu", type=int, default=DEFAULT_CONSUMER_GPU)
    parser.add_argument("--producer-port", type=int, default=DEFAULT_PRODUCER_PORT)
    parser.add_argument("--consumer-port", type=int, default=DEFAULT_CONSUMER_PORT)
    parser.add_argument(
        "--keep-servers",
        action="store_true",
        help="leave the shared-harness servers running on success",
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=("all", *ALL_TESTS),
        default=None,
        help="test(s) to run; pass multiple times or 'all'. default: all",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    selected = set(args.test or ["all"])
    if "all" in selected:
        selected = set(ALL_TESTS)
    normal_tests = selected & set(TESTS_REQUIRING_DEFAULT_HARNESS)
    custom_tests = selected & set(TESTS_REQUIRING_CUSTOM_HARNESS)

    log_dir = REPO_ROOT / "logs" / "cpu_ec_e2e" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] log dir: {log_dir}")
    print(f"[setup] tests selected: {sorted(selected)}")

    failures: list[tuple[str, BaseException]] = []

    if normal_tests:
        cleanup_dev_shm()
        producer, consumer = make_specs(args, log_dir)
        with Harness(
            producer, consumer, args.model, keep_on_exit=args.keep_servers
        ) as h:
            for test_name in TESTS_REQUIRING_DEFAULT_HARNESS:
                if test_name not in normal_tests:
                    continue
                try:
                    if test_name == "baseline":
                        test_baseline(h, args.image, args.prompt)
                    elif test_name == "cache-reuse":
                        test_cache_reuse(h, args.prompt)
                    elif test_name == "multi-image":
                        test_multi_image(h, args.prompt)
                    elif test_name == "concurrent":
                        test_concurrent_ec(h, args.prompt)
                except AssertionError as e:
                    failures.append((test_name, e))
                    print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    for test_name in TESTS_REQUIRING_CUSTOM_HARNESS:
        if test_name not in custom_tests:
            continue
        try:
            if test_name == "pool-exhaustion":
                test_pool_exhaustion(args, log_dir)
            elif test_name == "producer-restart":
                test_producer_restart(args, log_dir)
        except AssertionError as e:
            failures.append((test_name, e))
            print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    print(f"\n[teardown] logs preserved at {log_dir}")
    if failures:
        print(f"\n[done] {len(failures)} of {len(selected)} tests FAILED:")
        for test_name, e in failures:
            print(f"  - {test_name}: {e}")
        return 1
    print(f"\n[done] all {len(selected)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
