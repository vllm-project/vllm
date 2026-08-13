# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers and test functions for the CPU EC connector e2e suite.

Imported by both test_cpu_ec_e2e.py (single-pod) and
test_cpu_ec_multinode.py (OpenShift multi-node).
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pybase64
import regex as re
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPO_ROOT / "tests/v1/ec_connector/integration/hato.jpg"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
HEALTH_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 120

TESTS_REQUIRING_DEFAULT_HARNESS = (
    "baseline",
    "cache-reuse",
    "multi-image",
    "concurrent",
)
TESTS_REQUIRING_CUSTOM_HARNESS = ("pool-exhaustion", "producer-restart")
ALL_TESTS = TESTS_REQUIRING_DEFAULT_HARNESS + TESTS_REQUIRING_CUSTOM_HARNESS


# ---------------------------------------------------------------------------
# ServerSpec
# ---------------------------------------------------------------------------


@dataclass
class ServerSpec:
    role: str  # "producer" or "consumer"
    gpu: int  # GPU index (used by single-pod harness only)
    http_port: int
    side_channel_port: int
    engine_id: str
    gpu_memory_utilization: float
    log_path: Path
    # Base URL the driver talks to, and the structured event channel to assert
    # on. Both are filled in by whichever harness owns the server: loopback and
    # a local file for LocalHarness, a Route and `oc exec` for K8sHarness.
    base_url: str = ""
    events: EventLog = None  # type: ignore[assignment]
    # Byte budget for the shared EC region. The connector derives
    # num_blocks = ec_cpu_bytes // block_size_bytes at startup, so this is a
    # generous default sized to never evict during the default-harness tests.
    ec_cpu_bytes: int = 4 * 1024**3
    # Devices this instance owns, single-pod harness only; empty means just
    # `gpu`. Its length is the tensor-parallel size, one rank per device. In
    # K8s mode this stays empty because device assignment lives in the pod spec.
    devices: tuple[int, ...] = ()
    # Passed through as --load-format; None leaves vLLM on "auto". Mainly for
    # "fastsafetensors", where each rank reads only its own subset of
    # checkpoint files and the owning rank then broadcasts replicated tensors
    # and scatters TP-sharded ones over NCCL, so a TP>1 instance loads weights
    # without every rank re-reading the whole checkpoint.
    load_format: str | None = None

    @property
    def device_list(self) -> tuple[int, ...]:
        return self.devices or (self.gpu,)

    @property
    def tp_size(self) -> int:
        return len(self.device_list)


# ---------------------------------------------------------------------------
# Health check (shared by LocalHarness and K8sHarness)
# ---------------------------------------------------------------------------


def wait_for_health(base_url: str, proc, timeout_s: int) -> None:
    """Poll /health until the server responds, the process dies, or timeout.

    `proc` may be None (K8s mode — no local process to death-check).
    """
    deadline = time.monotonic() + timeout_s
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"server at {base_url} exited with code {proc.returncode} "
                f"before becoming healthy"
            )
        try:
            r = SESSION.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(
        f"server at {base_url} did not become healthy within {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# Event log (pull-based assertion channel)
# ---------------------------------------------------------------------------
#
# sitecustomize.py appends one JSON object per EC test event to a file inside
# the server's own filesystem. The driver pulls that file synchronously, so an
# assertion sees the true state at the moment it asks. Streamed pod logs
# (`oc logs -f`) are kept for humans only: they lag, they die, and the watchdog
# restarting them can replay history, none of which a test can reason about.
#
# Marks are event indices rather than byte offsets, so a mark stays meaningful
# even if the file is recreated, and a read never returns a half-written line.


class EventLog:
    """Pull-based view of one server's structured EC test events."""

    def read(self, start: int = 0) -> list[str]:
        """Return event messages from index *start* onward."""
        raise NotImplementedError

    def mark(self) -> int:
        """Index one past the last event currently recorded."""
        return len(self.read())

    def wait_all(
        self,
        start: int,
        needles: list[str],
        *,
        timeout_s: float = 20.0,
        poll_s: float = 0.5,
    ) -> list[str]:
        """Poll until every needle appears in events[start:], or timeout.

        Returns the events as of the last read either way, so the caller's
        assertions produce a useful message on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            events = self.read(start)
            if all(any(n in e for e in events) for n in needles):
                return events
            if time.monotonic() >= deadline:
                return events
            time.sleep(poll_s)

    def wait_count(
        self,
        start: int,
        needle: str,
        count: int,
        *,
        timeout_s: float = 20.0,
        poll_s: float = 0.5,
    ) -> list[str]:
        """Poll until *needle* has occurred at least *count* times, or timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            events = self.read(start)
            if count_events(events, needle) >= count:
                return events
            if time.monotonic() >= deadline:
                return events
            time.sleep(poll_s)


class LocalEventLog(EventLog):
    """Event log backed by a file the driver can read directly."""

    def __init__(self, path: Path):
        self.path = path

    def read(self, start: int = 0) -> list[str]:
        try:
            raw = self.path.read_text(errors="replace")
        except FileNotFoundError:
            return []
        return _decode_events(raw)[start:]


class OcExecEventLog(EventLog):
    """Event log read out of a pod with `oc exec`.

    Synchronous by construction: each call is a fresh round trip that returns
    what the pod has written as of now.
    """

    def __init__(self, namespace: str, deployment: str, path: str):
        self._namespace = namespace
        self._deployment = deployment
        self._path = path

    def _exec(self, script: str) -> str:
        """Run *script* in the pod. Raises if `oc exec` itself fails.

        The script is responsible for tolerating a missing event file (no events
        yet is normal). Anything else — an expired token, an unreachable pod —
        must surface: returning "" for those would show up as a puzzling
        "expected event ... not found" instead of the real cause.
        """
        result = subprocess.run(
            [
                "oc",
                "exec",
                f"deployment/{self._deployment}",
                "-n",
                self._namespace,
                "--",
                "sh",
                "-c",
                script,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"oc exec on deployment/{self._deployment} failed "
                f"(rc={result.returncode}): {result.stderr.strip()[:400]}"
            )
        return result.stdout

    def read(self, start: int = 0) -> list[str]:
        # `tail -n +N` ships only the new lines instead of the whole file, and
        # returns nothing if the file is shorter than N — so a pod restart that
        # truncates the file can never produce a stale match.
        return _decode_events(
            self._exec(f"tail -n +{start + 1} {self._path} 2>/dev/null || true")
        )

    def mark(self) -> int:
        out = self._exec(f"wc -l < {self._path} 2>/dev/null || echo 0")
        try:
            return int(out.strip() or 0)
        except ValueError:
            return 0


def _decode_events(raw: str) -> list[str]:
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line)["msg"])
        except (ValueError, KeyError):
            # Ignore anything that isn't one of our records.
            continue
    return events


def count_events(events: list[str], needle: str) -> int:
    return sum(needle in e for e in events)


def _render_events(events: list[str]) -> str:
    return "\n".join(events) if events else "<no events recorded>"


def assert_event(events: list[str], needle: str, *, where: str) -> None:
    if not any(needle in e for e in events):
        raise AssertionError(
            f"expected event {needle!r} in {where} but did not find it.\n"
            f"--- events ---\n{_render_events(events)}\n--- end events ---"
        )
    print(f"  ✓ found {needle!r} in {where}")


def assert_no_event(events: list[str], needle: str, *, where: str) -> None:
    if any(needle in e for e in events):
        raise AssertionError(
            f"did not expect event {needle!r} in {where} but found it.\n"
            f"--- events ---\n{_render_events(events)}\n--- end events ---"
        )
    print(f"  ✓ absent {needle!r} from {where}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


# One session for every request the driver makes. Routes are served with the
# cluster's default wildcard certificate, which won't validate locally, so the
# K8s harness turns verification off for its own base URLs.
SESSION = requests.Session()


def disable_tls_verify() -> None:
    import urllib3

    SESSION.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    b64 = pybase64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


def synth_image_data_url(seed: int, size: tuple[int, int] = (224, 224)) -> str:
    from PIL import Image

    rng_r = (seed * 73 + 11) & 0xFF
    rng_g = (seed * 151 + 29) & 0xFF
    rng_b = (seed * 223 + 47) & 0xFF
    w, h = size
    img = Image.new("RGB", size, (rng_r, rng_g, rng_b))
    px = img.load()
    for i in range(min(w, h)):
        px[i, i] = ((rng_r + 80) & 0xFF, (rng_g + 80) & 0xFF, (rng_b + 80) & 0xFF)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = pybase64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render(
    consumer_url: str, model: str, image_urls: str | list[str], prompt: str
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
    r = SESSION.post(
        f"{consumer_url}/v1/chat/completions/render",
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def generate(
    base_url: str,
    rendered: dict,
    *,
    max_tokens: int,
    ec_transfer_params: dict | None = None,
) -> dict:
    sampling_params: dict = {"max_tokens": max_tokens, "temperature": 0.0}
    if ec_transfer_params is not None:
        sampling_params["extra_args"] = {"ec_transfer_params": ec_transfer_params}
    body = {
        "token_ids": rendered["token_ids"],
        "features": rendered.get("features"),
        "sampling_params": sampling_params,
    }
    r = SESSION.post(
        f"{base_url}/inference/v1/generate",
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def reset_mm_cache(base_url: str) -> None:
    r = SESSION.post(f"{base_url}/reset_mm_cache", timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()


def reset_encoder_cache(base_url: str) -> None:
    """Evict cached encoder outputs (the scheduler's EncoderCacheManager and
    the GPU model runner's encoder_cache dict).

    reset_mm_cache alone does NOT clear this — it only clears the
    multimodal-input-side cache. Without this, a later request for the same
    mm_hash gets a free local cache hit and never reaches the EC connector's
    has_cache_item/NIXL-fetch path at all.
    """
    r = SESSION.post(f"{base_url}/reset_encoder_cache", timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()


def reset_prefix_cache(base_url: str) -> None:
    r = SESSION.post(f"{base_url}/reset_prefix_cache", timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    # 200 with success=false means the reset was refused (blocks still held by a
    # just-finished request). Silently ignoring that leaves a populated prefix
    # cache behind, which makes later cache-miss assertions meaningless.
    if r.json().get("success") is False:
        raise AssertionError(
            f"reset_prefix_cache refused at {base_url}: blocks still held"
        )


def decode_tokens(model: str, token_ids: list[int]) -> str:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return f"<{len(token_ids)} tokens; install transformers to decode>"
    tok = AutoTokenizer.from_pretrained(model)
    return tok.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-test helpers
# ---------------------------------------------------------------------------


def _producer_encode(h, rendered: dict) -> tuple[dict, list[str]]:
    """Drive a producer encode on `rendered`, return (ec_transfer_params, events)."""
    mark = h.producer.events.mark()
    resp = generate(h.producer.base_url, rendered, max_tokens=1)
    # The save's GPU->mmap copy is confirmed a step or two after the response,
    # so wait for request_finished rather than reading straight through.
    events = h.producer.events.wait_all(
        mark, ["producer request_finished"], timeout_s=10.0
    )
    return resp.get("ec_transfer_params") or {}, events


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_baseline(h, image: Path, prompt: str) -> None:
    print("\n=== test_baseline ===")
    # DIAGNOSTIC: swapped from image_data_url(image) (hato.jpg, ~100MB
    # base64 tensor payload) to a small synthetic image to isolate whether
    # the multinode oc port-forward stall is payload-size-dependent. Revert
    # to `image_data_url(image)` once diagnosed.
    data_url = synth_image_data_url(seed=999)
    rendered = render(h.consumer.base_url, h.model, data_url, prompt)
    target_hash = rendered["features"]["mm_hashes"]["image"][0]
    print(f"  rendered: token_ids={len(rendered['token_ids'])}, mm_hash={target_hash}")

    cmark = h.consumer.events.mark()
    ctrl = generate(h.consumer.base_url, rendered, max_tokens=80)
    assert_event(
        h.consumer.events.read(cmark),
        "consumer ENCODER FORWARD",
        where="consumer events [control]",
    )
    print(
        f"  control: {decode_tokens(h.model, ctrl['choices'][0]['token_ids'] or [])!r}"
    )

    reset_mm_cache(h.consumer.base_url)

    reset_encoder_cache(h.consumer.base_url)
    reset_prefix_cache(h.consumer.base_url)

    ec_params, prod_sl = _producer_encode(h, rendered)
    assert_event(prod_sl, "producer ENCODER FORWARD", where="producer events [encode]")
    assert_event(
        prod_sl,
        f"producer save mm_hash={target_hash}",
        where="producer events [encode]",
    )
    if target_hash not in ec_params:
        raise AssertionError(
            f"producer response missing ec_transfer_params for {target_hash}; "
            f"got keys={list(ec_params.keys())}"
        )
    info = ec_params[target_hash]
    for key in ("peer_host", "peer_port", "size_bytes"):
        if key not in info:
            raise AssertionError(f"ec_transfer_params[{target_hash}] missing {key!r}")

    pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
    resp = generate(
        h.consumer.base_url, rendered, max_tokens=80, ec_transfer_params=ec_params
    )
    psl = h.producer.events.wait_all(
        pmark,
        [
            f"producer XferReq mm_hash={target_hash}",
            f"producer read granted mm_hash={target_hash}",
        ],
    )
    csl = h.consumer.events.wait_all(
        cmark,
        [f"consumer read ok mm_hash={target_hash}", "consumer load mm_hashes="],
    )
    assert_event(
        psl, f"producer XferReq mm_hash={target_hash}", where="producer events [ec]"
    )
    assert_event(
        psl,
        f"producer read granted mm_hash={target_hash}",
        where="producer events [ec]",
    )
    assert_event(
        csl,
        f"consumer read ok mm_hash={target_hash}",
        where="consumer events [ec]",
    )
    assert_event(csl, "consumer load mm_hashes=", where="consumer events [ec]")
    assert_no_event(csl, "consumer ENCODER FORWARD", where="consumer events [ec]")
    print(f"  EC: {decode_tokens(h.model, resp['choices'][0]['token_ids'] or [])!r}")
    print("  ✓ test_baseline")


def test_cache_reuse(h, prompt: str, n_repeat: int = 5) -> None:
    print(f"\n=== test_cache_reuse (n_repeat={n_repeat}) ===")
    rendered = render(
        h.consumer.base_url, h.model, synth_image_data_url(seed=50), prompt
    )
    target_hash = rendered["features"]["mm_hashes"]["image"][0]
    reset_mm_cache(h.consumer.base_url)
    reset_encoder_cache(h.consumer.base_url)
    reset_prefix_cache(h.consumer.base_url)
    ec_params, _ = _producer_encode(h, rendered)
    if target_hash not in ec_params:
        raise AssertionError(f"producer did not announce {target_hash}")

    pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
    for i in range(n_repeat):
        # Evict the GPU-resident encoder cache each iteration too, not just
        # the KV prefix cache. Otherwise EncoderCacheManager keeps serving
        # this mm_hash's already-loaded GPU tensor directly (nothing else
        # competes for its slot), short-circuiting past has_cache_item /
        # start_load_caches on every repeat after the first — the CPU-side
        # cache reuse this test exists to exercise never gets touched again.
        reset_encoder_cache(h.consumer.base_url)
        reset_prefix_cache(h.consumer.base_url)
        resp = generate(
            h.consumer.base_url, rendered, max_tokens=8, ec_transfer_params=ec_params
        )
        if not (resp["choices"][0]["token_ids"] or []):
            raise AssertionError(f"empty response on iter {i}")

    # Wait for the consumer side to reach its expected load count before
    # counting anything. Every XferReq necessarily precedes the load it serves,
    # so once the loads are all in, the producer's counts have settled too.
    csl = h.consumer.events.wait_count(cmark, "consumer load mm_hashes=", n_repeat)
    psl = h.producer.events.read(pmark)
    n_xfer_req = count_events(psl, f"producer XferReq mm_hash={target_hash}")
    n_read_granted = count_events(psl, "producer read granted")
    n_load = count_events(csl, "consumer load mm_hashes=")
    n_encoder = count_events(csl, "consumer ENCODER FORWARD")
    print(
        f"  XferReqs={n_xfer_req}, read grants={n_read_granted}, "
        f"loads={n_load}, encoder forwards={n_encoder}"
    )
    if n_xfer_req != 1:
        raise AssertionError(
            f"expected exactly 1 XferReq across {n_repeat} repeats, got {n_xfer_req}"
        )
    if n_read_granted != 1:
        raise AssertionError(
            f"expected exactly 1 read granted across {n_repeat} repeats, got "
            f"{n_read_granted}"
        )
    if n_load < n_repeat:
        raise AssertionError(f"expected ≥{n_repeat} consumer loads, got {n_load}")
    if n_encoder != 0:
        raise AssertionError(
            f"consumer should never re-encode but ENCODER FORWARD fired {n_encoder}×"
        )
    print(f"  ✓ test_cache_reuse — single fetch served {n_repeat} requests from cache")


def test_multi_image(h, prompt: str, n_images: int = 3) -> None:
    print(f"\n=== test_multi_image (n_images={n_images}) ===")
    urls = [synth_image_data_url(seed=100 + i) for i in range(n_images)]
    rendered = render(h.consumer.base_url, h.model, urls, prompt)
    hashes = rendered["features"]["mm_hashes"]["image"]
    if len(hashes) != n_images:
        raise AssertionError(f"expected {n_images} mm_hashes, got {len(hashes)}")
    print(f"  mm_hashes: {hashes}")
    reset_mm_cache(h.consumer.base_url)
    reset_encoder_cache(h.consumer.base_url)
    reset_prefix_cache(h.consumer.base_url)

    ec_params, prod_sl = _producer_encode(h, rendered)
    for hh in hashes:
        if hh not in ec_params:
            raise AssertionError(f"producer omitted ec_transfer_params for {hh}")
        assert_event(
            prod_sl,
            f"producer save mm_hash={hh}",
            where="producer events [multi-image encode]",
        )

    pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
    generate(h.consumer.base_url, rendered, max_tokens=20, ec_transfer_params=ec_params)
    psl = h.producer.events.wait_all(
        pmark,
        [f"producer XferReq mm_hash={hh}" for hh in hashes]
        + [f"producer read granted mm_hash={hh}" for hh in hashes],
    )
    csl = h.consumer.events.wait_all(
        cmark,
        [f"consumer read ok mm_hash={hh}" for hh in hashes]
        + ["consumer load mm_hashes="],
    )
    for hh in hashes:
        assert_event(
            psl,
            f"producer XferReq mm_hash={hh}",
            where="producer events [multi-image ec]",
        )
        assert_event(
            psl,
            f"producer read granted mm_hash={hh}",
            where="producer events [multi-image ec]",
        )
        assert_event(
            csl,
            f"consumer read ok mm_hash={hh}",
            where="consumer events [multi-image ec]",
        )
    assert_event(
        csl, "consumer load mm_hashes=", where="consumer events [multi-image ec]"
    )
    assert_no_event(
        csl, "consumer ENCODER FORWARD", where="consumer events [multi-image ec]"
    )
    print(f"  ✓ test_multi_image — all {n_images} fetched, none re-encoded")


def test_concurrent_ec(h, prompt: str, k: int = 4) -> None:
    print(f"\n=== test_concurrent_ec (k={k}) ===")
    encoded: list[tuple[dict, dict, str]] = []
    for i in range(k):
        url = synth_image_data_url(seed=200 + i)
        rendered = render(h.consumer.base_url, h.model, url, prompt)
        target_hash = rendered["features"]["mm_hashes"]["image"][0]
        ec_params, _ = _producer_encode(h, rendered)
        if target_hash not in ec_params:
            raise AssertionError(f"producer did not announce {target_hash}")
        encoded.append((rendered, ec_params, target_hash))
    print(f"  pre-encoded {k} images on producer")

    reset_mm_cache(h.consumer.base_url)

    reset_encoder_cache(h.consumer.base_url)
    reset_prefix_cache(h.consumer.base_url)

    pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=k) as ex:
        futs = [
            ex.submit(
                generate,
                h.consumer.base_url,
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
    # The k fetches are served one at a time, seconds apart, so the last one's
    # lines reach the log stream well after the final generate() returns.
    psl = h.producer.events.wait_all(
        pmark,
        [f"producer XferReq mm_hash={hh}" for _, _, hh in encoded]
        + [f"producer read granted mm_hash={hh}" for _, _, hh in encoded],
        timeout_s=30.0,
    )
    csl = h.consumer.events.wait_all(
        cmark,
        [f"consumer read ok mm_hash={hh}" for _, _, hh in encoded]
        + ["consumer load mm_hashes="],
        timeout_s=30.0,
    )
    for _, _, hh in encoded:
        assert_event(
            psl, f"producer XferReq mm_hash={hh}", where="producer events [concurrent]"
        )
        assert_event(
            psl,
            f"producer read granted mm_hash={hh}",
            where="producer events [concurrent]",
        )
        assert_event(
            csl,
            f"consumer read ok mm_hash={hh}",
            where="consumer events [concurrent]",
        )
    assert_event(csl, "consumer load mm_hashes=", where="consumer events [concurrent]")
    assert_no_event(
        csl, "consumer ENCODER FORWARD", where="consumer events [concurrent]"
    )
    print(f"  ✓ test_concurrent_ec — k={k} parallel fetches in {elapsed:.2f}s")


def test_pool_exhaustion(
    log_dir: Path,
    model: str,
    prompt: str,
    make_specs_fn: Callable,
    make_harness: Callable,
    pre_harness: Callable[[], None] | None = None,
) -> None:
    """Tiny producer pool forces LRU eviction. Consumer requesting an evicted
    mm_hash gets a NACK and falls back to local encode."""
    print("\n=== test_pool_exhaustion ===")
    if pre_harness:
        pre_harness()
    # FIXME(ported-connector): this test still assumes the old connector's log
    # output. Its eviction math below mixes bytes/blocks and its
    # "producer save ... n_blocks=" parse targets a log line the ported
    # connector no longer emits. Both need rework once the log contract is
    # settled. `pool_size` is now a small region byte budget to force eviction.
    pool_size = 4 * 1024**2
    producer, consumer = make_specs_fn(log_dir, producer_ec_cpu_bytes=pool_size)
    with make_harness(producer, consumer, model) as h:
        rendered_a = render(
            h.consumer.base_url, h.model, synth_image_data_url(seed=300), prompt
        )
        hash_a = rendered_a["features"]["mm_hashes"]["image"][0]
        a_params, sl_a = _producer_encode(h, rendered_a)
        if hash_a not in a_params:
            raise AssertionError("producer did not announce A's encoding")
        m = re.search(
            rf"producer save mm_hash={hash_a} n_blocks=(\d+)", "\n".join(sl_a)
        )
        if not m:
            saves = [e for e in sl_a if "producer save" in e]
            raise AssertionError(
                "could not parse n_blocks from the producer's save event.\n"
                f"  save events seen: {saves or '<none>'}\n"
                f"  all events: {sl_a!r}"
            )
        per_image_blocks = int(m.group(1))
        # `pool_size` is a byte budget but `per_image_blocks` is a block
        # count; converting one to the other needs the server's internal
        # block_size_bytes, which isn't exposed to this test. A fixed small
        # count is fast and, for the tiny synthetic images used here, always
        # exceeds the tiny pool_size above — the assertion below still fails
        # loudly (with a "try lowering pool_size or raising n_extra" hint) if
        # that ever stops being true for a given model/image size.
        n_extra = 20
        print(
            f"  per_image_blocks={per_image_blocks}, pool={pool_size}, "
            f"encoding {n_extra} more images to force eviction"
        )
        for i in range(n_extra):
            rendered = render(
                h.consumer.base_url,
                h.model,
                synth_image_data_url(seed=301 + i),
                prompt,
            )
            _producer_encode(h, rendered)

        reset_mm_cache(h.consumer.base_url)

        reset_encoder_cache(h.consumer.base_url)
        reset_prefix_cache(h.consumer.base_url)
        pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
        resp = generate(
            h.consumer.base_url, rendered_a, max_tokens=8, ec_transfer_params=a_params
        )
        if not (resp["choices"][0]["token_ids"] or []):
            raise AssertionError("evicted-fallback request returned empty body")
        # Waiting for the XferReq also gives a (wrongly) granted read time to
        # show up, which makes the absence check below stricter, not weaker.
        psl = h.producer.events.wait_all(pmark, [f"producer XferReq mm_hash={hash_a}"])
        csl = h.consumer.events.wait_all(
            cmark,
            [f"consumer read failed mm_hash={hash_a}", "consumer ENCODER FORWARD"],
        )

        assert_event(
            psl,
            f"producer XferReq mm_hash={hash_a}",
            where="producer events [evicted-fetch]",
        )
        # Substring match per event, not `needle in psl`: against a list that
        # would silently degrade to exact equality, so any change to the event's
        # wording would turn this absence check into an unconditional pass.
        if any(f"producer read granted mm_hash={hash_a}" in e for e in psl):
            raise AssertionError(
                "expected no read granted for evicted hash; A may not have been "
                "evicted (try lowering pool_size or raising n_extra)"
            )
        assert_event(
            csl,
            f"consumer read failed mm_hash={hash_a}",
            where="consumer events [evicted-fetch]",
        )
        assert_event(
            csl, "consumer ENCODER FORWARD", where="consumer events [evicted-fetch]"
        )
        print("  ✓ test_pool_exhaustion — evicted hash NACK'd and locally re-encoded")


def test_producer_restart(
    log_dir: Path,
    model: str,
    prompt: str,
    make_specs_fn: Callable,
    make_harness: Callable,
    pre_harness: Callable[[], None] | None = None,
) -> None:
    """Producer restart while consumer is still up. The consumer detects the
    fresh NIXL metadata in the XferAck and re-registers the producer via
    register_source, logging 'consumer peer_pool REPLACE'."""
    print("\n=== test_producer_restart ===")
    if pre_harness:
        pre_harness()
    producer, consumer = make_specs_fn(log_dir)
    with make_harness(producer, consumer, model) as h:
        rendered_a = render(
            h.consumer.base_url, h.model, synth_image_data_url(seed=400), prompt
        )
        hash_a = rendered_a["features"]["mm_hashes"]["image"][0]
        reset_mm_cache(h.consumer.base_url)
        reset_encoder_cache(h.consumer.base_url)
        reset_prefix_cache(h.consumer.base_url)
        ec_params_a, _ = _producer_encode(h, rendered_a)
        if hash_a not in ec_params_a:
            raise AssertionError("producer did not announce hash A on first encode")
        resp_a = generate(
            h.consumer.base_url,
            rendered_a,
            max_tokens=8,
            ec_transfer_params=ec_params_a,
        )
        if not (resp_a["choices"][0]["token_ids"] or []):
            raise AssertionError("pre-restart EC request returned empty body")
        print("  pre-restart EC fetch OK")

        h.restart_producer()

        rendered_b = render(
            h.consumer.base_url, h.model, synth_image_data_url(seed=401), prompt
        )
        hash_b = rendered_b["features"]["mm_hashes"]["image"][0]
        reset_mm_cache(h.consumer.base_url)
        reset_encoder_cache(h.consumer.base_url)
        reset_prefix_cache(h.consumer.base_url)
        ec_params_b, _ = _producer_encode(h, rendered_b)
        if hash_b not in ec_params_b:
            raise AssertionError("producer did not announce hash B on second encode")

        pmark, cmark = h.producer.events.mark(), h.consumer.events.mark()
        resp_b = generate(
            h.consumer.base_url,
            rendered_b,
            max_tokens=8,
            ec_transfer_params=ec_params_b,
        )
        if not (resp_b["choices"][0]["token_ids"] or []):
            raise AssertionError("post-restart EC request returned empty body")
        psl = h.producer.events.wait_all(
            pmark,
            [
                f"producer XferReq mm_hash={hash_b}",
                f"producer read granted mm_hash={hash_b}",
            ],
            timeout_s=12.0,
        )
        # "consumer load" is the last of the three events asserted below, so
        # waiting on it settles "read ok" and "peer_pool" as well.
        csl = h.consumer.events.wait_all(
            cmark,
            ["consumer load mm_hashes="],
            timeout_s=12.0,
        )
        assert_event(
            psl,
            f"producer XferReq mm_hash={hash_b}",
            where="producer events [post-restart ec]",
        )
        assert_event(
            psl,
            f"producer read granted mm_hash={hash_b}",
            where="producer events [post-restart ec]",
        )
        assert_event(
            csl,
            f"consumer read ok mm_hash={hash_b}",
            where="consumer events [post-restart ec]",
        )
        assert_event(
            csl, "consumer load mm_hashes=", where="consumer events [post-restart ec]"
        )
        # After restart the old peer may have been evicted by poll_dead_peers
        # before the EC request fires (logs "ADD") or still be present when the
        # fresh XferAck arrives with different metadata (logs "REPLACE"). Either
        # way "consumer peer_pool" appears, confirming a fresh NIXL registration.
        assert_event(
            csl,
            "consumer peer_pool",
            where="consumer events [post-restart ec]",
        )
        assert_no_event(
            csl, "consumer ENCODER FORWARD", where="consumer events [post-restart ec]"
        )
        print(
            "  ✓ test_producer_restart — post-restart EC fetch OK with fresh metadata"
        )
