# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers and test functions for the CPU EC connector e2e suite.

Imported by both test_cpu_ec_e2e.py (single-pod) and
test_cpu_ec_multinode.py (OpenShift multi-node).
"""

from __future__ import annotations

import concurrent.futures
import io
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
    num_ec_blocks: int = 80000


# ---------------------------------------------------------------------------
# Health check (shared by LocalHarness and K8sHarness)
# ---------------------------------------------------------------------------


def wait_for_health(port: int, proc, timeout_s: int) -> None:
    """Poll /health until the server responds, the process dies, or timeout.

    `proc` may be None (K8s mode — no local process to death-check).
    """
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
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


# ---------------------------------------------------------------------------
# Log windowing
# ---------------------------------------------------------------------------


def log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def log_slice(path: Path, start: int) -> str:
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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


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
    consumer_port: int, model: str, image_urls: str | list[str], prompt: str
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
        f"http://127.0.0.1:{port}/reset_mm_cache", timeout=REQUEST_TIMEOUT_S
    )
    r.raise_for_status()


def reset_prefix_cache(port: int) -> None:
    r = requests.post(
        f"http://127.0.0.1:{port}/reset_prefix_cache", timeout=REQUEST_TIMEOUT_S
    )
    r.raise_for_status()


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


def _producer_encode(h, rendered: dict) -> tuple[dict, str]:
    """Drive a producer encode on `rendered`, return (ec_transfer_params, log_slice)."""
    mark = log_size(h.producer.log_path)
    resp = generate(h.producer.http_port, rendered, max_tokens=1)
    sl = log_slice(h.producer.log_path, mark)
    return resp.get("ec_transfer_params") or {}, sl


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_baseline(h, image: Path, prompt: str) -> None:
    print("\n=== test_baseline ===")
    data_url = image_data_url(image)
    rendered = render(h.consumer.http_port, h.model, data_url, prompt)
    target_hash = rendered["features"]["mm_hashes"]["image"][0]
    print(f"  rendered: token_ids={len(rendered['token_ids'])}, mm_hash={target_hash}")

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

    pmark, cmark = log_size(h.producer.log_path), log_size(h.consumer.log_path)
    resp = generate(
        h.consumer.http_port, rendered, max_tokens=80, ec_transfer_params=ec_params
    )
    psl = log_slice(h.producer.log_path, pmark)
    csl = log_slice(h.consumer.log_path, cmark)
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


def test_cache_reuse(h, prompt: str, n_repeat: int = 5) -> None:
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
            f"expected exactly 1 NIXL WRITE across {n_repeat} repeats, got "
            f"{n_nixl_write}"
        )
    if n_load < n_repeat:
        raise AssertionError(f"expected ≥{n_repeat} consumer loads, got {n_load}")
    if n_encoder != 0:
        raise AssertionError(
            f"consumer should never re-encode but ENCODER FORWARD fired {n_encoder}×"
        )
    print(f"  ✓ test_cache_reuse — single fetch served {n_repeat} requests via _loaded")


def test_multi_image(h, prompt: str, n_images: int = 3) -> None:
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
    generate(
        h.consumer.http_port, rendered, max_tokens=20, ec_transfer_params=ec_params
    )
    psl = log_slice(h.producer.log_path, pmark)
    csl = log_slice(h.consumer.log_path, cmark)
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


def test_concurrent_ec(h, prompt: str, k: int = 4) -> None:
    print(f"\n=== test_concurrent_ec (k={k}) ===")
    encoded: list[tuple[dict, dict, str]] = []
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
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=k) as ex:
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
    psl = log_slice(h.producer.log_path, pmark)
    csl = log_slice(h.consumer.log_path, cmark)
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
    pool_size = 400
    producer, consumer = make_specs_fn(log_dir, producer_num_ec_blocks=pool_size)
    with make_harness(producer, consumer, model) as h:
        rendered_a = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=300), prompt
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
                prompt,
            )
            _producer_encode(h, rendered)

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


def test_producer_restart(
    log_dir: Path,
    model: str,
    prompt: str,
    make_specs_fn: Callable,
    make_harness: Callable,
    pre_harness: Callable[[], None] | None = None,
) -> None:
    """Producer restart while consumer is still up. The consumer's cached
    nixl_agent_metadata for the producer is stale; on first contact after
    restart, _get_or_add_peer must detect the change and swap in a fresh agent."""
    print("\n=== test_producer_restart ===")
    if pre_harness:
        pre_harness()
    producer, consumer = make_specs_fn(log_dir)
    with make_harness(producer, consumer, model) as h:
        rendered_a = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=400), prompt
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

        rendered_b = render(
            h.consumer.http_port, h.model, synth_image_data_url(seed=401), prompt
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
        psl = log_slice(h.producer.log_path, pmark)
        csl = log_slice(h.consumer.log_path, cmark)
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
