# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: lazy media decode vs eager decode frontend cost.

Quantifies the per-request frontend savings of the lazy media decode change
(`LazyMedia` in vllm/multimodal/media/base.py): media bytes are hashed for
the mm cache without decoding, and only cache misses pay the decode cost.

Three paths are measured with the same code the connector/processor use:

- eager (old behavior): MediaIO.load_bytes + MultiModalHasher.hash_kwargs
- lazy cache hit:       MediaIO.load_bytes_lazy + hash_kwargs (no decode)
- lazy cache miss:      load_bytes_lazy + hash_kwargs + decode()

Run from the repo root:
    python benchmarks/benchmark_lazy_decode.py
"""

import argparse
import gc
import statistics
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Benchmarks import the vllm package from the enclosing source tree, not from
# whatever is installed in the active environment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.media.audio import AudioMediaIO
from vllm.multimodal.media.base import LazyMedia
from vllm.multimodal.media.connector import global_thread_pool
from vllm.multimodal.media.image import ImageMediaIO

HASHER_ALGORITHM = "blake3"  # MultiModalConfig.mm_hasher_algorithm default

IMAGE_SIZES = [(512, 512), (1024, 1024), (2048, 2048), (3840, 2160)]
IMAGE_FORMATS = ["JPEG", "PNG"]
HIT_RATES = [0.5, 0.9, 0.99]


def bench(fn, warmup: int, repeat: int) -> tuple[float, float]:
    """Return (mean, std) of fn() wall time in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(samples), statistics.stdev(samples)


def repeats_for(num_pixels: int) -> tuple[int, int]:
    """(warmup, repeat) scaled down for large inputs."""
    if num_pixels <= 512 * 512:
        return 3, 50
    if num_pixels <= 1024 * 1024:
        return 3, 30
    if num_pixels <= 2048 * 2048:
        return 2, 15
    return 2, 8


def make_image_bytes(width: int, height: int, fmt: str, seed: int) -> bytes:
    """Deterministic random-noise image (worst case for compression)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels, "RGB").save(buf, fmt)
    return buf.getvalue()


def image_seed(width: int, height: int, fmt: str) -> int:
    return zlib.crc32(f"{width}x{height}.{fmt}".encode())


def get_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found")


# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------


def eager_image_path(image_io: ImageMediaIO, data: bytes) -> None:
    media = image_io.load_bytes(data)
    MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=media)


def hit_image_path(image_io: ImageMediaIO, data: bytes) -> bool:
    """Return True if hashing secretly rasterized the header image.

    PngImageFile.getexif() calls self.load() when the PNG carries no "exif"
    chunk (a trailing eXIf chunk can only be found by reading to EOF), so the
    hasher's EXIF probe decodes the pixels even though LazyMedia.is_decoded
    stays False.
    """
    lazy = image_io.load_bytes_lazy(data)
    MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=lazy)
    assert not lazy.is_decoded
    header = lazy.header_image
    # Image.im raises when unloaded in Pillow 12; probe the private attribute.
    hidden_decode = header is not None and getattr(header, "_im", None) is not None
    lazy.release_bytes()
    return hidden_decode


def miss_image_path(image_io: ImageMediaIO, data: bytes) -> None:
    lazy = image_io.load_bytes_lazy(data)
    MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=lazy)
    lazy.decode()
    lazy.release_bytes()


def bench_image_paths(
    image_io: ImageMediaIO, data: bytes, warmup: int, repeat: int
) -> tuple[float, float, float, float, float, float, bool]:
    """Time the eager / lazy-hit / lazy-miss paths for one image payload."""
    t_eager, s_eager = bench(lambda: eager_image_path(image_io, data), warmup, repeat)
    hidden_decode = False

    def hit():
        nonlocal hidden_decode
        hidden_decode = hit_image_path(image_io, data) or hidden_decode

    t_hit, s_hit = bench(hit, warmup, repeat)
    t_miss, s_miss = bench(lambda: miss_image_path(image_io, data), warmup, repeat)
    return t_eager, s_eager, t_hit, s_hit, t_miss, s_miss, hidden_decode


def bench_images() -> dict[tuple, dict[str, float]]:
    print("=" * 88)
    print("IMAGES: per-request frontend cost (mean +/- std, ms)")
    print("=" * 88)
    header = (
        f"{'size':>11} {'fmt':>5} {'bytes':>9} {'eager(old)':>18} "
        f"{'lazy hit':>14} {'lazy miss':>14} {'hit speedup':>11}"
    )
    print(header)
    print("-" * len(header))

    results: dict[tuple, dict[str, float]] = {}
    image_io = ImageMediaIO()  # default image_mode="RGB"
    for width, height in IMAGE_SIZES:
        for fmt in IMAGE_FORMATS:
            data = make_image_bytes(
                width, height, fmt, seed=image_seed(width, height, fmt)
            )
            warmup, repeat = repeats_for(width * height)

            # Correctness: eager and lazy hashing must agree.
            eager_hash = MultiModalHasher.hash_kwargs(
                HASHER_ALGORITHM, image=image_io.load_bytes(data)
            )
            lazy = image_io.load_bytes_lazy(data)
            lazy_hash = MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=lazy)
            assert eager_hash == lazy_hash, f"hash mismatch {width}x{height} {fmt}"

            t_eager, s_eager, t_hit, s_hit, t_miss, s_miss, hidden = bench_image_paths(
                image_io, data, warmup, repeat
            )

            decoded = np.asarray(image_io.load_bytes(data).media)
            key = (width, height, fmt)
            results[key] = dict(
                eager=t_eager,
                hit=t_hit,
                miss=t_miss,
                encoded_bytes=len(data),
                decoded_nbytes=decoded.nbytes,
            )
            note = " *" if hidden else ""
            print(
                f"{f'{width}x{height}':>11} {fmt:>5} {len(data) / 1e6:>8.2f}M "
                f"{t_eager:>9.2f} +/-{s_eager:>5.2f} "
                f"{t_hit:>7.2f} +/-{s_hit:>5.2f} "
                f"{t_miss:>7.2f} +/-{s_miss:>5.2f} "
                f"{t_eager / t_hit:>9.1f}x{note}"
            )
    if any(
        hit_image_path(ImageMediaIO(), make_image_bytes(w, h, f, image_seed(w, h, f)))
        for (w, h, f) in [(512, 512, fmt) for fmt in IMAGE_FORMATS]
    ):
        print(
            " * = hashing the lazy item triggered a hidden pixel decode: "
            "PngImageFile.getexif() calls load() when the PNG has no 'exif' "
            "chunk (a trailing eXIf is only findable by reading to EOF). The "
            "PNG hit-path numbers above therefore include a full decode."
        )

    print()
    print("Blended per-request cost at cache hit rate p:  new = p*hit + (1-p)*miss")
    header = f"{'size':>11} {'fmt':>5} {'old (eager)':>12}" + "".join(
        f" {f'new p={p}':>10} {'saved':>7}" for p in HIT_RATES
    )
    print(header)
    print("-" * len(header))
    for (width, height, fmt), r in results.items():
        row = f"{f'{width}x{height}':>11} {fmt:>5} {r['eager']:>11.2f} "
        for p in HIT_RATES:
            blended = p * r["hit"] + (1 - p) * r["miss"]
            row += f" {blended:>10.2f} {1 - blended / r['eager']:>6.0%}"
        print(row)
    print()
    return results


# ---------------------------------------------------------------------------
# Audio paths
# ---------------------------------------------------------------------------


def make_wav_bytes(seconds: float, sample_rate: int = 16000, seed: int = 0) -> bytes:
    import soundfile

    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * rng.standard_normal(t.shape)
    buf = BytesIO()
    soundfile.write(
        buf,
        (tone + noise).astype(np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buf.getvalue()


def bench_audio_paths(
    audio_io: AudioMediaIO, data: bytes, warmup: int, repeat: int
) -> tuple[float, float, float, float, float, float]:
    def eager():
        media = audio_io.load_bytes(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, audio=media)

    def hit():
        lazy = audio_io.load_bytes_lazy(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, audio=lazy)
        assert not lazy.is_decoded
        lazy.release_bytes()

    def miss():
        lazy = audio_io.load_bytes_lazy(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, audio=lazy)
        lazy.decode()
        lazy.release_bytes()

    t_eager, s_eager = bench(eager, warmup, repeat)
    t_hit, s_hit = bench(hit, warmup, repeat)
    t_miss, s_miss = bench(miss, warmup, repeat)
    return t_eager, s_eager, t_hit, s_hit, t_miss, s_miss


def bench_audio() -> None:
    print("=" * 88)
    print("AUDIO (16 kHz mono WAV): per-request frontend cost (mean +/- std, ms)")
    print("=" * 88)
    header = (
        f"{'duration':>9} {'bytes':>9} {'eager(old)':>18} "
        f"{'lazy hit':>14} {'lazy miss':>14} {'hit speedup':>11}"
    )
    print(header)
    print("-" * len(header))

    audio_io = AudioMediaIO()
    for seconds in (1.0, 3.0, 10.0):
        data = make_wav_bytes(seconds)
        warmup, repeat = 3, 30

        eager_hash = MultiModalHasher.hash_kwargs(
            HASHER_ALGORITHM, audio=audio_io.load_bytes(data)
        )
        lazy_hash = MultiModalHasher.hash_kwargs(
            HASHER_ALGORITHM, audio=audio_io.load_bytes_lazy(data)
        )
        # Eager hashes the decoded float32 waveform, lazy hashes the encoded
        # WAV bytes; the digests differ by design (both are stable per input).
        del eager_hash, lazy_hash

        t_eager, s_eager, t_hit, s_hit, t_miss, s_miss = bench_audio_paths(
            audio_io, data, warmup, repeat
        )
        print(
            f"{seconds:>7.0f}s {len(data) / 1e6:>8.2f}M "
            f"{t_eager:>9.2f} +/-{s_eager:>5.2f} "
            f"{t_hit:>7.2f} +/-{s_hit:>5.2f} "
            f"{t_miss:>7.2f} +/-{s_miss:>5.2f} "
            f"{t_eager / t_hit:>10.1f}x"
        )
    print(
        "\nNote: eager hashes the decoded float32 PCM (4 B/sample) while lazy\n"
        "hashes the encoded WAV (2 B/sample), so the hit path also halves the\n"
        "bytes fed to the hasher."
    )
    print()


# ---------------------------------------------------------------------------
# Video paths
# ---------------------------------------------------------------------------


def bench_video() -> None:
    print("=" * 88)
    print("VIDEO (2 s, 30 fps, 640x480 mp4): per-request frontend cost (ms)")
    print("=" * 88)
    try:
        import cv2
    except ImportError:
        print("SKIP: opencv (cv2) not available\n")
        return

    # The eager video item is MediaWithBytes wrapping a (frames, metadata)
    # tuple, which the hasher cannot serialize structurally, so it falls back
    # to pickling the decoded frames (pre-existing behavior). Silence the
    # per-call warning and note it once here.
    import logging

    logging.getLogger("vllm.multimodal.hasher").setLevel(logging.ERROR)
    print("note: eager-path video hashing falls back to pickling the decoded")
    print("      frames (hasher has no serializer for MediaWithBytes[tuple]);")
    print("      the lazy path hashes the encoded bytes instead.")

    rng = np.random.default_rng(0)
    path = "/tmp/vllm_bench_lazy_decode.mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    if not writer.isOpened():
        print("SKIP: cv2.VideoWriter cannot write mp4 on this system\n")
        return
    for _ in range(60):
        writer.write(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    writer.release()
    with open(path, "rb") as f:
        data = f.read()

    from vllm.multimodal.media.video import VideoMediaIO

    video_io = VideoMediaIO(ImageMediaIO())  # default num_frames=32, opencv
    warmup, repeat = 1, 5

    def eager():
        media = video_io.load_bytes(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, video=media)

    def hit():
        lazy = video_io.load_bytes_lazy(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, video=lazy)
        assert not lazy.is_decoded
        lazy.release_bytes()

    def miss():
        lazy = video_io.load_bytes_lazy(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, video=lazy)
        lazy.decode()
        lazy.release_bytes()

    t_eager, s_eager = bench(eager, warmup, repeat)
    t_hit, s_hit = bench(hit, warmup, repeat)
    t_miss, s_miss = bench(miss, warmup, repeat)
    print(f"encoded size: {len(data) / 1e6:.2f} MB")
    print(f"eager(old) : {t_eager:8.2f} +/- {s_eager:5.2f}")
    print(f"lazy hit   : {t_hit:8.2f} +/- {s_hit:5.2f}  ({t_eager / t_hit:.1f}x)")
    print(f"lazy miss  : {t_miss:8.2f} +/- {s_miss:5.2f}")
    print()


# ---------------------------------------------------------------------------
# Hash throughput vs decode cost
# ---------------------------------------------------------------------------


def bench_hash_throughput(image_results: dict[tuple, dict[str, float]]) -> None:
    print("=" * 88)
    print(f"HASH THROUGHPUT ({HASHER_ALGORITHM}) vs decode cost")
    print("=" * 88)
    header = f"{'payload':>8} {'hash ms':>10} {'GB/s':>7}"
    print(header)
    print("-" * len(header))

    def hash_payload(data: bytes) -> None:
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, data=data)

    for size_mb in (1, 8, 32):
        payload = np.random.default_rng(0).bytes(size_mb * 1024 * 1024)
        t_hash, _ = bench(lambda payload=payload: hash_payload(payload), 2, 10)
        gbs = size_mb / 1024 / (t_hash / 1e3)
        print(f"{size_mb:>6}MB {t_hash:>10.3f} {gbs:>7.2f}")
    print()

    print("Hash vs full eager decode for the image matrix above:")
    header = (
        f"{'size':>11} {'fmt':>5} {'bytes':>9} {'hash ms':>9} "
        f"{'decode ms':>10} {'decode/hash':>11}"
    )
    print(header)
    print("-" * len(header))
    image_io = ImageMediaIO()
    for (width, height, fmt), r in image_results.items():
        data = make_image_bytes(width, height, fmt, seed=image_seed(width, height, fmt))
        lazy = image_io.load_bytes_lazy(data)

        def hash_lazy(lazy=lazy):
            MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=lazy)

        warmup, repeat = repeats_for(width * height)
        t_hash, _ = bench(hash_lazy, warmup, repeat)
        t_decode = r["eager"] - t_hash  # eager = header + decode + ~same hash
        print(
            f"{f'{width}x{height}':>11} {fmt:>5} {len(data) / 1e6:>8.2f}M "
            f"{t_hash:>9.3f} {t_decode:>10.2f} {t_decode / t_hash:>10.0f}x"
        )
    print()


# ---------------------------------------------------------------------------
# Parallel decode
# ---------------------------------------------------------------------------


def bench_parallel_decode() -> None:
    print("=" * 88)
    print(
        f"PARALLEL DECODE: 8 x 2048x2048 JPEG, "
        f"global_thread_pool ({global_thread_pool._max_workers} workers)"
    )
    print("=" * 88)
    image_io = ImageMediaIO()
    payloads = [make_image_bytes(2048, 2048, "JPEG", seed=i) for i in range(8)]

    def serial():
        for data in payloads:
            image_io.load_bytes_lazy(data).decode()

    def parallel(pool):
        lazies = [image_io.load_bytes_lazy(data) for data in payloads]
        list(pool.map(LazyMedia.decode, lazies))

    t_serial, _ = bench(serial, 1, 5)
    t_pool, _ = bench(lambda: parallel(global_thread_pool), 1, 5)
    with ThreadPoolExecutor(max_workers=1) as single:
        t_pool1, _ = bench(lambda: parallel(single), 1, 5)
    print(f"serial decode in-process     : {t_serial:8.2f} ms")
    print(f"thread pool (1 worker)       : {t_pool1:8.2f} ms")
    print(f"global_thread_pool (8 workers): {t_pool:8.2f} ms")
    print(f"wall-time speedup            : {t_serial / t_pool:8.2f}x")
    print()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def bench_memory(image_results: dict[tuple, dict[str, float]]) -> None:
    print("=" * 88)
    print("MEMORY: encoded bytes held vs decoded RGB pixels")
    print("=" * 88)
    header = f"{'size':>11} {'fmt':>5} {'encoded':>10} {'decoded RGB':>12} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for (width, height, fmt), r in image_results.items():
        enc, dec = r["encoded_bytes"], r["decoded_nbytes"]
        print(
            f"{f'{width}x{height}':>11} {fmt:>5} {enc / 1e6:>9.2f}M "
            f"{dec / 1e6:>11.2f}M {dec / enc:>6.1f}x"
        )
    print(
        "\nOn a cache hit a LazyMedia keeps only the encoded bytes (released\n"
        "after hashing); the eager path keeps encoded bytes + decoded pixels.\n"
    )

    # Measured: process RSS for 100 retained cache-hit items, eager vs lazy.
    n = 100
    image_io = ImageMediaIO()
    data = make_image_bytes(1024, 1024, "JPEG", seed=7)

    gc.collect()
    rss0 = get_rss_mb()
    eager_items = []
    for _ in range(n):
        media = image_io.load_bytes(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=media)
        eager_items.append(media)
    gc.collect()
    rss_eager = get_rss_mb()
    del eager_items
    gc.collect()

    rss1 = get_rss_mb()
    lazy_items = []
    for _ in range(n):
        lazy = image_io.load_bytes_lazy(data)
        MultiModalHasher.hash_kwargs(HASHER_ALGORITHM, image=lazy)
        lazy.release_bytes()
        lazy_items.append(lazy)
    gc.collect()
    rss_lazy = get_rss_mb()
    del lazy_items
    gc.collect()

    print(
        f"RSS for {n} retained 1024x1024 JPEG cache-hit items "
        f"({len(data) / 1e3:.0f} KB encoded each):"
    )
    print(f"  eager (MediaWithBytes kept): +{rss_eager - rss0:8.1f} MB")
    print(f"  lazy  (bytes released)     : +{rss_lazy - rss1:8.1f} MB")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-video", action="store_true", help="skip the video section"
    )
    args = parser.parse_args()

    print(f"hasher algorithm: {HASHER_ALGORITHM}")
    print(f"Pillow {Image.__version__}, numpy {np.__version__}\n")

    image_results = bench_images()
    bench_audio()
    if not args.skip_video:
        bench_video()
    bench_hash_throughput(image_results)
    bench_parallel_decode()
    bench_memory(image_results)


if __name__ == "__main__":
    main()
