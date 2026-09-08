#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the image pool and request sequence for the EC offload benchmark.

Emits a `custom_image` JSONL that `vllm bench serve --disable-shuffle` replays
in order, plus a `manifest.json` describing what the sequence *should* cost.
The manifest is what makes the server's numbers checkable: it states how many
distinct images must be encoded, how many references are reuses, and how many
bytes the working set occupies in the CPU region.

Two properties are load-bearing and neither is cosmetic:

  * Text comes BEFORE the image in every request, with a unique nonce. Image
    first would let the KV prefix cache serve a repeat outright, so the encoder
    output would never be wanted and there would be nothing to reload.
  * Consecutive requests draw from DIFFERENT size buckets (`--interleave-sizes`,
    on by default). Same-size allocations recycle a freed run intact, so a
    size-sorted sequence churns the region heavily while barely fragmenting it
    -- producing a clean fragmentation result that means nothing.

`--rounds N` replays the whole pool N times for the fragmentation arm, and
forces every request to carry an image: a text-only request would drop a pool
image out of that round, so the churn would no longer be uniform.

Photos are real and JPEG-encoded because both wire payload and server-side
decode land inside the TTFT being measured. Sources:

    dir:/path/to/photos          any directory PIL can read
    hf-tar:owner/repo[:file]     stream a .tar.gz from an HF repo, stop early

Example:

    python gen_workload.py --photo-source dir:/data/photos \
        --out-dir /vllm-workspace/bench/wl --pool-size 128 --rounds 6
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

# Qwen2-VL family: patch_size * spatial_merge_size. One embedding per 28x28 px.
DEFAULT_MERGE_STRIDE = 28
# Qwen2.5-VL-7B: hidden 3584 x 2 bytes (bfloat16) per embedding.
DEFAULT_HIDDEN_DIM = 3584
DEFAULT_ELEMENT_SIZE = 2

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_QUESTION = "Describe this image in one short sentence."


class Bucket(NamedTuple):
    width: int
    height: int
    weight: float


class PoolImage(NamedTuple):
    path: Path
    width: int
    height: int
    embeds: int  # encoder outputs this image occupies
    nbytes: int  # bytes it occupies in the CPU region
    upscale: float = 1.0  # linear factor applied to the source, 1.0 = none


def parse_buckets(spec: str) -> list[Bucket]:
    """Parse `1288x728:0.6,896x896:0.3` into normalized buckets."""
    buckets = []
    for part in spec.split(","):
        dims, _, weight = part.strip().partition(":")
        w, _, h = dims.lower().partition("x")
        buckets.append(Bucket(int(w), int(h), float(weight or 1.0)))
    total = sum(b.weight for b in buckets)
    return [Bucket(b.width, b.height, b.weight / total) for b in buckets]


def embeds_for(width: int, height: int, stride: int) -> int:
    return (width // stride) * (height // stride)


# ---------------------------------------------------------------------------
# Photo sources. Both yield decoded images lazily so a pool of 128 never pulls
# more of a remote archive than it needs.
# ---------------------------------------------------------------------------


def _iter_dir(path: Path) -> Iterator[tuple[str, object]]:
    from PIL import Image

    for entry in sorted(path.rglob("*")):
        if entry.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            img = Image.open(entry)
            img.load()
        except Exception as exc:  # unreadable file: skip, keep going
            print(f"[gen] skipping {entry.name}: {exc}", file=sys.stderr)
            continue
        yield entry.name, img


def _iter_hf_tar(repo: str, filename: str | None) -> Iterator[tuple[str, object]]:
    """Stream a tar.gz out of an HF repo, decoding members as they arrive.

    The generator is abandoned once the pool is full, which closes the response
    and leaves the rest of the archive undownloaded.
    """
    import tarfile

    import requests
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    from PIL import Image

    api = HfApi()
    if filename is None:
        info = api.repo_info(repo, repo_type="model")
        archives = sorted(
            s.rfilename for s in info.siblings if s.rfilename.endswith(".tar.gz")
        )
        if not archives:
            raise SystemExit(f"[gen] no .tar.gz in {repo}")
        filename = archives[0]
        print(f"[gen] streaming {repo}/{filename}")

    url = hf_hub_url(repo, filename, repo_type="model")
    resp = requests.get(url, headers=build_hf_headers(), stream=True, timeout=120)
    if resp.status_code in (401, 403):
        raise SystemExit(
            f"[gen] {repo} denied access ({resp.status_code}). If it is gated, "
            "accept its terms on huggingface.co with the account whose token "
            "this process resolves to. Note HF_TOKEN in the environment "
            "overrides a stored `hf auth login`, so a login can look like a "
            "no-op: run with `env -u HF_TOKEN` to use the stored token."
        )
    resp.raise_for_status()
    with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not member.name.lower().endswith(_IMAGE_SUFFIXES):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                img = Image.open(io.BytesIO(handle.read()))
                img.load()
            except Exception as exc:
                print(f"[gen] skipping {member.name}: {exc}", file=sys.stderr)
                continue
            yield Path(member.name).name, img


def iter_source_images(source: str) -> Iterator[tuple[str, object]]:
    scheme, _, rest = source.partition(":")
    if scheme == "dir":
        return _iter_dir(Path(rest).expanduser())
    if scheme == "hf-tar":
        repo, _, filename = rest.partition(":")
        return _iter_hf_tar(repo, filename or None)
    raise SystemExit(f"[gen] unknown --photo-source scheme {scheme!r}")


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


def build_pool(
    source: str,
    buckets: list[Bucket],
    pool_size: int,
    out_dir: Path,
    quality: int,
    stride: int,
    bytes_per_embed: int,
    rng: random.Random,
    allow_upscale: bool = False,
    min_source_px: int = 0,
) -> list[PoolImage]:
    """Crop and resize real photos into a fixed pool, one file per pool slot.

    By default a source photo is only used for a bucket it already covers.
    `allow_upscale` lifts that, which is what makes resolutions above the
    dataset's own range reachable: every quantity this benchmark measures --
    embeddings per image, cache entry bytes, transform cost, vision-tower time --
    is a function of pixel count, not of detail, so a Lanczos-enlarged photo
    exercises them identically to a native one of that size. Measured on this
    dataset, it also matches a native crop's compressed size (0.54 MB at
    2048x2048, q85, for both), so the wire payload stays representative too.

    `min_source_px` keeps a thumbnail from being blown up many times over.
    """
    from PIL import Image

    pool_dir = out_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    wanted = rng.choices(buckets, weights=[b.weight for b in buckets], k=pool_size)

    pool: list[PoolImage] = []
    source_iter = iter_source_images(source)
    skipped_small = 0
    for idx, bucket in enumerate(wanted):
        img = None
        for _, candidate in source_iter:
            covers = (
                candidate.width >= bucket.width and candidate.height >= bucket.height
            )
            big_enough_to_enlarge = (
                allow_upscale and candidate.width * candidate.height >= min_source_px
            )
            if covers or big_enough_to_enlarge:
                img = candidate
                break
            skipped_small += 1
            candidate.close()
        if img is None:
            raise SystemExit(
                f"[gen] ran out of source photos at pool slot {idx}: need "
                f">= {bucket.width}x{bucket.height} ({skipped_small} were too small)"
            )
        # Center-crop to the bucket's aspect, then downscale to exact size.
        scale = max(bucket.width / img.width, bucket.height / img.height)
        crop_w = int(round(bucket.width / scale))
        crop_h = int(round(bucket.height / scale))
        left = (img.width - crop_w) // 2
        top = (img.height - crop_h) // 2
        out = img.convert("RGB").crop((left, top, left + crop_w, top + crop_h))
        out = out.resize((bucket.width, bucket.height), Image.LANCZOS)
        path = pool_dir / f"{bucket.width}x{bucket.height}_{idx:04d}.jpg"
        out.save(path, format="JPEG", quality=quality)
        img.close()
        embeds = embeds_for(bucket.width, bucket.height, stride)
        pool.append(
            PoolImage(
                path,
                bucket.width,
                bucket.height,
                embeds,
                embeds * bytes_per_embed,
                round(max(scale, 1.0), 3),
            )
        )
    upscales = [p.upscale for p in pool]
    total_mb = sum(p.path.stat().st_size for p in pool) / 1e6
    megapixels = sum(p.width * p.height for p in pool) / 1e6
    print(
        f"[gen] pool: {len(pool)} photos, {skipped_small} source photos rejected, "
        f"{total_mb:.1f} MB on disk ({total_mb / megapixels:.2f} MB/MP)"
    )
    print(
        f"[gen] enlargement factor: min {min(upscales):.2f} median "
        f"{sorted(upscales)[len(upscales) // 2]:.2f} max {max(upscales):.2f}"
    )
    return pool


# ---------------------------------------------------------------------------
# Request sequence
# ---------------------------------------------------------------------------


def _nonce_text(rng: random.Random, approx_tokens: int) -> str:
    """Pseudo-random filler, unique per request.

    Length is approximate -- one short pseudo-word is roughly one token, but
    the true count depends on the tokenizer, so treat `--prefix-tokens` as a
    target rather than a guarantee.
    """
    words = [
        "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(3, 7)))
        for _ in range(max(1, approx_tokens))
    ]
    return " ".join(words)


def _interleaved(pool_indices: list[int], pool: list[PoolImage]) -> list[int]:
    """Reorder so consecutive entries come from different size buckets."""
    by_size: dict[tuple[int, int], list[int]] = {}
    for i in pool_indices:
        by_size.setdefault((pool[i].width, pool[i].height), []).append(i)
    queues = list(by_size.values())
    out: list[int] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def build_sequence(
    pool: list[PoolImage],
    *,
    rounds: int,
    num_requests: int,
    reuse: str,
    mm_fraction: float,
    multi_image_fraction: float,
    images_per_request: int,
    prefix_tokens: int,
    interleave_sizes: bool,
    rng: random.Random,
) -> tuple[list[dict], list[list[int]]]:
    """Return `(jsonl_records, per_request_image_indices)`.

    `rounds > 0` replays the whole pool once per round -- deterministic churn
    for the fragmentation arm. Otherwise `num_requests` are drawn according to
    `--reuse`, which is the shape a serving workload actually has.
    """
    if rounds > 0:
        # Every request carries its image: this mode exists to churn the region
        # a fixed number of times, and a text-only request would silently drop
        # a pool image out of the round.
        mm_fraction = 1.0
        picks: list[list[int]] = []
        for _ in range(rounds):
            order = list(range(len(pool)))
            order = _interleaved(order, pool) if interleave_sizes else order
            picks.extend([i] for i in order)
    else:
        if reuse.startswith("zipf"):
            _, _, exponent = reuse.partition(":")
            a = float(exponent or 1.1)
            weights = [1.0 / (i + 1) ** a for i in range(len(pool))]
        elif reuse == "uniform":
            weights = [1.0] * len(pool)
        elif reuse == "none":
            weights = None
        else:
            raise SystemExit(f"[gen] unknown --reuse {reuse!r}")
        picks = []
        for n in range(num_requests):
            if weights is None:
                picks.append([n % len(pool)])
            else:
                picks.append(rng.choices(range(len(pool)), weights=weights, k=1))

    records: list[dict] = []
    per_request: list[list[int]] = []
    for i, chosen in enumerate(picks):
        nonce = f"[req {i} n{rng.getrandbits(48):012x}] " + _nonce_text(
            rng, prefix_tokens
        )
        if rng.random() >= mm_fraction:
            records.append({"content": [{"type": "text", "text": nonce}]})
            per_request.append([])
            continue
        # Fan-out width: the proxy issues one encode call per image, so the
        # images in a request is the ceiling on how many encoders that single
        # request can occupy at once. A width-N request cannot use an (N+1)th
        # encoder, which is what makes a scaling curve flatten at N.
        width = min(images_per_request, len(pool))
        if width > 1 and rng.random() < multi_image_fraction:
            while len(chosen) < width:
                candidate = rng.randrange(len(pool))
                if candidate not in chosen:
                    chosen.append(candidate)
        content: list[dict] = [{"type": "text", "text": nonce}]
        for idx in chosen:
            content.append(
                {"type": "image_url", "image_url": {"url": str(pool[idx].path)}}
            )
        content.append({"type": "text", "text": _QUESTION})
        records.append({"content": content})
        per_request.append(chosen)
    return records, per_request


def _count_histogram(per_request: list[list[int]]) -> dict[str, int]:
    """How many requests carried how many images."""
    hist: dict[str, int] = {}
    for chosen in per_request:
        key = str(len(chosen))
        hist[key] = hist.get(key, 0) + 1
    return hist


def build_manifest(
    pool: list[PoolImage],
    per_request: list[list[int]],
    *,
    bytes_per_embed: int,
    args: argparse.Namespace,
) -> dict:
    refs: dict[int, int] = {}
    for chosen in per_request:
        for idx in chosen:
            refs[idx] = refs.get(idx, 0) + 1
    total_refs = sum(refs.values())
    distinct = len(refs)
    working_set = sum(pool[i].nbytes for i in refs)
    histogram: dict[str, int] = {}
    for count in refs.values():
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    return {
        "args": {k: str(v) for k, v in vars(args).items()},
        "bytes_per_embed": bytes_per_embed,
        "pool": [
            {
                "path": str(p.path),
                "w": p.width,
                "h": p.height,
                "embeds": p.embeds,
                "region_bytes": p.nbytes,
                "jpeg_bytes": p.path.stat().st_size,
                "upscale": p.upscale,
            }
            for p in pool
        ],
        "sequence": {
            "images_per_request": args.images_per_request,
            "image_count_histogram": _count_histogram(per_request),
            "requests": len(per_request),
            "mm_requests": sum(1 for c in per_request if c),
            "text_only_requests": sum(1 for c in per_request if not c),
            "image_references": total_refs,
            "distinct_images_used": distinct,
            "reference_histogram": histogram,
        },
        # Upper bound on what the connector can serve: every reference after
        # the first is reloadable IF the region holds the entry and the GPU
        # encoder cache has evicted it. A GPU-cache hit serves the request
        # without consulting the connector, so measured EC loads land at or
        # below this.
        "expected": {
            "first_encodes": distinct,
            "reuses": total_refs - distinct,
            "max_hit_rate": round((total_refs - distinct) / total_refs, 4)
            if total_refs
            else 0.0,
            "working_set_bytes": working_set,
            # Every image of one request must hold cache space at once, so this
            # is the floor for the consumer's encoder_cache_size; below it the
            # request is split across steps, which confounds a fan-out
            # measurement with chunking.
            "max_embeds_per_request": max(
                (sum(pool[i].embeds for i in chosen) for chosen in per_request),
                default=0,
            ),
            "suggested_ec_cpu_bytes": int(working_set * 1.25),
            "fragmentation_arm_ec_cpu_bytes": int(working_set * 0.5),
        },
    }


def self_check(jsonl_path: Path, expected: int) -> None:
    """Load the emitted file through vLLM's own dataset class.

    Catches a schema mistake here instead of twenty minutes into a run.
    """
    try:
        from vllm.benchmarks.datasets.datasets import CustomImageDataset
    except ImportError as exc:
        raise SystemExit(
            f"[gen] --self-check needs an interpreter that can import vllm ({exc})"
        ) from exc

    dataset = CustomImageDataset(dataset_path=str(jsonl_path), disable_shuffle=True)
    dataset.load_data()
    assert len(dataset.data) == expected, (
        f"dataset loaded {len(dataset.data)} lines, expected {expected}"
    )
    for line_no, item in enumerate(dataset.data, start=1):
        parts = CustomImageDataset._validate_content_parts(item["content"])
        assert parts[0]["type"] == "text", f"line {line_no}: image is not text-preceded"
    print(f"[gen] self-check OK: {expected} lines load through CustomImageDataset")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--photo-source", default="hf-tar:ofsoundof/LSDIR")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--buckets", default="1288x728:0.6,896x896:0.3,448x448:0.1")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument(
        "--allow-upscale",
        action="store_true",
        help="enlarge sources that do not already cover the bucket, so "
        "resolutions above the dataset's own range are reachable",
    )
    p.add_argument(
        "--min-source-mp",
        type=float,
        default=0.8,
        help="with --allow-upscale, smallest source accepted, in megapixels",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="replay the whole pool this many times (fragmentation arm); "
        "0 means sample --num-requests according to --reuse",
    )
    p.add_argument("--num-requests", type=int, default=400)
    p.add_argument("--reuse", default="zipf:1.1", help="zipf:A | uniform | none")
    p.add_argument("--mm-fraction", type=float, default=0.8)
    p.add_argument(
        "--multi-image-fraction",
        type=float,
        default=0.15,
        help="fraction of image-bearing requests that carry the full "
        "--images-per-request width; the rest carry one",
    )
    p.add_argument(
        "--images-per-request",
        type=int,
        default=2,
        help="fan-out width: images in a full-width request. The proxy issues "
        "one encode call per image, so this caps how many encoder instances a "
        "single request can use in parallel",
    )
    p.add_argument("--prefix-tokens", type=int, default=32)
    p.add_argument(
        "--interleave-sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="consecutive requests use different size buckets, so the region "
        "actually fragments as it churns",
    )
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--element-size", type=int, default=DEFAULT_ELEMENT_SIZE)
    p.add_argument("--merge-stride", type=int, default=DEFAULT_MERGE_STRIDE)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--reuse-pool",
        action="store_true",
        help="keep an existing pool/ directory instead of rebuilding",
    )
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    buckets = parse_buckets(args.buckets)
    bytes_per_embed = args.hidden_dim * args.element_size
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.json"
    if args.reuse_pool and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        pool = [
            PoolImage(
                Path(e["path"]), e["w"], e["h"], e["embeds"], e["region_bytes"],
                e.get("upscale", 1.0),
            )
            for e in previous["pool"]
        ]
        print(f"[gen] reusing existing pool of {len(pool)} photos")
    else:
        pool = build_pool(
            args.photo_source,
            buckets,
            args.pool_size,
            args.out_dir,
            args.jpeg_quality,
            args.merge_stride,
            bytes_per_embed,
            rng,
            args.allow_upscale,
            int(args.min_source_mp * 1e6),
        )

    records, per_request = build_sequence(
        pool,
        rounds=args.rounds,
        num_requests=args.num_requests,
        reuse=args.reuse,
        mm_fraction=args.mm_fraction,
        multi_image_fraction=args.multi_image_fraction,
        images_per_request=args.images_per_request,
        prefix_tokens=args.prefix_tokens,
        interleave_sizes=args.interleave_sizes,
        rng=rng,
    )

    jsonl_path = args.out_dir / "workload.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    manifest = build_manifest(
        pool, per_request, bytes_per_embed=bytes_per_embed, args=args
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    seq, exp = manifest["sequence"], manifest["expected"]
    print(
        f"[gen] {jsonl_path}: {seq['requests']} requests "
        f"({seq['mm_requests']} mm, {seq['text_only_requests']} text-only), "
        f"{seq['image_references']} image refs over "
        f"{seq['distinct_images_used']} distinct images"
    )
    print(
        f"[gen] fan-out width {args.images_per_request}, images per request "
        f"{seq['image_count_histogram']}, max embeds in one request "
        f"{exp['max_embeds_per_request']} (consumer encoder_cache_size must be "
        f"at least this)"
    )
    print(
        f"[gen] working set {exp['working_set_bytes'] / 1024**3:.2f} GiB, "
        f"max hit rate {exp['max_hit_rate'] * 100:.1f}%, "
        f"suggested ec_cpu_bytes {exp['suggested_ec_cpu_bytes']}, "
        f"fragmentation arm {exp['fragmentation_arm_ec_cpu_bytes']}"
    )

    if args.self_check:
        self_check(jsonl_path, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
