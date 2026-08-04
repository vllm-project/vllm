# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses and CLI-arg parsing for ``lmcache bench engine``."""

# Standard
from dataclasses import dataclass
import argparse
import json
import os
import urllib.error
import urllib.request

# Third Party
from openai import OpenAI

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_GB = 1024**3


@dataclass
class EngineBenchConfig:
    """Top-level config produced from CLI args, interactive mode, or saved config.

    Contains only general benchmark parameters. Workload-specific configs
    (e.g., ``LongDocQAConfig``) live in their respective workload modules
    and are resolved by the workload factory.
    """

    engine_url: str
    model: str
    workload: str
    kv_cache_volume_gb: float
    tokens_per_gb_kvcache: int
    seed: int
    output_dir: str
    export_csv: bool
    export_json: bool
    quiet: bool
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if not self.engine_url:
            raise ValueError("engine_url must be non-empty")
        if self.kv_cache_volume_gb <= 0:
            raise ValueError(
                f"kv_cache_volume_gb must be positive, got {self.kv_cache_volume_gb}"
            )
        if self.tokens_per_gb_kvcache <= 0:
            raise ValueError(
                f"tokens_per_gb_kvcache must be positive, "
                f"got {self.tokens_per_gb_kvcache}"
            )


def auto_detect_model(engine_url: str) -> str:
    """Fetch the first model ID from the engine's ``/v1/models`` endpoint.

    Args:
        engine_url: Base URL of the inference engine (e.g.,
            ``http://localhost:8000``).

    Returns:
        The model ID string.

    Raises:
        RuntimeError: If the engine is unreachable or returns no models.
    """
    base_url = engine_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    logger.debug("Auto-detecting model from %s/models", base_url)

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        models = client.models.list()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch models from {base_url}/models: {e}") from e

    if not models.data:
        raise RuntimeError(
            f"No models returned by {base_url}/models; pass --model explicitly."
        )

    model_id = models.data[0].id
    logger.debug("Auto-detected model: %s", model_id)
    return model_id


def _fetch_lmcache_status(lmcache_url: str) -> dict:
    """Fetch ``/status`` from the LMCache HTTP server.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If the server is unreachable.
    """
    url = lmcache_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    status_url = f"{url}/status"

    logger.debug("Fetching LMCache status from %s", status_url)

    try:
        req = urllib.request.Request(status_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"Cannot connect to LMCache server at {status_url}: {e}"
        ) from e


def _find_model_meta(
    gpu_meta: dict,
    model_name: str,
) -> dict:
    """Find the GPU metadata entry matching *model_name*.

    Args:
        gpu_meta: The ``cache_context_meta`` dict from ``/status``.
        model_name: Model name to match.

    Returns:
        The matching GPU metadata dict.

    Raises:
        RuntimeError: If no entry matches *model_name*.
    """
    for meta in gpu_meta.values():
        if meta.get("model_name") == model_name:
            return meta

    available = sorted({m.get("model_name", "?") for m in gpu_meta.values()})
    raise RuntimeError(
        f"Model {model_name!r} not found on LMCache server. "
        f"Available: {', '.join(available)}"
    )


def resolve_tokens_per_gb(lmcache_url: str, model_name: str) -> int:
    """Query the LMCache server and compute tokens per GB of KV cache.

    Fetches ``/status``, finds the model entry matching
    *model_name*, and computes::

        global_bytes_per_token = cache_size_per_token * world_size
        tokens_per_gb = (1024**3) // global_bytes_per_token

    ``cache_size_per_token`` is rank-local, so it must be multiplied
    by ``world_size`` for tensor-parallel models.

    Args:
        lmcache_url: URL of the LMCache HTTP server.
        model_name: Model name to look up (must match a model served
            by the LMCache server).

    Returns:
        tokens_per_gb_kvcache value.

    Raises:
        RuntimeError: If the server is unreachable, the model is not
            found, or the layout is missing required fields.
    """
    data = _fetch_lmcache_status(lmcache_url)

    gpu_meta = data.get("cache_context_meta", {})
    if not gpu_meta:
        # CB-only deployments (engine_type="blend") populate
        # cb_gpu_context_meta instead of cache_context_meta.
        gpu_meta = data.get("cb_gpu_context_meta", {})
    if not gpu_meta:
        raise RuntimeError(
            "No model info returned by LMCache server; "
            "is the server running with a model loaded?"
        )

    meta = _find_model_meta(gpu_meta, model_name)
    layout = meta.get("kv_cache_layout")
    if not layout:
        raise RuntimeError(f"No kv_cache_layout for model {model_name!r}")

    cache_size_per_token = layout.get("cache_size_per_token")
    if cache_size_per_token is None:
        raise RuntimeError(
            f"cache_size_per_token not available for model "
            f"{model_name!r}; is the LMCache server up to date?"
        )

    world_size = meta.get("world_size", 1)
    global_bytes_per_token = cache_size_per_token * world_size
    tokens_per_gb = _GB // global_bytes_per_token

    logger.info(
        "Resolved from LMCache: model=%s, "
        "cache_size_per_token=%d bytes (rank-local), "
        "world_size=%d -> %d bytes/token (global) -> %d tokens/GB",
        model_name,
        cache_size_per_token,
        world_size,
        global_bytes_per_token,
        tokens_per_gb,
    )
    return tokens_per_gb


def parse_args_to_config(args: argparse.Namespace) -> EngineBenchConfig:
    """Convert parsed CLI arguments into a fully-resolved EngineBenchConfig.

    Handles model auto-detection and tokens-per-GB resolution from the
    LMCache server when ``--lmcache-url`` is provided.

    Args:
        args: Parsed argparse Namespace from the bench engine subcommand.

    Returns:
        A fully-resolved EngineBenchConfig.
    """
    model = args.model if args.model else auto_detect_model(args.engine_url)

    tokens_per_gb = args.tokens_per_gb_kvcache
    if tokens_per_gb is None:
        lmcache_url = getattr(args, "lmcache_url", None)
        if lmcache_url is not None:
            tokens_per_gb = resolve_tokens_per_gb(lmcache_url, model)
        else:
            raise ValueError(
                "--tokens-per-gb-kvcache is required when --lmcache-url is not set"
            )

    return EngineBenchConfig(
        engine_url=args.engine_url,
        model=model,
        workload=args.workload,
        kv_cache_volume_gb=args.kv_cache_volume,
        tokens_per_gb_kvcache=tokens_per_gb,
        seed=args.seed,
        output_dir=args.output_dir,
        export_csv=not args.no_csv,
        export_json=args.json,
        quiet=args.quiet,
        ignore_eos=args.ignore_eos,
    )
