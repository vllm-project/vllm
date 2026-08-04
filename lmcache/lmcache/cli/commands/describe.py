# SPDX-License-Identifier: Apache-2.0
"""``lmcache describe`` — show detailed status of a running LMCache service.

Usage::

    lmcache describe kvcache --url http://localhost:8000
"""

# Standard
from typing import TYPE_CHECKING
import argparse
import json
import sys
import urllib.error
import urllib.request

# Third Party
from prometheus_client.parser import text_string_to_metric_families

# First Party
from lmcache.cli.commands.base import BaseCommand

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.metrics import Metrics

# Default server URLs per describe target (ZMQ/HTTP semantics differ).
DEFAULT_URLS: dict[str, str] = {
    "kvcache": "http://localhost:8080",
    "engine": "http://localhost:8000",
}

# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------


class DescribeError(Exception):
    """Raised when the describe command cannot fetch or parse status data."""


def normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def fetch_json(url: str, timeout: int = 10) -> dict:
    """GET *url* and return the parsed JSON body.

    Raises:
        DescribeError: On network/HTTP errors.
    """
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            body = exc.read().decode()
            try:
                detail = json.loads(body).get("error", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise DescribeError(f"Server unhealthy: {detail}") from exc
        raise DescribeError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DescribeError(f"Cannot connect to {url}: {exc.reason}") from exc
    except OSError as exc:
        raise DescribeError(f"Cannot connect to {url}: {exc}") from exc


def fetch_health(url: str, timeout: int = 10) -> bool:
    """Return whether *url* responds with HTTP 200.

    A lightweight liveness probe for endpoints (e.g. the vLLM ``/health``
    route) that return an empty body rather than JSON, so :func:`fetch_json`
    cannot be used.

    Args:
        url: Full health-check URL to GET.
        timeout: Socket timeout in seconds.

    Returns:
        ``True`` if the server responds with HTTP 200, ``False`` on any
        non-200 status or connection error.
    """
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url), timeout=timeout
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def fetch_running_requests(url: str, timeout: int = 10) -> int | None:
    """Return the number of in-flight requests from a vLLM ``/metrics`` page.

    Parses the Prometheus ``vllm:num_requests_running`` gauge, summing the
    value across all reported series (e.g. one per model). This is best
    effort: the metric is informational, so any failure to fetch or parse
    degrades to ``None`` (rendered as ``N/A``) rather than raising.

    Args:
        url: Full ``/metrics`` URL to GET.
        timeout: Socket timeout in seconds.

    Returns:
        The total running-request count, or ``None`` if the endpoint is
        unreachable or the metric is absent (e.g. metrics disabled or an
        unsupported engine version).
    """
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url), timeout=timeout
        ) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, OSError):
        return None

    total = 0.0
    found = False
    try:
        for family in text_string_to_metric_families(text):
            if family.name != "vllm:num_requests_running":
                continue
            for sample in family.samples:
                total += sample.value
                found = True
    except ValueError:
        # Malformed exposition text; treat as unavailable.
        return None
    return int(total) if found else None


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def fmt_health(is_healthy: object) -> str | None:
    """Format a boolean health flag as ``'OK'`` / ``'UNHEALTHY'``."""
    if is_healthy is None:
        return None
    return "OK" if is_healthy else "UNHEALTHY"


def safe_get(data: dict, *keys, default=None):  # type: ignore[type-arg]
    """Walk nested dicts by *keys*, returning *default* on any miss."""
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


# -------------------------------------------------------------------
# KVCache describer
# -------------------------------------------------------------------


class KVCacheDescriber:
    """Builds the ``describe kvcache`` output from a ``/status`` response.

    Each ``add_*`` method populates one logical section. The orchestrating
    :meth:`describe` calls them in order and emits the result.  Adding a
    new section is a one-method change — no other code needs to know
    about it.
    """

    def __init__(self, metrics: "Metrics", data: dict, base_url: str) -> None:
        self.metrics = metrics
        self.data = data
        self.base_url = base_url

    def describe(self) -> None:
        """Run all section builders and emit."""
        self.add_overview()
        self.add_l1_storage()
        self.add_models()
        self.add_l2_adapters()
        self.metrics.emit()

    # -- sections --------------------------------------------------------

    def add_overview(self) -> None:
        """Top-level engine overview."""
        self.metrics.add("health", "Health", fmt_health(self.data.get("is_healthy")))
        self.metrics.add("url", "URL", self.base_url)
        self.metrics.add("engine_type", "Engine type", self.data.get("engine_type"))
        self.metrics.add("chunk_size", "Chunk size", self.data.get("chunk_size"))

    def add_l1_storage(self) -> None:
        """L1 cache capacity, usage, eviction, and object count."""
        total_bytes = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_total_bytes"
        )
        if total_bytes is not None:
            self.metrics.add(
                "l1_capacity_gb",
                "L1 capacity (GB)",
                round(total_bytes / (1024**3), 2),
            )
        else:
            self.metrics.add("l1_capacity_gb", "L1 capacity (GB)", None)

        used_bytes = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_used_bytes"
        )
        usage_ratio = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_usage_ratio"
        )
        if used_bytes is not None and usage_ratio is not None:
            gb = used_bytes / (1024**3)
            pct = usage_ratio * 100
            self.metrics.add("l1_used_gb", "L1 used (GB)", f"{gb:.2f} ({pct:.1f}%)")
        else:
            self.metrics.add("l1_used_gb", "L1 used (GB)", None)

        self.metrics.add(
            "eviction_policy",
            "Eviction policy",
            safe_get(
                self.data,
                "storage_manager",
                "eviction_controller",
                "eviction_policy",
            ),
        )
        self.metrics.add(
            "cached_objects",
            "Cached objects",
            safe_get(self.data, "storage_manager", "l1_manager", "total_object_count"),
        )
        self.metrics.add(
            "active_sessions", "Active sessions", self.data.get("active_sessions")
        )

    def add_models(self) -> None:
        """Per-model KV cache layout sections.

        Each model gets one section with context-wide fields, followed by
        one ``kernel_groups`` list entry per kernel group carrying that
        group's identity and geometry.
        """
        gpu_meta = self.data.get("cache_context_meta", {})
        if not gpu_meta:
            return

        # Deduplicate by (model_name, world_size) — multiple GPU IDs
        # may share the same model.
        seen: dict[tuple[str, int], dict] = {}
        for gpu_id, meta in gpu_meta.items():
            key = (meta["model_name"], meta["world_size"])
            if key not in seen:
                seen[key] = {
                    "gpu_ids": [],
                    "layout": meta.get("kv_cache_layout"),
                }
            seen[key]["gpu_ids"].append(gpu_id)

        for idx, ((model_name, world_size), info) in enumerate(seen.items()):
            section_key = f"model_{idx}"
            self.metrics.add_list_section("models", section_key, f"Model: {model_name}")
            sec = self.metrics[section_key]
            sec.add("model", "Model", model_name)
            sec.add("world_size", "World size", world_size)
            sec.add("gpu_ids", "GPU IDs", ", ".join(info["gpu_ids"]))

            layout = info.get("layout")
            if not layout:
                continue
            for _key, _label in (
                ("num_layers", "Num layers"),
                ("num_blocks", "Num blocks"),
                ("cache_size_per_token", "Cache size per token (bytes)"),
            ):
                if _key in layout:
                    sec.add(_key, _label, layout[_key])

            self._add_kernel_groups(idx, model_name, layout.get("kernel_groups", []))

    def _add_kernel_groups(
        self, model_idx: int, model_name: str, kernel_groups: list
    ) -> None:
        """Emit one ``kernel_groups`` list section per kernel group.

        Args:
            model_idx: Index of the owning model section (keeps section keys
                unique across models).
            model_name: Human-readable model name, shown in each group header.
            kernel_groups: The model layout's ``kernel_groups`` list (each a
                dict produced by ``GPUCacheContext.report_status``).
        """
        for group in kernel_groups:
            kg_idx = group.get("kernel_group_idx")
            section_key = f"model_{model_idx}_kg_{kg_idx}"
            self.metrics.add_list_section(
                "kernel_groups",
                section_key,
                f"Kernel group {kg_idx} ({model_name})",
            )
            sec = self.metrics[section_key]
            sec.add("model", "Model", model_name)
            for _key, _label in (
                ("kernel_group_idx", "Kernel group index"),
                ("engine_group_idx", "Engine group index"),
                ("object_group_idx", "Object group index"),
                ("num_layers", "Num layers"),
                ("tokens_per_block", "Tokens per block"),
                ("slots_per_block", "Slots per block"),
                ("dtype", "Dtype"),
                ("is_mla", "MLA"),
                ("attention_backend", "Attention backend"),
                ("engine_kv_shape", "Engine KV shape"),
                ("engine_kv_concrete_shape", "Engine KV tensor shape"),
            ):
                if _key in group:
                    sec.add(_key, _label, group[_key])

    def add_l2_adapters(self) -> None:
        """L2 adapter sections."""
        l2_adapters = safe_get(self.data, "storage_manager", "l2_adapters") or []
        for idx, adapter in enumerate(l2_adapters):
            adapter_type = adapter.get("type", "Unknown")
            section_key = f"l2_{idx}"
            self.metrics.add_list_section(
                "l2_adapters", section_key, f"L2: {adapter_type}"
            )
            sec = self.metrics[section_key]
            sec.add("type", "Type", adapter_type)
            sec.add("health", "Health", fmt_health(adapter.get("is_healthy")))

            if "backend" in adapter:
                sec.add("backend", "Backend", adapter["backend"])
            if "base_path" in adapter:
                sec.add("base_path", "Base path", adapter["base_path"])
            if "stored_object_count" in adapter:
                sec.add(
                    "stored_object_count",
                    "Stored objects",
                    adapter["stored_object_count"],
                )

            cap = adapter.get("max_capacity_bytes")
            used = adapter.get("current_size_bytes")
            if cap is not None and used is not None:
                pct = used / cap * 100 if cap > 0 else 0.0
                sec.add(
                    "used",
                    "Used",
                    f"{fmt_bytes(used)} / {fmt_bytes(cap)} ({pct:.1f}%)",
                )

            pool_size = adapter.get("pool_size")
            pool_free = adapter.get("pool_free_slots")
            if pool_size is not None and pool_free is not None:
                pool_used = pool_size - pool_free
                pct = pool_used / pool_size * 100 if pool_size > 0 else 0.0
                sec.add(
                    "pool_used",
                    "Pool used",
                    f"{pool_used} / {pool_size} ({pct:.1f}%)",
                )


# -------------------------------------------------------------------
# Engine describer
# -------------------------------------------------------------------


class EngineDescriber:
    """Builds the ``describe engine`` output from vLLM server responses.

    Reads model identity and context window from the engine's
    ``/v1/models`` response and combines them with a ``/health`` liveness
    result to render a concise status view.
    """

    def __init__(
        self,
        metrics: "Metrics",
        models_data: dict,
        is_healthy: bool,
        running_requests: int | None,
        base_url: str,
    ) -> None:
        self.metrics = metrics
        self.models_data = models_data
        self.is_healthy = is_healthy
        self.running_requests = running_requests
        self.base_url = base_url

    def describe(self) -> None:
        """Run all section builders and emit."""
        self.add_overview()
        self.metrics.emit()

    def add_overview(self) -> None:
        """Model identity, context window, health status, and load."""
        model = self._first_model()
        self.metrics.add("model", "Model", model.get("id") if model else None)
        self.metrics.add(
            "max_context",
            "Max context (tokens)",
            model.get("max_model_len") if model else None,
        )
        self.metrics.add("status", "Status", fmt_health(self.is_healthy))
        self.metrics.add("running_requests", "Running requests", self.running_requests)

    def _first_model(self) -> dict | None:
        """Return the first model entry from a ``/v1/models`` response."""
        data = self.models_data.get("data") or []
        return data[0] if data else None


# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------


class DescribeCommand(BaseCommand):
    """Show detailed status of a running LMCache service."""

    def name(self) -> str:
        return "describe"

    def help(self) -> str:
        return "Show detailed status of a running LMCache service."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            choices=["kvcache", "engine"],
            help="What to describe.",
        )
        parser.add_argument(
            "--url",
            default=None,
            help=(
                "Server URL (default: http://localhost:8080 for kvcache, "
                "http://localhost:8000 for engine)."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        if getattr(args, "url", None) is None:
            args.url = DEFAULT_URLS[args.target]
        if args.target == "kvcache":
            self._describe_kvcache(args)
        elif args.target == "engine":
            self._describe_engine(args)

    def _describe_kvcache(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        try:
            data = fetch_json(f"{base_url}/status")
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        metrics = self.create_metrics("LMCache KV Cache Service", args, width=50)
        KVCacheDescriber(metrics, data, base_url).describe()

    def _describe_engine(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        try:
            models = fetch_json(f"{base_url}/v1/models")
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        is_healthy = fetch_health(f"{base_url}/health")
        running_requests = fetch_running_requests(f"{base_url}/metrics")
        metrics = self.create_metrics("Inference Engine", args, width=50)
        EngineDescriber(
            metrics, models, is_healthy, running_requests, base_url
        ).describe()
