# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the MP-mode observability stack.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import argparse
import uuid

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.event_bus import EventBus

# First Party
from lmcache.v1.mp_observability.gc_monitor import GCMonitorConfig
from lmcache.v1.mp_observability.subscribers.logging.lookup_hash import (
    LookupHashLogConfig,
)


@dataclass
class ObservabilityConfig:
    """Unified configuration for the EventBus-based observability system.

    Controls the EventBus, OTel metrics/tracing pipelines, and subscriber
    registration.
    """

    enabled: bool = True
    """Master switch for the EventBus."""

    max_queue_size: int = 10_000
    """Maximum events in the EventBus queue before tail-drop."""

    metrics_enabled: bool = True
    """Register metrics subscribers (OTel counters / histograms)."""

    logging_enabled: bool = True
    """Register logging subscribers."""

    tracing_enabled: bool = False
    """Register span subscribers (OTel traces)."""

    otlp_endpoint: str | None = None
    """OTLP gRPC endpoint (e.g. ``http://localhost:4317``).  When set,
    metrics and traces are pushed to an OTel collector.  When ``None``,
    metrics fall back to an in-process Prometheus ``/metrics`` endpoint."""

    prometheus_port: int = 9090
    """Port for the Prometheus /metrics endpoint.  Only used when
    ``otlp_endpoint`` is ``None`` (Prometheus pull fallback)."""

    metrics_sample_rate: float = 0.01
    """Fraction of chunks/blocks to track for lifecycle histograms (0, 1.0].
    Counters always count all events regardless of this setting."""

    lookup_hash_log: LookupHashLogConfig = field(default_factory=LookupHashLogConfig)
    """Configuration for lookup hash file logging.  Disabled by default
    (empty ``output_dir``)."""

    gc_monitor: GCMonitorConfig = field(default_factory=GCMonitorConfig)
    """Configuration for CPython garbage-collection timing.  Disabled by
    default.  The monitor logs directly rather than publishing events, so it
    is independent of every other flag here including :attr:`enabled`."""

    extra_logging_enabled: bool = False
    """Register the periodic extra-stats logging subscriber (opt-in):
    per-GPU L0<->L1 store/retrieve throughput and token counts at INFO."""

    extra_logging_interval: float = 10.0
    """Seconds between extra-stats log flushes."""

    trace_level: str | None = None
    """If set, enables trace recording at the given level.  Currently
    only ``"storage"`` is supported.  See
    :mod:`lmcache.v1.mp_observability.trace` for details."""

    trace_output: str | None = None
    """Path to write the trace file.  When :attr:`trace_level` is set
    but this is ``None``, a timestamped path under ``$TMPDIR`` is
    minted and logged at INFO."""

    service_instance_id: str | None = None
    """OTel ``service.instance.id`` resource attribute, attached to every
    metric and span. There is no CLI flag for it: the operator-facing id is
    ``--instance-id`` (``MPServerConfig.instance_id``), which ``run_cache_server``
    projects onto this attribute so telemetry and coordinator membership share
    one id.

    ``None`` (the default) means "not set": standalone callers that build an
    ``ObservabilityConfig`` directly (e.g. the trace CLI driver) fall back to a
    random UUID v4 at ``init_observability`` time. An explicit value is
    preserved verbatim."""


DEFAULT_OBSERVABILITY_CONFIG = ObservabilityConfig(enabled=False)


def add_observability_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add observability configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with observability arguments added.
    """
    group = parser.add_argument_group(
        "Observability", "Configuration for metrics, logging, and tracing"
    )
    group.add_argument(
        "--disable-observability",
        action="store_true",
        default=False,
        help="Disable the observability EventBus entirely.",
    )
    group.add_argument(
        "--disable-metrics",
        action="store_true",
        default=False,
        help="Disable metrics subscribers (OTel counters).",
    )
    group.add_argument(
        "--disable-logging",
        action="store_true",
        default=False,
        help="Disable logging subscribers.",
    )
    group.add_argument(
        "--enable-tracing",
        action="store_true",
        default=False,
        help="Enable span subscribers (OTel traces). Disabled by default.",
    )
    group.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        help=(
            "OTLP gRPC endpoint (e.g. http://localhost:4317). "
            "When set, metrics/traces are pushed to an OTel collector. "
            "When unset, falls back to Prometheus pull mode."
        ),
    )
    group.add_argument(
        "--event-bus-queue-size",
        type=int,
        default=10_000,
        help=(
            "Maximum number of events in the EventBus queue before "
            "tail-drop. Default is 10000."
        ),
    )
    group.add_argument(
        "--prometheus-port",
        type=int,
        default=9090,
        help=(
            "Port for the Prometheus /metrics endpoint. "
            "Only used when --otlp-endpoint is not set. Default is 9090."
        ),
    )
    group.add_argument(
        "--metrics-sample-rate",
        type=float,
        default=0.01,
        help=(
            "Fraction of chunks/blocks to track for lifecycle histograms "
            "(0, 1.0]. Counters always count all events. Default is 0.01 (1%%)."
        ),
    )

    # Lookup hash logging config
    log_group = parser.add_argument_group(
        "Lookup Hash Logging",
        "Configuration for lookup hash file logging (offline analysis)",
    )
    log_group.add_argument(
        "--lookup-hash-log-dir",
        type=str,
        default="",
        help="Directory to write lookup hash JSONL files for offline analysis. "
        "Empty string (default) disables logging.",
    )
    log_group.add_argument(
        "--lookup-hash-log-rotation-interval",
        type=int,
        default=6 * 3600,
        help="Time interval in seconds before rotating to a new log file. "
        "Default is 21600 (6 hours).",
    )
    log_group.add_argument(
        "--lookup-hash-log-rotation-max-size",
        type=int,
        default=100 * 1024 * 1024,
        help="Max file size in bytes before rotating even if the time "
        "interval has not elapsed. Default is 100MB (104857600).",
    )
    log_group.add_argument(
        "--lookup-hash-log-max-files",
        type=int,
        default=100,
        help="Max number of lookup hash log files to keep. "
        "Oldest files are deleted when this limit is exceeded. Default is 100.",
    )

    gc_group = parser.add_argument_group(
        "GC Monitoring",
        "Time CPython garbage collections so their pauses can be "
        "correlated with request tail latency",
    )
    gc_group.add_argument(
        "--enable-gc-monitor",
        action="store_true",
        default=False,
        help="Time every CPython garbage collection and log the slow ones at "
        "INFO. Disabled by default.",
    )
    gc_group.add_argument(
        "--gc-monitor-min-pause-ms",
        type=float,
        default=1.0,
        help="Only log garbage collections that took at least this many "
        "milliseconds. Default is 1.0; 0 logs every collection, including "
        "the very frequent sub-millisecond gen-0 sweeps.",
    )
    gc_group.add_argument(
        "--gc-monitor-top-objects",
        type=int,
        default=0,
        help="Include a breakdown of the N most common object types in each "
        "logged collection. This walks the whole generation on EVERY "
        "collection and can itself cost hundreds of milliseconds on a large "
        "cache. Default is 0 (off); use only while debugging.",
    )

    extra_group = parser.add_argument_group(
        "Extra Logging",
        "Periodic INFO-level L0<->L1 throughput/token stats per GPU and "
        "L1 memory usage",
    )
    extra_group.add_argument(
        "--enable-extra-logging",
        action="store_true",
        default=False,
        help="Periodically log L0<->L1 store/retrieve throughput and token "
        "counts per GPU, and L1 memory usage. Disabled by default.",
    )
    extra_group.add_argument(
        "--extra-logging-interval",
        type=float,
        default=10.0,
        help="Seconds between extra-logging emissions. Default is 10.0. "
        "Values below 1.0 are limited by the 1 Hz internal heartbeat.",
    )

    trace_group = parser.add_argument_group(
        "Trace Recording",
        "Capture LMCache operations to a binary trace file for replay "
        "(see `lmcache trace`).",
    )
    trace_group.add_argument(
        "--trace-level",
        type=str,
        choices=["storage"],
        default=None,
        help="Enable trace recording at the given level. Currently only "
        "'storage' is supported (records StorageManager public-API calls).",
    )
    trace_group.add_argument(
        "--trace-output",
        type=str,
        default=None,
        help="Path to write the trace file. Defaults to a timestamped "
        "file under $TMPDIR when --trace-level is set without an explicit "
        "output path.",
    )

    return parser


def parse_args_to_observability_config(
    args: argparse.Namespace,
) -> ObservabilityConfig:
    """Convert parsed command line arguments to an ObservabilityConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        The configuration object.
    """
    config = ObservabilityConfig(
        enabled=not args.disable_observability,
        max_queue_size=args.event_bus_queue_size,
        metrics_enabled=not args.disable_metrics,
        logging_enabled=not args.disable_logging,
        tracing_enabled=args.enable_tracing,
        otlp_endpoint=args.otlp_endpoint,
        prometheus_port=args.prometheus_port,
        metrics_sample_rate=args.metrics_sample_rate,
        lookup_hash_log=LookupHashLogConfig(
            output_dir=args.lookup_hash_log_dir,
            rotation_interval_sec=args.lookup_hash_log_rotation_interval,
            rotation_max_size=args.lookup_hash_log_rotation_max_size,
            max_files=args.lookup_hash_log_max_files,
        ),
        gc_monitor=GCMonitorConfig(
            enabled=args.enable_gc_monitor,
            min_pause_ms=args.gc_monitor_min_pause_ms,
            top_objects=args.gc_monitor_top_objects,
        ),
        extra_logging_enabled=args.enable_extra_logging,
        extra_logging_interval=args.extra_logging_interval,
        trace_level=args.trace_level,
        trace_output=args.trace_output,
    )

    if config.tracing_enabled and config.otlp_endpoint is None:
        raise ValueError(
            "--enable-tracing requires --otlp-endpoint to be set. "
            "Tracing needs an OTLP gRPC endpoint to export spans."
        )

    if config.extra_logging_enabled and not config.enabled:
        raise ValueError(
            "--enable-extra-logging requires the observability EventBus; "
            "remove --disable-observability."
        )

    if config.extra_logging_interval <= 0:
        raise ValueError("--extra-logging-interval must be > 0.")

    return config


def init_observability(
    obs_config: ObservabilityConfig,
    *,
    start_prometheus_http_server: bool = True,
) -> EventBus:
    """Initialize OTel providers, EventBus, and register subscribers.

    This is the single entry-point that every MP server calls at startup.
    Returns a **started** EventBus.

    Args:
        obs_config: Observability configuration.
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server.  Set to ``False`` when an
            external HTTP framework already serves ``/metrics``.
    """
    # First Party
    from lmcache.v1.mp_observability.event_bus import (
        EventBusConfig,
        init_event_bus,
    )

    # Set up OTel providers BEFORE creating subscribers so that
    # module-level get_meter()/get_tracer() calls bind to the real provider
    instance_id = (
        obs_config.service_instance_id
        if obs_config.service_instance_id is not None
        else str(uuid.uuid4())
    )
    resource_attrs = {"service.instance.id": instance_id}

    if obs_config.enabled and obs_config.metrics_enabled:
        # First Party
        from lmcache.v1.mp_observability.otel_init import init_otel_metrics

        init_otel_metrics(
            otlp_endpoint=obs_config.otlp_endpoint,
            prometheus_port=obs_config.prometheus_port,
            resource_attributes=resource_attrs,
            start_http_server=start_prometheus_http_server,
        )

    if obs_config.enabled and obs_config.tracing_enabled:
        # First Party
        from lmcache.v1.mp_observability.otel_init import init_otel_tracing

        init_otel_tracing(
            otlp_endpoint=obs_config.otlp_endpoint,
            resource_attributes=resource_attrs,
        )

    bus = init_event_bus(
        EventBusConfig(
            enabled=obs_config.enabled,
            max_queue_size=obs_config.max_queue_size,
        )
    )

    if obs_config.metrics_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.metrics import (
            BlendMetricsSubscriber,
            EngineMetricsSubscriber,
            EventBusSelfMetricsSubscriber,
            L0L1ThroughputSubscriber,
            L0LifecycleSubscriber,
            L1EvictionLoopSubscriber,
            L1FailureMetricsSubscriber,
            L1LifecycleSubscriber,
            L1MetricsSubscriber,
            L2FailureMetricsSubscriber,
            L2MetricsSubscriber,
            L2ThroughputSubscriber,
            LookupMetricsSubscriber,
            SMLifecycleSubscriber,
            TimeoutMetricsSubscriber,
        )

        sample_rate = obs_config.metrics_sample_rate
        bus.register_subscriber(L0LifecycleSubscriber(sample_rate=sample_rate))
        bus.register_subscriber(L1MetricsSubscriber())
        bus.register_subscriber(L1LifecycleSubscriber(sample_rate=sample_rate))
        bus.register_subscriber(L1FailureMetricsSubscriber())
        bus.register_subscriber(L1EvictionLoopSubscriber())
        bus.register_subscriber(L0L1ThroughputSubscriber())
        bus.register_subscriber(L2MetricsSubscriber())
        bus.register_subscriber(L2FailureMetricsSubscriber())
        bus.register_subscriber(L2ThroughputSubscriber())
        bus.register_subscriber(LookupMetricsSubscriber())
        bus.register_subscriber(SMLifecycleSubscriber(sample_rate=sample_rate))
        bus.register_subscriber(BlendMetricsSubscriber())
        bus.register_subscriber(EngineMetricsSubscriber())
        bus.register_subscriber(EventBusSelfMetricsSubscriber(bus))
        bus.register_subscriber(TimeoutMetricsSubscriber())

    if obs_config.logging_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.logging import (
            BlendLoggingSubscriber,
            L1LoggingSubscriber,
            L2LoggingSubscriber,
            MPServerLoggingSubscriber,
            SMLoggingSubscriber,
            TimeoutLoggingSubscriber,
        )

        bus.register_subscriber(MPServerLoggingSubscriber())
        bus.register_subscriber(L1LoggingSubscriber())
        bus.register_subscriber(L2LoggingSubscriber())
        bus.register_subscriber(SMLoggingSubscriber())
        bus.register_subscriber(BlendLoggingSubscriber())
        bus.register_subscriber(TimeoutLoggingSubscriber())

    if obs_config.tracing_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.tracing import (
            BlendTracingSubscriber,
            MPServerTracingSubscriber,
            TimeoutTracingSubscriber,
            get_span_registry,
        )

        registry = get_span_registry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.register_subscriber(TimeoutTracingSubscriber(registry))

    # Lookup hash file logging (independent of the logging_enabled flag —
    # it has its own enable gate via output_dir).
    if obs_config.lookup_hash_log.enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.logging.lookup_hash import (
            LookupHashLoggingSubscriber,
        )

        bus.register_subscriber(LookupHashLoggingSubscriber(obs_config.lookup_hash_log))

    # Extra-stats logging (independent of the logging_enabled flag — it has
    # its own enable gate).
    if obs_config.extra_logging_enabled:
        # First Party
        from lmcache.v1.mp_observability.subscribers.logging.extra_stats import (
            ExtraStatsLoggingSubscriber,
        )

        bus.register_subscriber(
            ExtraStatsLoggingSubscriber(obs_config.extra_logging_interval)
        )

    bus.start()
    return bus
