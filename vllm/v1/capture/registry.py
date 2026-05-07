# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry-point discovery for capture consumers.

Third-party plugins and vLLM's own built-in consumers register by
advertising a class under the ``vllm.capture_consumers`` entry-point
group in their ``pyproject.toml``::

    [project.entry-points."vllm.capture_consumers"]
    my_trainer = "my_plugin:RewardTrainer"

This module enumerates that group on first access and caches the
resolved ``name -> class`` map for the process lifetime. Cache
invalidation is not supported — plugins are discovered once per
engine startup.

Phase A ships the singular helpers used by ad-hoc instantiation.
Phase G added the plural ``build_consumers`` helper that walks config
and installs driver bridges for ``location = "driver"`` consumers.
Phase D fixed ``build_consumers`` to iterate ``CaptureConsumersConfig``
attributes (the Phase E dataclass shape) rather than dict keys, and
augmented its return type to a three-tuple so callers can reach the
per-consumer validator and the ``name → index`` map needed for
per-request client-spec routing.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import TYPE_CHECKING, Any

from vllm.v1.capture.consumer import CaptureConsumer, _BatchedAdapter
from vllm.v1.capture.driver_bridge import _DriverQueueShim, install_driver_consumer
from vllm.v1.capture.errors import UnknownCaptureConsumerError
from vllm.v1.capture.sink import CaptureSink

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


ENTRY_POINT_GROUP = "vllm.capture_consumers"

_cache_lock = threading.Lock()
_class_cache: dict[str, type[CaptureConsumer]] | None = None


def _load_entry_points() -> dict[str, type[CaptureConsumer]]:
    """Enumerate the ``vllm.capture_consumers`` entry-point group and
    resolve every entry point to its class.

    Caches the resulting map; subsequent calls return the cached dict.
    """
    global _class_cache
    with _cache_lock:
        if _class_cache is not None:
            return _class_cache

        resolved: dict[str, type[CaptureConsumer]] = {}
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        for ep in entry_points:
            cls = ep.load()
            if not isinstance(cls, type):
                raise TypeError(
                    f"Entry point {ep.name!r} in group "
                    f"{ENTRY_POINT_GROUP!r} resolved to {cls!r}, which is "
                    f"not a class."
                )
            # Accept both ``CaptureConsumer`` subclasses (the common
            # user-facing base) and classes that implement
            # ``CaptureSink`` directly (e.g. ``FilesystemConsumer``,
            # which owns its own streaming writer pool and bypasses
            # ``_BatchedAdapter``).  A direct-sink consumer is
            # identified by declaring ``submit_chunk`` /
            # ``submit_finalize`` on the class.
            is_consumer_subclass = issubclass(cls, CaptureConsumer)
            is_direct_sink = all(
                hasattr(cls, attr)
                for attr in ("submit_chunk", "submit_finalize", "get_result")
            )
            if not (is_consumer_subclass or is_direct_sink):
                raise TypeError(
                    f"Entry point {ep.name!r} in group "
                    f"{ENTRY_POINT_GROUP!r} resolved to {cls!r}, which is "
                    f"neither a CaptureConsumer subclass nor a direct "
                    f"CaptureSink implementation."
                )
            resolved[ep.name] = cls

        _class_cache = resolved
        return _class_cache


def load_consumer_class(name: str) -> type[CaptureConsumer]:
    """Return the ``CaptureConsumer`` subclass registered under
    ``name`` in the ``vllm.capture_consumers`` entry-point group.

    Raises ``UnknownCaptureConsumerError`` if no entry point matches.
    """
    resolved = _load_entry_points()
    try:
        return resolved[name]
    except KeyError:
        available = sorted(resolved.keys())
        raise UnknownCaptureConsumerError(
            f"No capture consumer named {name!r} is registered. "
            f"Available consumers: {available}. "
            f"Plugins must advertise a class under the "
            f"{ENTRY_POINT_GROUP!r} entry-point group."
        ) from None


def build_consumer(
    name: str,
    vllm_config: VllmConfig,
    params: dict[str, Any],
) -> CaptureConsumer:
    """Resolve ``name`` via the registry and construct an instance.

    Equivalent to ``load_consumer_class(name)(vllm_config, params)``.
    """
    cls = load_consumer_class(name)
    return cls(vllm_config, params)


def build_consumers(
    vllm_config: VllmConfig,
    consumer_instances: list[CaptureConsumer] | None = None,
) -> tuple[
    tuple[CaptureSink, ...],
    tuple[Any, ...],
    dict[str, int],
]:
    """Build the consumer sinks, validators, and name-to-index map.

    Two sources of consumers are supported:

    1. **Config entries** — ``CaptureConsumerSpec`` entries stored on
       ``vllm_config.capture_consumers_config.consumers``.  These are
       resolved through the entry-point registry via ``build_consumer``
       and can have any ``location``.

    2. **Pre-constructed instances** — passed directly (e.g. from
       ``LLM(..., capture_consumers=[instance])``).  These *must*
       declare ``location = "driver"`` since the ``LLM`` constructor
       runs in the driver process.

    For each consumer:

    - ``location = "worker"`` → wrap in a ``_BatchedAdapter`` (runs
      in-process on the worker) or, for ``FilesystemConsumer`` and any
      other consumer that already implements ``CaptureSink`` directly
      (detected via ``on_capture`` being absent on the instance's
      abstract surface), install the sink as-is.
    - ``location = "driver"`` → install a driver bridge via
      ``install_driver_consumer``, which returns a worker-side shim.

    Returns:
        A three-tuple ``(sinks, validators, name_to_index)``:

        - ``sinks`` — tuple of ``CaptureSink`` objects in the same order
          as the config entries followed by the instance list.
        - ``validators`` — parallel tuple of objects that expose
          ``validate_client_spec(raw_spec, ctx)``.  For
          ``_BatchedAdapter``-wrapped consumers this is the underlying
          consumer; for direct ``CaptureSink`` implementations (like
          ``FilesystemConsumer``) the sink itself acts as the validator;
          for driver shims the original driver-side consumer.
        - ``name_to_index`` — maps ``spec.instance_name or spec.name``
          for config entries and ``type(instance).__name__`` (with a
          collision suffix) for pre-constructed instances onto the
          consumer index used by ``CaptureManager.register_request``.
    """
    sinks: list[CaptureSink] = []
    validators: list[Any] = []
    name_to_index: dict[str, int] = {}

    # --- config-driven consumers (entry-point registry) ---
    config = getattr(vllm_config, "capture_consumers_config", None)
    # Support two shapes:
    # 1. Phase E ``CaptureConsumersConfig`` dataclass with
    #    ``.consumers: list[CaptureConsumerSpec]``.
    # 2. Plain ``list[dict]`` (test + legacy convenience).
    config_entries: list[Any]
    if config is None:
        config_entries = []
    elif hasattr(config, "consumers"):
        config_entries = list(config.consumers)
    else:
        config_entries = list(config)

    for entry in config_entries:
        if hasattr(entry, "name"):
            # ``CaptureConsumerSpec`` dataclass (Phase E).
            entry_name: str = entry.name
            instance_name: str | None = getattr(entry, "instance_name", None)
            params: dict[str, Any] = getattr(entry, "params", {}) or {}
        else:
            # Legacy dict form.
            entry_name = entry["name"]
            instance_name = entry.get("instance_name")
            params = entry.get("params", {}) or {}

        consumer = build_consumer(entry_name, vllm_config, params)
        sink = _wrap_consumer(consumer)
        sinks.append(sink)
        validators.append(_select_validator(sink, consumer))

        key = instance_name or entry_name
        _insert_unique(name_to_index, key, len(sinks) - 1)

    # --- pre-constructed instances ---
    for instance in consumer_instances or []:
        if instance.location != "driver":
            raise ValueError(
                f"Pre-constructed CaptureConsumer instances passed to "
                f"LLM() must have location='driver', but "
                f"{type(instance).__name__} has "
                f"location={instance.location!r}."
            )
        sink = _wrap_consumer(instance)
        sinks.append(sink)
        validators.append(_select_validator(sink, instance))
        _insert_unique(name_to_index, type(instance).__name__, len(sinks) - 1)

    return tuple(sinks), tuple(validators), name_to_index


def _insert_unique(mapping: dict[str, int], key: str, value: int) -> None:
    """Insert ``key -> value`` into *mapping*, suffixing duplicates.

    Pre-constructed instances can collide by class name; configured
    instances with the same entry-point name are supposed to carry
    distinct ``instance_name`` values but might not.  We suffix ``#2``,
    ``#3``, … on collision so callers always get a unique key.
    """
    if key not in mapping:
        mapping[key] = value
        return
    suffix = 2
    while f"{key}#{suffix}" in mapping:
        suffix += 1
    mapping[f"{key}#{suffix}"] = value


def _select_validator(sink: CaptureSink, consumer: Any) -> Any:
    """Return the object exposing ``validate_client_spec`` for a consumer.

    - ``_BatchedAdapter``: return the wrapped consumer (the adapter
      doesn't expose ``validate_client_spec`` itself).
    - ``_DriverQueueShim``: return the original driver-side consumer
      reference (the shim lives on the worker; the validator lives on
      the driver but only gets called for its side-effect-free
      ``validate_client_spec``).
    - Any other sink (e.g. ``FilesystemConsumer`` which implements
      ``CaptureSink`` directly): the sink is its own validator.
    """
    if isinstance(sink, _BatchedAdapter):
        # Ensure the adapter exposes ``consumer`` so downstream code can
        # reach ``validate_client_spec`` uniformly.  ``_BatchedAdapter``
        # stores the consumer as ``_consumer``; alias it to ``consumer``.
        if not hasattr(sink, "consumer"):
            sink.consumer = consumer  # type: ignore[attr-defined]
        return consumer
    if isinstance(sink, _DriverQueueShim):
        return consumer
    return sink


def _wrap_consumer(consumer: CaptureConsumer) -> CaptureSink:
    """Wrap a consumer with the appropriate sink for its location."""
    if consumer.location == "driver":
        logger.info(
            "Installing driver bridge for consumer %s",
            type(consumer).__name__,
        )
        return install_driver_consumer(consumer)
    # Worker consumers that implement ``CaptureSink`` directly (e.g.
    # ``FilesystemConsumer``) are used as-is; anything else goes through
    # the in-process batched adapter.  ``CaptureConsumer`` subclasses
    # declare ``on_capture`` so we test for that to distinguish the two.
    if isinstance(consumer, CaptureConsumer):
        return _BatchedAdapter(consumer)
    return consumer  # type: ignore[return-value]


def _reset_cache_for_testing() -> None:
    """Drop the cached entry-point map.

    Tests that patch ``importlib.metadata.entry_points`` must call this
    before and after patching so the cache does not leak across tests.
    """
    global _class_cache
    with _cache_lock:
        _class_cache = None
