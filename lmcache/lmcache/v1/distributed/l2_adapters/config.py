# SPDX-License-Identifier: Apache-2.0

"""
Configuration for L2 adapters.

Supports multiple adapter instances (including multiple instances of the same
adapter type with different configs) via repeatable --l2-adapter <JSON>.
Each JSON object must include "type" (adapter type name) and type-specific keys.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar
import json

if TYPE_CHECKING:
    # Standard
    import argparse

    # First Party
    from lmcache.v1.distributed.config import EvictionConfig

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.serde import SerdeConfig

logger = init_logger(__name__)

T = TypeVar("T", bound="L2AdapterConfigBase")


@dataclass(frozen=True)
class PersistConfig:
    """
    Configuration for persist on an L2 adapter.

    When enabled, data files are kept on disk at shutdown instead of
    deleted. Lookup always checks secondary storage (disk) on miss
    regardless of this setting.
    """

    persist_enabled: bool = True
    """ If True, data files are kept on disk at shutdown instead of deleted. """


# -----------------------------------------------------------------------------
# Registry: adapter type name -> config class
# -----------------------------------------------------------------------------

_L2_ADAPTER_CONFIG_REGISTRY: dict[str, type[L2AdapterConfigBase]] = {}


def register_l2_adapter_type(
    name: str,
    config_cls: type[L2AdapterConfigBase],
) -> None:
    """
    Register an L2 adapter config class under a type name.

    The type name is used in JSON specs as the "type" field.
    Each adapter config module should call this at import time.

    Args:
        name: Adapter type name (e.g. "fs", "mock").
        config_cls: Config class that can parse from dict
            via ``from_dict()``.
    """
    if name in _L2_ADAPTER_CONFIG_REGISTRY:
        raise ValueError("L2 adapter type already registered: %s" % repr(name))
    _L2_ADAPTER_CONFIG_REGISTRY[name] = config_cls


def _ensure_config_loaded(name: str) -> None:
    """Trigger lazy import for *name* if it is not
    yet in the config registry.

    Raises:
        ImportError: If the adapter module cannot be
            imported (missing dependency).
    """
    if name in _L2_ADAPTER_CONFIG_REGISTRY:
        return
    # Lazy import lives in factory to avoid circular deps
    # First Party
    from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: PLC0415
        ensure_adapter_loaded,
    )

    ensure_adapter_loaded(name)


def get_l2_adapter_config_class(type_name: str) -> type["L2AdapterConfigBase"]:
    """Resolve a registered L2-adapter config class by type name.

    Lazily imports the defining module if needed (via
    :func:`_ensure_config_loaded`), then returns the registered config class.
    Public accessor so callers need not reach into the private registry.

    Args:
        type_name: Registered adapter type (e.g. ``"fs_native"``, ``"mock"``).

    Returns:
        The :class:`L2AdapterConfigBase` subclass registered under *type_name*.

    Raises:
        ValueError: If *type_name* is not a registered adapter type.
    """
    _ensure_config_loaded(type_name)
    if type_name not in _L2_ADAPTER_CONFIG_REGISTRY:
        raise ValueError(f"unknown L2 adapter type {type_name!r}")
    return _L2_ADAPTER_CONFIG_REGISTRY[type_name]


def get_registered_l2_adapter_types() -> list[str]:
    """Return all known adapter type names (eager
    and lazy)."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: PLC0415
        get_all_registered_names,
    )

    return get_all_registered_names()


def get_type_name_for_config(
    config: L2AdapterConfigBase,
) -> str:
    """
    Reverse-lookup the registered type name for a config
    instance.

    Args:
        config: An L2 adapter config instance.

    Returns:
        The registered type name (e.g., "mock", "fs").

    Raises:
        ValueError: If the config's class is not registered.
    """
    for name, cls in _L2_ADAPTER_CONFIG_REGISTRY.items():
        if type(config) is cls:
            return name
    raise ValueError("Unregistered L2 adapter config type: %s" % type(config).__name__)


# -----------------------------------------------------------------------------
# Base config class for a single L2 adapter
# -----------------------------------------------------------------------------


class L2AdapterConfigBase(ABC):
    """
    Base class for per-adapter configs.

    Each adapter type (e.g. disk, redis) defines a config class that:
    - Subclasses this base.
    - Implements from_dict() to parse a dict (from JSON) into an instance.
    - Is registered via register_l2_adapter_type("type_name", ConfigClass).

    An optional ``"eviction"`` key in the JSON dict enables per-adapter L2
    eviction. When present it is parsed by ``_parse_eviction_config()`` and
    stored on ``eviction_config``; the L2EvictionController is then created
    for this adapter in the StorageManager.
    """

    #: Populated by ``_parse_eviction_config`` after ``from_dict``; ``None``
    #: means L2 eviction is disabled for this adapter.
    eviction_config: EvictionConfig | None = None

    #: Populated by ``_parse_persist_config`` after ``from_dict``.
    #: Defaults to ``PersistConfig()`` (persist enabled).
    persist_config: PersistConfig = PersistConfig()

    #: Populated by ``_parse_serde_config`` after ``from_dict``; ``None``
    #: means serde is disabled for this adapter. When set,
    #: ``StorageManager`` wraps the adapter with
    #: ``SerdeL2AdapterWrapper`` so controllers see a plain L2 adapter
    #: and serde runs transparently around store / load.
    #:
    #: JSON schema::
    #:
    #:     {
    #:         "type": "<registered_serde_name>",
    #:         ...type_specific_keys (forwarded to the factory)
    #:     }
    #:
    #: Built-in types and their kwargs:
    #:   - ``"fp8"`` — see :class:`Fp8QuantizationSerializer`.
    #:     Accepts ``fp8_dtype`` (torch dtype name, default
    #:     ``"float8_e4m3fn"``) and ``max_workers`` (thread-pool size
    #:     for async (de)serialize, default ``1``).
    serde_config: SerdeConfig | None = None

    @staticmethod
    def _parse_serde_config(d: dict[str, object]) -> SerdeConfig | None:
        """Parse an optional ``"serde"`` sub-dict from an adapter JSON spec.

        Expected format::

            {
                "type": "fs",
                ...,
                "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
            }

        Returns ``None`` when the key is absent (serde disabled).
        """
        serde_dict = d.get("serde")
        if serde_dict is None:
            return None
        if not isinstance(serde_dict, dict):
            raise ValueError(f"'serde' must be a dict, got {type(serde_dict).__name__}")
        serde_type = serde_dict.get("type")
        if not isinstance(serde_type, str):
            raise ValueError("'serde' dict must include a 'type' field")
        # Forward all keys except "type" as type-specific kwargs.
        kwargs = {k: v for k, v in serde_dict.items() if k != "type"}
        return SerdeConfig(type=serde_type, kwargs=kwargs)

    @staticmethod
    def _parse_eviction_config(d: dict) -> EvictionConfig | None:
        """
        Parse an optional ``"eviction"`` sub-dict from an adapter JSON spec.

        Expected format::

            {
                "type": "mock",
                ...
                "eviction": {
                    "eviction_policy": "LRU",
                    "trigger_watermark": 0.8,
                    "eviction_ratio": 0.2
                }
            }

        Returns ``None`` when the key is absent (eviction disabled).
        """
        eviction_dict = d.get("eviction")
        if eviction_dict is None:
            return None

        # Lazy import to avoid circular dependency:
        # l2_adapters/config.py <- config.py <- l2_adapters/config.py
        # First Party
        from lmcache.v1.distributed.config import EvictionConfig  # noqa: PLC0415

        policy = eviction_dict.get("eviction_policy")
        if policy not in ("LRU", "IsolatedLRU", "noop"):
            raise ValueError(
                "eviction.eviction_policy must be 'LRU', 'IsolatedLRU', or "
                f"'noop', got {policy!r}"
            )
        return EvictionConfig(
            eviction_policy=policy,
            trigger_watermark=float(eviction_dict.get("trigger_watermark", 0.8)),
            eviction_ratio=float(eviction_dict.get("eviction_ratio", 0.2)),
        )

    @staticmethod
    def _parse_persist_config(d: dict) -> PersistConfig:
        """
        Parse optional ``"persist_enabled"`` key from an adapter JSON spec.

        Defaults to ``True``.

        Expected format::

            {
                "type": "nixl_store_dynamic",
                ...
                "persist_enabled": true
            }
        """
        persist_enabled = bool(d.get("persist_enabled", True))
        return PersistConfig(persist_enabled=persist_enabled)

    @staticmethod
    def _validate_num_workers(raw: object) -> int:
        """Validate and return a positive integer worker count.

        Raises:
            ValueError: If ``raw`` is not a positive integer.
        """
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError("num_workers must be a positive integer")
        return raw

    @staticmethod
    def _validate_per_op_workers(
        per_op_workers: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Validate per-operation worker counts (``None`` is a no-op).

        Raises:
            ValueError: If any worker count is not a positive integer.
        """
        if per_op_workers is None:
            return None
        for key, value in per_op_workers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"per_op_workers[{key!r}] must be a positive integer, got {value!r}"
                )
        return per_op_workers

    @staticmethod
    def _parse_per_op_workers_from_dict(
        d: dict[str, object],
    ) -> dict[str, int] | None:
        """Parse ``per_op_workers`` from a raw configuration dict.

        Returns ``None`` if the key is absent.

        Raises:
            ValueError: If the value is not a dict of integers.
        """
        raw = d.get("per_op_workers")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("per_op_workers must be a dict")
        per_op_workers: dict[str, int] = {}
        for k, v in raw.items():
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(
                    f"per_op_workers[{k!r}] must be an integer, got {type(v).__name__}"
                )
            per_op_workers[str(k)] = v
        return per_op_workers

    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """
        Build a config instance from a dict (e.g. from parsed JSON).

        The dict will contain the "type" key used for dispatch; the concrete
        class may ignore it. All other keys are type-specific.

        Args:
            d: Adapter spec dict (must include type-specific keys).

        Returns:
            An instance of the config class.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        ...

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """
        Return a help string describing the config fields for this adapter type.

        This is used in command-line help to explain the expected JSON format for
        each adapter type.

        Returns:
            A help string describing the config fields for this adapter type.
        """
        ...


# -----------------------------------------------------------------------------
# Main config: list of adapter configs (order = adapter order)
# -----------------------------------------------------------------------------


@dataclass
class L2AdaptersConfig:
    """
    Main config for L2 adapters.

    Holds an ordered list of adapter configs. Each element corresponds to one
    L2 adapter instance (e.g. two disk adapters with different paths appear
    as two entries).
    """

    adapters: list[L2AdapterConfigBase]
    """ Ordered list of adapter configs; one per L2 adapter instance. """


# -----------------------------------------------------------------------------
# Command-line: add args and parse to config
# -----------------------------------------------------------------------------

_L2_ADAPTER_ARG_DEST = "l2_adapter"


def add_l2_adapters_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add L2 adapter configuration arguments to an existing parser.

    Adds a repeatable --l2-adapter <JSON> argument. Each JSON object specifies
    one adapter: it must include "type" (registered adapter type name) and
    type-specific keys. Order of arguments is the order of adapters.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with L2 adapter arguments added.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_l2_adapters_args(parser)
        >>> args = parser.parse_args(["--l2-adapter", '{"type":"disk","path":"/data"}'])
        >>> config = parse_args_to_l2_adapters_config(args)
    """
    group = parser.add_argument_group(
        "L2 Adapters",
        "L2 adapter instances. Each --l2-adapter is a JSON object with 'type' and "
        "type-specific keys. Repeat for multiple adapters.",
    )
    group.add_argument(
        "--l2-adapter",
        dest=_L2_ADAPTER_ARG_DEST,
        action="append",
        default=[],
        type=str,
        metavar="JSON",
        help='Adapter spec as JSON with a "type" field and adapter-specific configs'
        ', e.g. \'{"type":"disk","path":"/data"}\'.'
        "Repeat for multiple adapters."
        "Supported adapters are: ["
        + ", ".join(sorted(get_registered_l2_adapter_types()))
        + "].",
    )
    return parser


def parse_args_to_l2_adapters_config(args: argparse.Namespace) -> L2AdaptersConfig:
    """
    Build L2AdaptersConfig from parsed command-line arguments.

    Expects args to have the attribute added by add_l2_adapters_args (a list
    of JSON strings). Each string is parsed; the "type" field selects the
    config class from the registry, and from_dict() builds the config instance.

    Args:
        args: Parsed arguments (e.g. from parser.parse_args()).

    Returns:
        L2AdaptersConfig with one entry per --l2-adapter argument.

    Raises:
        KeyError: If an adapter "type" is not registered.
        ValueError: If JSON is invalid or a config class raises from_dict().
    """
    raw_list = getattr(args, _L2_ADAPTER_ARG_DEST, None)
    if raw_list is None:
        raw_list = []

    adapter_configs: list[L2AdapterConfigBase] = []
    for i, raw in enumerate(raw_list):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for --l2-adapter #{i + 1}: {e}") from e

        if not isinstance(d, dict):
            raise ValueError(
                f"--l2-adapter #{i + 1}: expected a JSON object, got {type(d).__name__}"
            )

        type_name = d.get("type")
        if type_name is None:
            raise ValueError("--l2-adapter #%d: missing 'type' field" % (i + 1))

        # Trigger lazy import for this adapter type
        _ensure_config_loaded(type_name)

        if type_name not in _L2_ADAPTER_CONFIG_REGISTRY:
            known = ", ".join(sorted(_L2_ADAPTER_CONFIG_REGISTRY)) or "(none)"
            raise ValueError(
                f"--l2-adapter #{i + 1}: unknown adapter type "
                f"{type_name!r}. Known: {known}"
            )

        config_cls = _L2_ADAPTER_CONFIG_REGISTRY[type_name]
        try:
            adapter_cfg = config_cls.from_dict(d)
            adapter_cfg.eviction_config = L2AdapterConfigBase._parse_eviction_config(d)
            adapter_cfg.persist_config = L2AdapterConfigBase._parse_persist_config(d)
            adapter_cfg.serde_config = L2AdapterConfigBase._parse_serde_config(d)
            adapter_configs.append(adapter_cfg)
        except (TypeError, ValueError) as e:
            logger.error(
                "Error parsing --l2-adapter #%d (type %r): %s",
                i + 1,
                type_name,
                e,
            )
            logger.error(
                "Adapter config help for %s adapter:\n"
                "---------------------\n"
                "%s\n"
                "---------------------\n\n",
                type_name,
                config_cls.help(),
            )
            raise ValueError(f"--l2-adapter #{i + 1} ({type_name!r}): {e}") from e

    return L2AdaptersConfig(adapters=adapter_configs)
