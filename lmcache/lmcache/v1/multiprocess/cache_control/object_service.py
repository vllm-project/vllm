# SPDX-License-Identifier: Apache-2.0
"""Node-local cache object operations (adapter listing, object listing, delete).

:class:`ObjectService` wraps the storage manager's L2 adapters and exposes
adapter resolution, paginated listing, and key-addressed deletion. It performs
its own validation and raises transport-agnostic domain errors (see
:mod:`cache_control.errors`); the HTTP layer maps those to status codes.
Blocking adapter I/O is off-loaded to a worker thread so callers can ``await``.
"""

# Standard
from typing import Any
import asyncio

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, Tier
from lmcache.v1.multiprocess.cache_control.errors import (
    InvalidRequest,
    NotFound,
    Unavailable,
)

# Hard cap on how many keys a single delete request may target. Keeps the
# request body bounded and prevents one call from monopolizing the adapter's
# I/O loop for an unbounded interval.
MAX_DELETE_BATCH = 10_000

# Object operations address the L2 tier today; other tiers are rejected.
_SUPPORTED_TIER = Tier.L2


class ObjectService:
    """Adapter-listing, object-listing, and key-addressed deletion on one node.

    Args:
        engine: The node's cache engine; its ``storage_manager`` owns the L2
            adapters.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @staticmethod
    def _require_supported_tier(tier: Tier) -> None:
        """Raise :class:`InvalidRequest` unless ``tier`` is the supported one."""
        if tier != _SUPPORTED_TIER:
            raise InvalidRequest(
                f"tier {tier.value!r} not supported; only {_SUPPORTED_TIER.value!r}"
            )

    def _resolve_adapter(self, selector: str | None) -> tuple[Any, Any]:
        """Resolve the ``(descriptor, adapter)`` pair a request targets.

        ``selector=None`` picks the primary (first-configured) adapter.

        Raises:
            Unavailable: No adapters configured.
            NotFound: ``selector`` matches none.
        """
        adapters = self._engine.storage_manager.l2_adapters()
        if not adapters:
            raise Unavailable("no L2 adapters configured")
        if selector is None:
            return adapters[0]
        for desc, adapter in adapters:
            if desc.type_name == selector:
                return desc, adapter
        raise NotFound(f"no L2 adapter with type_name={selector!r}")

    def list_adapters(self) -> dict[str, object]:
        """Return the live L2 adapter inventory.

        Returns:
            ``{"adapters": [{index, type_name, tier, primary, reconfigurable},
            ...]}`` (empty list when no backends are configured).
            ``reconfigurable`` is ``True`` for adapters that accept runtime
            ``/reconfigure`` operations; the ``{backend}`` path parameter for
            those routes is the adapter's ``type_name``. This is the single live
            adapter listing -- it supersedes the old ``/reconfigure/backends``.
        """
        sm = self._engine.storage_manager
        adapters = sm.l2_adapters()
        reconfigurable = sm.reconfigurable_l2_backends()
        return {
            "adapters": [
                {
                    "index": i,
                    "type_name": desc.type_name,
                    "tier": _SUPPORTED_TIER.value,
                    "primary": i == 0,
                    "reconfigurable": desc.type_name in reconfigurable,
                }
                for i, (desc, _) in enumerate(adapters)
            ]
        }

    async def list_objects(
        self,
        tier: Tier,
        adapter_selector: str | None,
        model_name: str | None,
        page_size: int,
        page_token: str | None,
    ) -> dict[str, object]:
        """List keys resident in one adapter, paginated.

        Returns:
            ``{"adapter", "entries", "next_page_token"}``.

        Raises:
            InvalidRequest: unsupported ``tier`` or malformed ``page_token``.
            Unavailable: no adapters configured, or the adapter does not
                implement listing.
            NotFound: ``adapter_selector`` matches none.
        """
        self._require_supported_tier(tier)
        desc, adapter = self._resolve_adapter(adapter_selector)
        try:
            page = await asyncio.to_thread(
                adapter.list_l2_keys,
                model_name=model_name,
                page_size=page_size,
                cursor=page_token,
            )
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from None
        except NotImplementedError as exc:
            raise Unavailable(
                f"L2 adapter {desc.type_name!r} does not support listing: {exc}"
            ) from None
        return {
            "adapter": desc.type_name,
            "entries": page.entries,
            "next_page_token": page.next_page_token,
        }

    async def delete_objects(
        self,
        tier: Tier,
        adapter_selector: str | None,
        keys: list[EncodedObjectKey],
        force: bool = False,
    ) -> dict[str, object]:
        """Delete a caller-supplied list of keys from L1, L2, or both (``tier``).

        Key-addressed. L1 deletion respects read/write locks unless ``force``;
        L2 deletion is idempotent at the adapter level (absent / locked keys are
        skipped). The L2 adapter is only resolved when the tier includes L2, so a
        pure ``l1`` delete works on an L1-only server.

        Returns:
            ``{"deleted", "skipped", "ok"[, "error"]}``: ``deleted`` is the total
            keys removed across the requested tiers (L1 removals plus the L2 batch
            size); ``skipped`` is the L1 keys refused because they were locked
            (non-force only); ``ok`` is ``False`` with ``error`` set when the L2
            adapter raised (a structured failure, not a crash).

        Raises:
            InvalidRequest: batch too large, or an ``ObjectKey`` invariant
                violation.
            Unavailable / NotFound: L2 adapter resolution (only when ``tier``
                includes L2).
        """
        if len(keys) > MAX_DELETE_BATCH:
            raise InvalidRequest(
                f"too many keys in a single request "
                f"(limit={MAX_DELETE_BATCH}, got={len(keys)})"
            )
        parsed = []
        for i, cache_key in enumerate(keys):
            try:
                parsed.append(cache_key.to_object_key())
            except ValueError as exc:
                raise InvalidRequest(f"keys[{i}]: {exc}") from None

        deleted = 0
        skipped = 0
        ok = True
        error: str | None = None

        if tier in (Tier.L1, Tier.ALL):
            l1_deleted, l1_skipped = self._engine.storage_manager.delete_l1_keys(
                parsed, force=force
            )
            deleted += l1_deleted
            skipped += l1_skipped

        if tier in (Tier.L2, Tier.ALL):
            _, adapter = self._resolve_adapter(adapter_selector)
            try:
                await asyncio.to_thread(adapter.delete, parsed)
                deleted += len(parsed)
            except Exception as exc:  # noqa: BLE001 - structured result, not a crash
                ok = False
                error = str(exc)

        result: dict[str, object] = {"deleted": deleted, "skipped": skipped, "ok": ok}
        if error is not None:
            result["error"] = error
        return result
