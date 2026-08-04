# SPDX-License-Identifier: Apache-2.0
"""Blend mp-server side client for the coordinator fingerprint directory.

The blend module runs in sync thread pools with no asyncio loop, so this client
owns a synchronous ``httpx.Client`` plus a background daemon, mirroring the
module's existing ``_fingerprint_queue`` worker. STORE publishes are best-effort
and fire-and-forget; LOOKUP match queries follow the same non-blocking
submit-once / poll-on-recall pattern the blend lookup already uses for its prefix
and sparse legs (the handler never blocks a worker thread on the round-trip).

It is **opt-in**: with no coordinator URL configured the blend module receives
``None`` and every publish/query path is skipped, so behavior is unchanged.
"""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.schemas import encode_tokens

logger = init_logger(__name__)

# poll_match sentinel: query submitted, round-trip not finished yet.
PENDING = object()

# (method, path, json_body) -> json_dict
_RequestFn = Callable[[str, str, dict], dict]


@dataclass
class RemoteMatch:
    """One chunk matched in the fleet directory (see directory.GlobalMatch).

    Attributes:
        object_key: Shared-L2 storage key of the matched chunk.
        old_st: Token position in the stored sequence (re-RoPE source).
        cur_st: Token position in the request (re-RoPE target).
    """

    object_key: str
    old_st: int
    cur_st: int


@dataclass
class _PublishItem:
    """A queued best-effort write: an HTTP ``method``/``path``/``payload``."""

    method: str
    path: str
    payload: dict


@dataclass
class _MatchItem:
    """A queued match query for request ``rid``."""

    rid: str
    model_scope: str
    tokens: list[int]


class BlendCoordinatorClient:
    """Background bridge from the blend module to the coordinator directory.

    Thread-safe. Handler threads enqueue publishes and match queries; one daemon
    thread dequeues them, dispatching each match query to a thread pool so
    round-trips run concurrently. Match results land in a dict the handler
    polls. Match queries drain ahead of publishes so lookup latency is not held
    up by best-effort store traffic.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        request_fn: _RequestFn | None = None,
        request_timeout: float = 2.0,
        match_budget_s: float = 2.0,
        match_concurrency: int = 8,
    ) -> None:
        """Create the client and start its daemon.

        Args:
            base_url: Coordinator base URL (e.g. ``http://coordinator:9300``).
                Ignored when ``request_fn`` is supplied.
            request_fn: Optional injected ``(method, path, json_body) ->
                json_dict`` used instead of the built-in HTTP client (for
                testing). Must raise on transport failure.
            request_timeout: Per-request HTTP timeout in seconds.
            match_budget_s: Per-lookup wall-clock budget the blend module uses to
                bound the optional global leg, including queue wait.
            match_concurrency: Max match round-trips in flight at once (the
                match dispatch pool size). Must be >= 1.

        Raises:
            ValueError: If ``match_concurrency`` is not positive.
        """
        if match_concurrency < 1:
            raise ValueError(f"match_concurrency must be >= 1, got {match_concurrency}")
        self.match_budget_s = match_budget_s
        self._client = None
        if request_fn is None:
            # Third Party
            import httpx

            client = httpx.Client(timeout=request_timeout)
            base = base_url.rstrip("/")

            def _http_request(method: str, path: str, payload: dict) -> dict:
                resp = client.request(method, f"{base}{path}", json=payload)
                resp.raise_for_status()
                return resp.json()

            self._client = client
            self._request: _RequestFn = _http_request
        else:
            self._request = request_fn

        self._publish_q: "Queue[_PublishItem]" = Queue()
        self._match_q: "Queue[_MatchItem]" = Queue()
        self._results: dict[str, object] = {}
        self._results_lock = threading.Lock()
        self._stop = threading.Event()
        self._match_pool = ThreadPoolExecutor(
            max_workers=match_concurrency, thread_name_prefix="cb-coord-match"
        )
        self._worker = threading.Thread(
            target=self._run, name="cb-coordinator-client", daemon=True
        )
        self._worker.start()

    def enqueue_register(self, ranges: list[dict]) -> None:
        """Queue a best-effort register of stored ranges.

        Args:
            ranges: Wire-form ``StoreRange`` dicts to register.
        """
        if not ranges:
            return
        self._publish_q.put(
            _PublishItem("POST", "/blend/fingerprints", {"ranges": ranges})
        )

    def enqueue_evict(self, object_keys: list[str]) -> None:
        """Queue a best-effort eviction of fingerprints by storage key.

        Args:
            object_keys: ``object_key`` values to evict.
        """
        if not object_keys:
            return
        self._publish_q.put(
            _PublishItem("DELETE", "/blend/fingerprints", {"object_keys": object_keys})
        )

    def submit_match(self, rid: str, model_scope: str, tokens: list[int]) -> None:
        """Submit a match query for a request once (idempotent per ``rid``).

        Args:
            rid: Request id, the poll key.
            model_scope: Scope to match within.
            tokens: The request tokens (the coordinator hashes and probes them).
        """
        with self._results_lock:
            if rid in self._results:
                return
            self._results[rid] = PENDING
        self._match_q.put(_MatchItem(rid=rid, model_scope=model_scope, tokens=tokens))

    def poll_match(self, rid: str) -> object:
        """Return the match state for a request.

        Args:
            rid: Request id used in :meth:`submit_match`.

        Returns:
            :data:`PENDING` while in flight, a ``list[RemoteMatch]`` when ready,
            or ``None`` if never submitted (or already consumed).
        """
        with self._results_lock:
            return self._results.get(rid)

    def take_match(self, rid: str) -> None:
        """Drop a request's stored match state once consumed.

        Args:
            rid: Request id to clear.
        """
        with self._results_lock:
            self._results.pop(rid, None)

    def close(self) -> None:
        """Stop the daemon, drain the match pool, and close the HTTP client."""
        self._stop.set()
        self._worker.join(timeout=2.0)
        self._match_pool.shutdown(wait=False)
        if self._client is not None:
            self._client.close()

    @classmethod
    def maybe_create(cls, url: str | None) -> "BlendCoordinatorClient | None":
        """Build a started client for ``url``; an empty URL returns ``None``.

        Timing knobs are read from the environment:
        ``LMCACHE_COORDINATOR_BLEND_TIMEOUT`` (seconds, default 1.0; used as
        both the per-request HTTP timeout and the per-lookup match budget) and
        ``LMCACHE_COORDINATOR_BLEND_MATCH_CONCURRENCY`` (default 8).

        Args:
            url: Coordinator base URL; empty or ``None`` disables the client.

        Returns:
            A started client, or ``None`` when ``url`` is empty (the blend
            module then runs purely local).
        """
        url = (url or "").strip()
        if not url:
            return None
        concurrency = int(os.getenv("LMCACHE_COORDINATOR_BLEND_MATCH_CONCURRENCY", "8"))
        timeout = float(os.getenv("LMCACHE_COORDINATOR_BLEND_TIMEOUT", "1.0"))
        logger.info("Blend coordinator client enabled -> %s", url)
        return cls(
            url,
            request_timeout=timeout,
            match_budget_s=timeout,
            match_concurrency=concurrency,
        )

    # -- daemon ------------------------------------------------------------

    def _run(self) -> None:
        """Dispatch match queries to the pool (priority), then publishes."""
        while not self._stop.is_set():
            try:
                item = self._match_q.get(timeout=0.05)
            except Empty:
                item = None
            if item is not None:
                self._match_pool.submit(self._handle_match, item)
                continue
            try:
                self._handle_publish(self._publish_q.get_nowait())
            except Empty:
                pass

    def _handle_match(self, item: _MatchItem) -> None:
        """POST a match query and store the parsed matches (miss on error)."""
        matches: list[RemoteMatch] = []
        try:
            body = self._request(
                "POST",
                "/blend/match",
                {
                    "model_scope": item.model_scope,
                    "tokens_b64": encode_tokens(item.tokens),
                },
            )
            matches = [
                RemoteMatch(
                    object_key=m["object_key"],
                    old_st=int(m["old_st"]),
                    cur_st=int(m["cur_st"]),
                )
                for m in body.get("matches", [])
            ]
        except Exception as e:
            # Best-effort: a failed query degrades to local-only (empty result).
            logger.warning("Blend coordinator match failed for %s: %r", item.rid, e)
        with self._results_lock:
            # Only fill if still awaited (not taken/cleared meanwhile).
            if self._results.get(item.rid) is PENDING:
                self._results[item.rid] = matches

    def _handle_publish(self, item: _PublishItem) -> None:
        """Send a best-effort register/evict; failures are logged and dropped."""
        try:
            self._request(item.method, item.path, item.payload)
        except Exception as e:
            logger.warning(
                "Blend coordinator %s %s failed: %r", item.method, item.path, e
            )
