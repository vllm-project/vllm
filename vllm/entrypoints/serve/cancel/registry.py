# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict

DEFAULT_MAX_TRACKED_REQUESTS = 8192


class CancellationRegistry:
    """Maps a client-visible request id to the engine request ids behind it.

    A single client request is fanned out to one engine request per prompt or
    per choice, under derived ids such as ``cmpl-abc-0``. The client only ever
    sees the undivided ``cmpl-abc``, so cancelling by the id from the response
    body has to be resolved back to the derived ids first.

    Entries are evicted by a bounded LRU rather than when a request finishes.
    Aborting a request that already completed is a no-op, so a stale entry
    costs nothing beyond a 404.
    """

    def __init__(self, max_tracked: int = DEFAULT_MAX_TRACKED_REQUESTS) -> None:
        self._max_tracked = max_tracked
        self._engine_req_ids: OrderedDict[str, list[str]] = OrderedDict()

    def add(self, client_req_id: str, engine_req_id: str) -> None:
        if (engine_req_ids := self._engine_req_ids.get(client_req_id)) is None:
            self._engine_req_ids[client_req_id] = engine_req_ids = []
            while len(self._engine_req_ids) > self._max_tracked:
                self._engine_req_ids.popitem(last=False)
        else:
            self._engine_req_ids.move_to_end(client_req_id)
        engine_req_ids.append(engine_req_id)

    def resolve(self, client_req_id: str) -> list[str]:
        """Return the engine request ids to abort for a client request id.

        Ids from a path that does not fan out, or one evicted from the LRU,
        are returned unchanged so that they are still tried against the engine.
        """
        if (engine_req_ids := self._engine_req_ids.get(client_req_id)) is None:
            return [client_req_id]
        return list(engine_req_ids)
