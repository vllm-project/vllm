# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanitize client-supplied ``kv_transfer_params`` before they enter the
engine.

Topology fields (``remote_host``, ``remote_port``) must only come from a
trusted router/proxy, never from a public API client.  Stripping them at
the API boundary prevents SSRF-class issues (CWE-918) regardless of which
KV connector is active downstream.
"""

from typing import Any

_TOPOLOGY_KEYS = frozenset({"remote_host", "remote_port"})


def sanitize_kv_transfer_params(
    params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a copy of *params* with peer-topology fields removed.

    Strips ``remote_host`` and ``remote_port`` from the top level
    (used by the NIXL connector) and from the ``prefill`` / ``decode``
    sub-dicts (used by P2P offloading) so that clients cannot steer
    internal control-plane connections.  All other keys (e.g.
    ``kv_request_id``, ``do_remote_prefill``) are preserved.

    Returns ``None`` when *params* is ``None`` or becomes empty after
    stripping.
    """
    if not params:
        return params

    out = {k: v for k, v in params.items() if k not in _TOPOLOGY_KEYS}

    for role_key in ("prefill", "decode"):
        role = out.get(role_key)
        if not isinstance(role, dict):
            continue
        if role.keys() & _TOPOLOGY_KEYS:
            out[role_key] = {k: v for k, v in role.items() if k not in _TOPOLOGY_KEYS}

    return out or None
