# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage gate for the KV-cache key-partitioning conformance suite.

A parametrized suite over known cache tiers has the same weakness as the
per-fix tests it replaces: a tier added tomorrow is simply not in the list.
This gate walks the KV-connector registry and fails on any connector that
is neither exercised by the suite nor exempted with a written reason, so
adding a connector without declaring how it partitions the keyspace breaks
the build instead of passing silently.
"""

import inspect

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

# Connectors exercised by the connector tier, mapped to the harness that does it.
COVERED: dict[str, str] = {
    "ExampleConnector": "test_connector_tier.ExampleConnectorHarness",
    "SimpleCPUOffloadConnector": "test_connector_tier.SimpleCPUOffloadHarness",
}

# Connectors that cannot run in this suite, each with the reason. An entry
# here is a declaration that the connector's keying is out of scope for CI,
# which is the thing a reviewer should push back on.
EXEMPT: dict[str, str] = {
    "DecodeBenchConnector": "benchmark-only connector, never serves real KV",
    "ExampleHiddenStatesConnector": "store-only; path is keyed per request id",
    "FlexKVConnectorV1": "needs the flexkv package and a GPU",
    "HF3FSKVConnector": "needs the hf3fs client",
    "LMCacheConnectorV1": "needs the lmcache package",
    "LMCacheMPConnector": "needs an lmcache MP server",
    "MoRIIOConnector": "needs MoRI and a GPU",
    "MooncakeConnector": "needs mooncake and a GPU",
    "MooncakeStoreConnector": "needs a mooncake store",
    "MultiConnector": "composes other connectors; keyed by its children",
    "NixlConnector": "needs NIXL and a GPU",
    "NixlPullConnector": "needs NIXL and a GPU",
    "NixlPushConnector": "needs NIXL and a GPU",
    "OffloadingConnector": "keyed by request.block_hashes; CPU backend needs CUDA",
}


def _builtin_connectors() -> set[str]:
    """Registered connectors that ship with vLLM.

    Test modules register their own mock connectors into the same registry;
    those are keyed by a module path under ``tests`` and are not this gate's
    concern.
    """
    builtin = set()
    for name, loader in KVConnectorFactory._registry.items():
        module_path = inspect.getclosurevars(loader).nonlocals["module_path"]
        if not module_path.startswith("tests."):
            builtin.add(name)
    return builtin


def test_every_registered_connector_declares_partitioning():
    registered = _builtin_connectors()
    undeclared = registered - COVERED.keys() - EXEMPT.keys()
    assert not undeclared, (
        "KV connectors registered but neither covered by the key-partitioning "
        f"suite nor exempted with a reason: {sorted(undeclared)}"
    )
    stale = (COVERED.keys() | EXEMPT.keys()) - registered
    assert not stale, f"declared but no longer registered: {sorted(stale)}"
    assert not COVERED.keys() & EXEMPT.keys()
