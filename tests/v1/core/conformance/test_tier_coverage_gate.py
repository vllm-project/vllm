# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage gate for the KV-cache key-partitioning conformance suite.

A parametrized suite over known cache tiers has the same weakness as the
per-fix tests it replaces: a tier added tomorrow is simply not in the list.
This gate walks the KV-connector registry and fails on any connector that
is neither exercised by the suite nor exempted with a written reason, so
adding a connector without declaring how it partitions the keyspace breaks
the build instead of passing silently.

Most connectors cannot run here: they need a GPU, a third-party package, or
a live external store. That is a fact about CI, not about their keying, and
an exemption that records only the missing dependency hides the claim that
actually matters. So an exemption must ALSO declare the connector's
:class:`Keying` shape. Exempt then means "CI cannot execute this", not
"nobody has said what this does", and the declaration is a checked-in claim
that the next person to touch the connector can be wrong about.

The shapes are the same two the connector tier already distinguishes via
``KEYS_FROM_BLOCK_HASHES``, plus the two honest non-answers.
"""

import inspect
import os.path
from dataclasses import dataclass
from enum import Enum

import regex as re

from vllm.distributed.kv_transfer import kv_connector
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from .test_connector_tier import HARNESS_CLASSES


class Keying(Enum):
    """How a connector derives the key for its external storage."""

    #: Keys off ``request.block_hashes``, so every partitioning dimension the
    #: engine hash mixes in is inherited for free. Correct exactly as far as
    #: the engine hash is correct, and complete the moment it is.
    FROM_BLOCK_HASHES = "keys external storage from request.block_hashes"
    #: Builds its own key from raw token ids. Drops every partitioning
    #: dimension unless each one is separately plumbed into the key, which
    #: makes every new dimension an opt-in per path.
    REDERIVES = "re-derives keys from raw token ids"
    #: Does no prefix lookup at all: keyed per request id, or benchmark-only,
    #: or delegates wholly to child connectors. The invariant does not apply.
    NO_PREFIX_LOOKUP = "no prefix lookup; keyed per request id or delegating"
    #: The key is built inside a third-party package that is not in this repo,
    #: so the shape cannot be read here. Must record what the vLLM-side
    #: adapter hands across, since that bounds which dimensions can reach it.
    OPAQUE = "key construction lives in an out-of-repo package"


@dataclass(frozen=True)
class Exemption:
    """A connector the suite cannot execute, and what it does anyway.

    Attributes:
        keying: The shape claim. This is the part a reviewer should argue
            with; ``blocked_by`` is only why CI cannot check it.
        blocked_by: What stops this connector from running in the suite.
        evidence: Where the claim was read, as ``path:line``. For
            ``OPAQUE``, what the adapter hands across the boundary instead.
        bug: Required when ``keying`` is :attr:`Keying.REDERIVES`. A
            re-deriver drops the partitioning dimensions by construction, so
            it is a live instance of this RFC's bug class and may not be
            exempted silently; it must point at the issue tracking it.
    """

    keying: Keying
    blocked_by: str
    evidence: str
    bug: str | None = None


# Connectors exercised by the connector tier, derived from the harness manifest
# rather than restated. A hand-written list here could name a harness that does
# not exist, or miss one that does, and the gate would still pass; deriving it
# makes "covered" mean "a harness runs this connector" by construction.
COVERED: dict[str, str] = {
    name: f"test_connector_tier.{harness_cls.__name__}"
    for name, harness_cls in HARNESS_CLASSES.items()
}

EXEMPT: dict[str, Exemption] = {
    "DecodeBenchConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "benchmark-only connector, never serves real KV",
        "v1/decode_bench_connector.py:213",
    ),
    "ExampleHiddenStatesConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "store-only; lookup is hard-coded to zero",
        "v1/example_hidden_states_connector.py:450",
    ),
    "FlexKVConnectorV1": Exemption(
        Keying.OPAQUE,
        "needs the flexkv package and a GPU",
        "v1/flexkv_connector.py:196 hands the whole Request to flexkv's "
        "adapter, so block_hashes, prompt_token_ids, cache_salt and "
        "lora_request are all reachable and which one is keyed on is "
        "decided out of repo",
    ),
    "HF3FSKVConnector": Exemption(
        Keying.REDERIVES,
        "needs the hf3fs client",
        "v1/hf3fs/hf3fs_connector.py:1014 chains "
        "md5(f'{previous_hash}_{token_ids}') over request.prompt_token_ids; "
        "the connector directory contains no reference to cache_salt or lora "
        "at all, so LoRA identity is dropped too, which is wider than the "
        "cache_salt scope #51748 tracks",
        bug="#51748",
    ),
    "LMCacheConnectorV1": Exemption(
        Keying.REDERIVES,
        "needs the lmcache package",
        "v1/lmcache_integration/vllm_v1_adapter.py:1165 passes "
        "request.prompt_token_ids to lookup(); cache_salt is never mapped "
        "into request_configs and lora_request is never read",
        bug="#53194",
    ),
    "LMCacheMPConnector": Exemption(
        Keying.REDERIVES,
        "needs an lmcache MP server",
        "v1/lmcache_mp_connector.py:770 passes all_token_ids plus cache_salt "
        "as a first-class kwarg, so the salt at least crosses the boundary; "
        "LoRA identity does not, and whether the salt reaches the key or "
        "only a namespace tag is decided inside lmcache",
        bug="#53194",
    ),
    "MoRIIOConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "needs MoRI and a GPU",
        "v1/moriio/moriio_connector.py:500 gates on do_remote_prefill and "
        "keys transfers by request_id",
    ),
    "MooncakeConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "needs mooncake and a GPU",
        "v1/mooncake/mooncake_connector.py:719 gates on do_remote_prefill; "
        "state is keyed by ReqId",
    ),
    "MooncakeStoreConnector": Exemption(
        Keying.FROM_BLOCK_HASHES,
        "needs a mooncake store",
        "v1/mooncake/store/data.py:189 keys objects as "
        "f'{prefix}@{chunk_hash.hex()}' over the BlockHash handed in from "
        "request.block_hashes at store/scheduler.py:120",
    ),
    "MultiConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "composes other connectors; keyed by its children",
        "v1/multi_connector.py:393 returns the first child that hits and "
        "builds no key of its own",
    ),
    "NixlConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "needs NIXL and a GPU",
        "v1/nixl/connector.py:12 aliases NixlPullConnector",
    ),
    "NixlPullConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "needs NIXL and a GPU",
        "v1/nixl/pull_scheduler.py:60 gates on do_remote_prefill and keys by "
        "remote_request_id",
    ),
    "NixlPushConnector": Exemption(
        Keying.NO_PREFIX_LOOKUP,
        "needs NIXL and a GPU",
        "v1/nixl/push_scheduler.py:118 gates on do_remote_prefill and keys "
        "by request_id",
    ),
    "OffloadingConnector": Exemption(
        Keying.FROM_BLOCK_HASHES,
        "CPU backend needs CUDA",
        "v1/offloading/scheduler.py:874 builds the key with "
        "make_offload_key(request.block_hashes[hash_idx], group_idx)",
    ),
}


def _builtin_connectors() -> set[str]:
    """Registered connectors that ship with vLLM.

    Test modules register their own mock connectors into the same registry;
    those are keyed by a module path under ``tests`` and are not this gate's
    concern.
    """
    builtin = set()
    for name, loader in KVConnectorFactory._registry.items():
        try:
            module_path = inspect.getclosurevars(loader).nonlocals["module_path"]
        except (TypeError, KeyError) as exc:
            # Reading the loader closure couples this gate to how
            # KVConnectorFactory.register_connector builds its registry
            # entries. If that changes (a functools.partial, a renamed local),
            # say so here rather than surfacing a bare KeyError that reads as
            # "the conformance gate is broken".
            raise AssertionError(
                "cannot classify registered connector "
                f"{name!r}: this gate reads the module path out of "
                "KVConnectorFactory.register_connector's loader closure, and "
                f"that closure no longer exposes it ({exc!r}). Update this "
                "helper, or expose the module path on the factory."
            ) from exc
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


#: Root the evidence pointers are relative to, resolved from the package
#: itself so the check does not depend on the working directory.
CONNECTOR_ROOT = os.path.dirname(kv_connector.__file__)

#: Leading token of an evidence string: ``path/to/file.py:123``.
EVIDENCE_POINTER = re.compile(r"^(?P<path>[\w./-]+\.py):(?P<line>\d+)\b")


def test_every_exemption_declares_a_keying_shape():
    """An exemption records why CI cannot run the connector; the shape claim
    is the part that survives the exemption and can be argued with."""
    for name, exemption in EXEMPT.items():
        assert isinstance(exemption.keying, Keying), name
        assert exemption.blocked_by.strip(), f"{name} exempted with no reason"
        assert exemption.evidence.strip(), f"{name} declares a shape with no evidence"


def test_exemption_evidence_points_at_a_file_that_still_exists():
    """A shape claim is only reviewable if the reader can find what it was read
    from, and this tree moves: the nixl split and the mooncake restructure are
    both recent. So the leading ``path:line`` token must name a file that is
    still there. Line numbers and wording are deliberately NOT checked --
    pinning those would drag this file into every unrelated refactor -- but a
    file that moves should fail here the day it moves."""
    for name, exemption in EXEMPT.items():
        match = EVIDENCE_POINTER.match(exemption.evidence)
        assert match is not None, (
            f"{name}: evidence must lead with a path:line pointer relative to "
            f"{CONNECTOR_ROOT}, got {exemption.evidence[:60]!r}"
        )
        path = os.path.join(CONNECTOR_ROOT, match["path"])
        assert os.path.exists(path), (
            f"{name}: evidence points at {match['path']}, which no longer "
            "exists. The connector moved, so re-read it and update both the "
            "pointer and the shape claim it supports."
        )


def test_rederiving_connectors_are_not_silently_exempted():
    """A re-deriver drops every partitioning dimension by construction, so it
    is a live instance of the bug class this suite exists to catch. Not being
    runnable in CI is not a reason to stop tracking it: the exemption has to
    carry the issue, or this gate fails."""
    untracked = [
        name
        for name, exemption in EXEMPT.items()
        if exemption.keying is Keying.REDERIVES and not exemption.bug
    ]
    assert not untracked, (
        "connectors that re-derive their own keys, and so drop every "
        "partitioning dimension, must reference the issue tracking them "
        f"rather than being exempted on a missing dependency: {untracked}"
    )
