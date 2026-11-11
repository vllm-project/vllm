# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native scheduler half of MooncakeStoreConnector
(`vllm._rust_core`, `connectors::mooncake_store_scheduler`).

Its scheduling behavior is checked against the Python connector by the Rust
trace replay (`sched::scheduler::tests::mooncake_store`), which compares the
metadata in a canonical form. These tests close the loop to the worker: the
Python metadata built from the Rust one pickles to the same bytes as the
Python connector's, and the native port accepts exactly the configurations it
implements.
"""

import importlib.util
import os
import pickle

import pytest

rust_core = pytest.importorskip("vllm._rust_core")

pytestmark = pytest.mark.cpu_test

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
GEN_TRACES = os.path.join(
    REPO_ROOT, "rust/src/engine-core-connectors/src/tests/gen_mooncake_traces.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_mooncake_traces", GEN_TRACES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


@pytest.mark.parametrize(
    "kind",
    [
        kind
        for kind in gen.SCENARIOS
        if gen.SCENARIOS[kind].layout != "dsv41"
        or os.path.exists(gen.DSV41_KV_CACHE_CONFIG)
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_rust_built_metadata_pickles_identically(kind: str, seed: int):
    """Every step's metadata, rebuilt from its Rust form, pickles to the bytes
    of the Python connector's: the same values and the same shared objects
    (the live unfinished-request set, request ids, the live block-hash lists,
    and the preempted set of the `SchedulerOutput`)."""
    converter = rust_core.MooncakeStoreMetaConverter()
    num_steps = 0

    def on_schedule(scheduler, so, canonical, recorder):
        nonlocal num_steps
        num_steps += 1
        finished_hashes = {
            rid: list(req.block_hashes) for rid, req in recorder.finished.items()
        }
        rust_meta = rust_core.RustMooncakeStoreMetadata(canonical, finished_hashes)
        rebuilt = converter.convert(
            rust_meta, scheduler.requests.get, so.preempted_req_ids
        )
        expected = so.kv_connector_metadata
        for protocol in (4, pickle.HIGHEST_PROTOCOL):
            assert pickle.dumps(rebuilt, protocol) == pickle.dumps(
                expected, protocol
            ), f"step {num_steps}"

    gen.record(seed, gen.SCENARIOS[kind], num_steps=50, on_schedule=on_schedule)
    assert num_steps == 50


def _connector(kind: str):
    sc = gen.SCENARIOS[kind]
    scheduler, _, _ = gen.make_scheduler(sc, seed=0)
    connector = scheduler.connector
    return connector


def test_native_port_accepts_supported_configs():
    for kind in ("both", "producer", "consumer", "capacity", "mamba_partial"):
        assert (
            rust_core.mooncake_store_native_scheduler_unsupported(_connector(kind))
            is None
        ), kind


def test_native_port_rejects_kv_events():
    connector = _connector("both")
    connector.connector_scheduler.enable_kv_events = True
    reason = rust_core.mooncake_store_native_scheduler_unsupported(connector)
    assert reason is not None and "KV cache events" in reason


def test_native_port_rejects_subclasses():
    connector = _connector("both")

    class Wrapped(type(connector)):  # type: ignore[misc]
        pass

    connector.__class__ = Wrapped
    assert rust_core.mooncake_store_native_scheduler_unsupported(connector)
