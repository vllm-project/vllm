# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``system_fingerprint`` construction."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from vllm.entrypoints.serve.utils import fingerprint as fp


class _FakeResponse(BaseModel):
    """Stand-in for ChatCompletionResponse/CompletionResponse: both are
    plain pydantic models with a nullable ``system_fingerprint`` field, and
    ``dump_response_with_fingerprint`` only cares about that shape."""

    id: str = "resp-1"
    system_fingerprint: str | None = None


def _cfg(tp=1, pp=1, dp=1, ep=False, digest="a3b21f94deadbeef"):
    c = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        )
    )
    c.compute_hash = lambda: digest  # type: ignore[attr-defined]
    return c


@pytest.fixture(autouse=True)
def _reset():
    fp.set_default_fingerprint_mode("full")
    yield
    fp.set_default_fingerprint_mode("full")


def test_four_modes_produce_expected_shapes():
    from vllm import __version__ as v

    cfg = _cfg(tp=8, ep=True)

    assert fp.build_system_fingerprint(cfg, "full") == (f"vllm-{v}-tp8-ep-a3b21f94")
    assert fp.build_system_fingerprint(cfg, "hash") == f"vllm-{v}-a3b21f94"
    assert fp.build_system_fingerprint(cfg, "custom", "my-fp") == "my-fp"
    assert fp.build_system_fingerprint(cfg, "none") is None


def test_full_mode_emits_only_non_trivial_parallelism():
    from vllm import __version__ as v

    # Single-GPU: nothing between version and hash.
    assert fp.build_system_fingerprint(_cfg(), "full") == f"vllm-{v}-a3b21f94"
    # All parallelism axes.
    assert (
        fp.build_system_fingerprint(_cfg(tp=8, pp=2, dp=4, ep=True), "full")
        == f"vllm-{v}-tp8-pp2-dp4-ep-a3b21f94"
    )


def test_get_respects_set_default():
    cfg = _cfg(tp=8)
    full = fp.get_system_fingerprint(cfg)
    assert full == fp.get_system_fingerprint(cfg)

    fp.set_default_fingerprint_mode("hash")
    hashed = fp.get_system_fingerprint(cfg)
    assert hashed != full
    assert "tp8" not in hashed

    fp.set_default_fingerprint_mode("custom", "deploy-42")
    assert fp.get_system_fingerprint(cfg) == "deploy-42"

    fp.set_default_fingerprint_mode("none")
    assert fp.get_system_fingerprint(cfg) is None


def test_compute_hash_failure_does_not_raise():
    cfg = _cfg()
    cfg.compute_hash = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert fp.build_system_fingerprint(cfg, "full").endswith("-nohash")
    assert fp.build_system_fingerprint(cfg, "hash").endswith("-nohash")


def test_dump_response_omits_key_only_in_none_mode():
    with_fp = _FakeResponse(system_fingerprint="vllm-0.1.0-a3b21f94")
    unset = _FakeResponse(system_fingerprint=None)

    # Default mode ("full"): a set fingerprint stays present.
    assert "system_fingerprint" in fp.dump_response_with_fingerprint(with_fp)

    # "none" mode: the key is omitted entirely, not serialized as null.
    fp.set_default_fingerprint_mode("none")
    content = fp.dump_response_with_fingerprint(unset)
    assert "system_fingerprint" not in content

    # Other fields are unaffected.
    assert content["id"] == "resp-1"

    # A field that's None for an unrelated reason (e.g. misconfigured
    # "custom" mode with no value) is left alone: only "none" mode omits it.
    fp.set_default_fingerprint_mode("custom", None)
    content = fp.dump_response_with_fingerprint(unset)
    assert content["system_fingerprint"] is None
