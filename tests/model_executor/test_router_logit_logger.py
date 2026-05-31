# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the B1 router-logit logging hook.

Runtime-focused tests for ``router_logit_logger`` (B1 Q6). These need only ``torch`` —
not a full vLLM build — so they run on a laptop. The module is loaded by file path to
avoid importing the ``vllm`` package (which needs the compiled extension). On the
cluster they also collect normally via pytest.

Run standalone (no pytest / vLLM import needed):
    python tests/model_executor/test_router_logit_logger.py
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Load the module directly from its file so we don't trigger `import vllm`.
_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm/model_executor/layers/router_logit_logger.py"
)
_spec = importlib.util.spec_from_file_location("b1_router_logit_logger", _MOD_PATH)
rll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rll)


def _reset(monkeypatch, tmp_path, **env):
    """Fresh logger state with a clean env pointing at tmp_path (or disabled)."""
    for k in list(os.environ):
        if k.startswith("VLLM_B1_"):
            monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    rll._state = None  # force re-read of env on next call


def _records(tmp_path):
    files = list(Path(tmp_path).glob("router_logits.*.jsonl"))
    out = []
    for f in files:
        out += [json.loads(line) for line in f.read_text().splitlines() if line]
    return out


def test_disabled_by_default_is_noop(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)  # no VLLM_B1_ROUTER_LOG_DIR
    rll.maybe_log_router_logits(torch.randn(4, 8), layer_id=0)
    assert _records(tmp_path) == []
    assert rll._get_state() is rll._DISABLED


def test_enabled_writes_record_with_expected_fields(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path)
    logits = torch.randn(5, 8)
    rll.maybe_log_router_logits(logits, layer_id=3)
    recs = _records(tmp_path)
    assert len(recs) == 1
    r = recs[0]
    assert r["layer_id"] == 3
    assert r["num_tokens"] == 5
    assert r["n_experts"] == 8
    assert r["sampled_tokens"] == 5
    assert len(r["router_logits"]) == 5 and len(r["router_logits"][0]) == 8
    assert r["call_idx"] == 0 and "t_wall" in r and "pid" in r


def test_sampling_stride_logs_one_in_n(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path,
           VLLM_B1_ROUTER_LOG_SAMPLE=0.5)  # stride 2 -> log calls 0, 2, 4, ...
    for _ in range(6):
        rll.maybe_log_router_logits(torch.randn(2, 4), layer_id=1)
    recs = _records(tmp_path)
    assert [r["call_idx"] for r in recs] == [0, 2, 4]


def test_per_layer_counters_independent(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path,
           VLLM_B1_ROUTER_LOG_SAMPLE=0.5)
    rll.maybe_log_router_logits(torch.randn(2, 4), layer_id=0)  # logged (count 0)
    rll.maybe_log_router_logits(torch.randn(2, 4), layer_id=1)  # logged (count 0)
    rll.maybe_log_router_logits(torch.randn(2, 4), layer_id=0)  # skipped (count 1)
    recs = _records(tmp_path)
    assert sorted(r["layer_id"] for r in recs) == [0, 1]


def test_max_tokens_caps_rows(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path,
           VLLM_B1_ROUTER_LOG_MAX_TOKENS=16)
    rll.maybe_log_router_logits(torch.randn(1000, 8), layer_id=0)
    r = _records(tmp_path)[0]
    assert r["num_tokens"] == 1000  # original count preserved
    assert r["sampled_tokens"] == 16  # but rows are capped
    assert len(r["router_logits"]) == 16


def test_replica_and_version_tags(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path,
           VLLM_B1_REPLICA_ID="r7", VLLM_B1_ROUTER_VERSION="v3")
    rll.maybe_log_router_logits(torch.randn(2, 4), layer_id=0)
    r = _records(tmp_path)[0]
    assert r["replica_id"] == "r7" and r["router_version"] == "v3"


def test_noop_while_compiling(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, VLLM_B1_ROUTER_LOG_DIR=tmp_path)
    monkeypatch.setattr(rll, "_is_compiling", lambda: True)
    rll.maybe_log_router_logits(torch.randn(4, 8), layer_id=0)
    assert _records(tmp_path) == []  # never break a torch.compile graph


if __name__ == "__main__":  # standalone runner (no pytest/vLLM import)
    import tempfile

    class _MP:  # minimal monkeypatch shim
        def __init__(self):
            self._undo = []

        def setenv(self, k, v):
            self._undo.append((k, os.environ.get(k)))
            os.environ[k] = v

        def delenv(self, k, raising=True):
            self._undo.append((k, os.environ.get(k)))
            os.environ.pop(k, None)

        def setattr(self, obj, name, val):
            old = getattr(obj, name)
            self._undo.append(("__attr__", (obj, name, old)))
            setattr(obj, name, val)

        def undo(self):
            for k, v in reversed(self._undo):
                if k == "__attr__":
                    obj, name, old = v
                    setattr(obj, name, old)
                elif v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            self._undo.clear()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        mp = _MP()
        with tempfile.TemporaryDirectory() as d:
            try:
                t(mp, Path(d))
                print(f"PASS {t.__name__}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {t.__name__}: {e!r}")
            finally:
                mp.undo()
                rll._state = None
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
