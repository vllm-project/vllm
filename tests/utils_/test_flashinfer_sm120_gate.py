"""The SM120 decode gate must probe everything its consumer imports.

aldc-john-moran (vllm-project/vllm#41834) served the branch on 2x GB10 with
FlashInfer 0.6.12 -- the version the harness Dockerfile pins -- and both ranks
died at model init on ModuleNotFoundError, immediately after a successful
cross-node NCCL rendezvous. The gate is documented as "availability-gated
(silent FlashMLA fallback when the kernel is absent)"; it probed
trtllm_batch_decode_sparse_mla_dsv4, which 0.6.12 has, while the consumer
imports _SparseMLAPagedAttentionRunner from flashinfer.mla._sparse_mla_sm120,
which it does not. The gate said yes, the fallback never ran, the import raised.
"""

import ast
import inspect

from vllm.models.deepseek_v4.nvidia import flashinfer_sm120_decode
from vllm.utils import flashinfer as fi_utils


def _probed_symbols() -> set[str]:
    """Names the gate imports inside its own body."""
    src = inspect.getsource(fi_utils.has_flashinfer_trtllm_sparse_mla_dsv4)
    tree = ast.parse(src.lstrip())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_gate_probes_every_flashinfer_symbol_its_consumer_imports():
    consumer = inspect.getsource(flashinfer_sm120_decode)
    imported = set()
    for node in ast.walk(ast.parse(consumer)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "flashinfer"
        ):
            for alias in node.names:
                imported.add(alias.name)

    probed = _probed_symbols()
    missing = imported - probed
    assert not missing, (
        f"{sorted(missing)} imported by flashinfer_sm120_decode but never probed "
        f"by has_flashinfer_trtllm_sparse_mla_dsv4 (probes {sorted(probed)}); "
        "the gate can return True while the import raises"
    )


def test_gate_reports_false_when_the_runner_is_absent(monkeypatch):
    """The behaviour the fallback depends on, not just the symbol list."""
    fi_utils.has_flashinfer_trtllm_sparse_mla_dsv4.cache_clear()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flashinfer.mla._sparse_mla_sm120":
            raise ImportError("no _sparse_mla_sm120 (as on FlashInfer 0.6.12)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert fi_utils.has_flashinfer_trtllm_sparse_mla_dsv4() is False
    finally:
        fi_utils.has_flashinfer_trtllm_sparse_mla_dsv4.cache_clear()


def test_unavailable_reason_covers_the_version_half(monkeypatch):
    """flashinfer-python 0.6.13 exposes nearby sparse-MLA APIs without the
    SM120 module, and a cubin mismatched against python fails at first kernel
    call — symbols alone fail open in both cases (alexbi29,
    vllm-project/vllm#41834)."""
    import importlib.metadata as md

    from vllm.utils import flashinfer as fi

    versions = {"flashinfer-python": "0.6.13", "flashinfer-cubin": "0.6.13"}
    monkeypatch.setattr(md, "version", lambda name: versions[name])
    reason = fi.flashinfer_sm120_sparse_mla_unavailable_reason()
    assert reason is not None and "0.6.14" in reason

    versions = {"flashinfer-python": "0.6.16", "flashinfer-cubin": "0.6.15"}
    reason = fi.flashinfer_sm120_sparse_mla_unavailable_reason()
    assert reason is not None and "does not match" in reason
