# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.compile.backend import TestBackend
from vllm import LLM, SamplingParams, ir
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.config import get_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor import kernel_packages
from vllm.platforms import current_platform

SITE = "vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd"
# A pass-through Solution with a setup step, fitting one-chunk prefills (K == 1).
KERNEL = """\
from vllm.model_executor.layers.mamba.ops import ssd_chunk_scan


def setup(context):
    return None


def run(state, **kwargs):
    return ssd_chunk_scan._chunk_scan_fwd.__wrapped__(**kwargs)
"""
SHAPES = {"cb": "KGSS", "x": "LHP", "dt": "HKS", "dA_cumsum": "HKS", "C": "LGN"}
SHAPES |= {"states": "KHPN", "cu_chunk_seqlens": "J", "out": "LHP", "seq_idx": "K"}
SHAPES |= {"D": "H"}
MAMBA2 = {
    "architectures": ["Mamba2ForCausalLM"],
    "model_type": "mamba2",
    "hidden_size": 256,
    "num_hidden_layers": 1,
    "num_heads": 8,
    "head_dim": 64,
    "n_groups": 1,
    "state_size": 64,
    "vocab_size": 1024,
    "torch_dtype": "bfloat16",
}


def seal(root, kernel=KERNEL, chunks=1):
    definition = {
        "name": "chunk_scan",
        "op_type": "mamba",
        "tags": [f"fi_api:{SITE}"],
        "axes": {"K": {"type": "const", "value": chunks}},
        "inputs": {name: {"shape": list(axes)} for name, axes in SHAPES.items()},
        "outputs": {},
    }
    solution = {
        "name": "pass_through",
        "definition": "chunk_scan",
        "author": "test",
        "spec": {
            "language": "python",
            "target_hardware": ["any"],
            "entry_point": "kernel.py::run",
            "setup_entry_point": "kernel.py::setup",
            "destination_passing_style": True,
        },
        "sources": [{"path": "kernel.py", "content": kernel}],
    }
    files = {"definition.json": definition, "solution.json": solution}
    files = {name: json.dumps(doc).encode() for name, doc in files.items()}
    files["kernel.py"] = kernel.encode()
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    listing = "".join(f"{name}\0{d}\n" for name, d in sorted(digests.items()))
    bundle_sha256 = hashlib.sha256(listing.encode()).hexdigest()
    manifest = json.dumps({"files": digests, "bundle_sha256": bundle_sha256}).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (root / digest / "bundle").mkdir(parents=True)
    (root / digest / "manifest.json").write_bytes(manifest)
    for name, data in files.items():
        (root / digest / "bundle" / name).write_bytes(data)
    return digest


class Doubler:
    @kernel_packages.kernel_site
    def forward(self, x, k=None, scale=2.0):
        return x * scale


def test_site_fit_rules(monkeypatch):
    stats = pytest.importorskip("flashinfer.trace_apply").stats
    site = f"{__name__}.Doubler.forward"
    inputs = {"x": {"shape": ["T", "H"]}, "k": {"optional": True}, "scale": {}}
    pinned = {"inputs": inputs, "axes": {"T": {"type": "const", "value": 2}}}
    absent = {"inputs": {**inputs, "ghost": {"optional": True}}}  # not a parameter
    routes = [
        (absent, lambda **kwargs: None),  # never fits, so the next package is tried
        (pinned, lambda x, k, scale: x + 1),
        ({"inputs": inputs}, lambda x, k, scale: x - 1),
    ]
    monkeypatch.setitem(kernel_packages._routes, site, routes)
    forward = Doubler().forward

    # A method, a parameter left at its default and an optional input at None fit.
    assert forward(torch.zeros(2, 4)).sum() == 8
    assert forward(torch.zeros(3, 4)).sum() == -12  # T is not 2: the next package
    assert forward(torch.ones(4)).sum() == 8  # no package fits: the stock function
    counts = stats()[site]
    assert counts == {"hit": 2, "fallback_rank_x": 1}

    # A compiler tracing the call: a pinned size is not matched against a symbolic
    # one, the package that fits is compiled in, and no request is counted.
    trace = make_fx(lambda x: forward(x), tracing_mode="symbolic")
    traced = trace(torch.zeros(2, 4))
    assert "sub" in traced.code and "add" not in traced.code
    assert stats()[site] == counts


def test_package_is_compiled_into_a_lowered_norm(monkeypatch, default_vllm_config):
    stats = pytest.importorskip("flashinfer.trace_apply").stats
    site = "vllm.ir.ops.layernorm.rms_norm"
    inputs = {"x": {"shape": ["T", "H"]}, "weight": {"shape": ["H"]}, "epsilon": {}}
    routes = [({"inputs": inputs}, lambda x, weight, epsilon: x + 1)]
    monkeypatch.setitem(kernel_packages._routes, site, routes)
    torch.set_default_device(current_platform.device_type)
    x, weight = torch.randn(8, 16), torch.ones(16)

    backend = TestBackend(VllmIRLoweringPass(get_current_vllm_config()))
    with ir.ops.rms_norm.set_priority(["native"]), ir.enable_torch_wrap(True):
        norm = torch.compile(
            lambda x: ir.ops.rms_norm(x, weight, 1e-5), backend=backend, fullgraph=True
        )
        assert torch.equal(norm(x), x + 1)
    assert site not in stats()  # chosen at compile time: no request is counted


def test_package_serves_mamba2_chunk_scan(tmp_path, monkeypatch):
    # Skipped here, not at import: a run of this file alone then exits 0.
    packages = pytest.importorskip("flashinfer.trace_apply.packages")
    if not hasattr(packages, "installed_packages"):
        pytest.skip("needs FlashInfer's installed_packages()")
    from flashinfer.trace_apply import stats

    (tmp_path / "config.json").write_text(json.dumps(MAMBA2))
    root = tmp_path / "packages"
    # A package that fails to import, selected first under the same names: its
    # Definition (K == 3) must not be routed to the next package's kernel.
    digests = [seal(root, "raise RuntimeError\n", 3), seal(root)]
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGE_ROOT", str(root))
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGES", ",".join(digests))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # FlashInfer was imported before the env; load as its import does.
    packages.load_packages(root, digests, strict=False)
    args = dict(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.3,
    )
    try:
        llm = LLM(**args)
        sampling = SamplingParams(max_tokens=1)
        llm.generate([{"prompt_token_ids": list(range(300))}], sampling)  # two chunks
        llm.generate([{"prompt_token_ids": list(range(500, 508))}], sampling)
        assert stats()[SITE]["hit"] >= 1
        assert stats()[SITE]["fallback_axis_K"] >= 1
        with pytest.raises(ValueError, match="reload_weights would rewrite"):
            llm.collective_rpc("reload_weights")
        del llm
        cleanup_dist_env_and_memory()

        with pytest.raises(ValueError, match="sleep mode would rewrite"):
            LLM(**args, enable_sleep_mode=True)
    finally:
        kernel_packages._routes.clear()
        packages.release_packages()
        cleanup_dist_env_and_memory()
