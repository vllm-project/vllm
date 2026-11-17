# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json

import pytest

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor import kernel_packages

packages = pytest.importorskip("flashinfer.trace_apply.packages")

from flashinfer.trace_apply import stats  # noqa: E402

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


def seal(root):
    definition = {
        "name": "chunk_scan",
        "op_type": "mamba",
        "tags": [f"fi_api:{SITE}"],
        "axes": {"K": {"type": "const", "value": 1}},
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
        "sources": [{"path": "kernel.py", "content": KERNEL}],
    }
    files = {"definition.json": definition, "solution.json": solution}
    files = {name: json.dumps(doc).encode() for name, doc in files.items()}
    files["kernel.py"] = KERNEL.encode()
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


def test_package_serves_mamba2_chunk_scan(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(MAMBA2))
    root = tmp_path / "packages"
    digest = seal(root)
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGE_ROOT", str(root))
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGES", digest)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    packages.load_packages(root, [digest])  # FlashInfer was imported before the env
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
