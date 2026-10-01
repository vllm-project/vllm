# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit-equals-fresh for streaming weight reload (modulewise reload).

An engine that loaded checkpoint A, with CUDA graphs captured, reloads B the
way NCCL/IPC weight transfer does (device tensors through the model's own
`load_weights`). Afterwards every registered tensor and every tensor reachable
from module attributes (quant methods, kernels, quant configs: the
"off-registry" state captured graphs read) must equal a fresh load of B, and
none of them may have moved: a moved tensor is one a captured graph no longer
reads. Checksums, not logits, are compared: logits of separately started
processes can differ (kernel selection), weights cannot.
"""

import glob
import json
import os
import shutil
import zlib

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

from .test_reload import _fp8_reload_unsupported

HARNESS = "tests.model_executor.model_loader.reload_harness.ReloadHarnessExtension"

CASES = [
    pytest.param(
        "Qwen/Qwen3-0.6B",
        "inference-optimization/Qwen3-0.6B-debug-multiply",
        {},
        id="bf16-dense-tied",
    ),
    pytest.param(
        "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
        "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
        {},
        id="fp8-block-dense",
    ),
    pytest.param(
        "inference-optimization/DeepSeek-V3-debug-empty",
        "inference-optimization/DeepSeek-V3-debug-multiply",
        {},
        id="mla-moe",
    ),
    pytest.param(
        "tiiuae/Falcon-H1-0.5B-Base",
        "tiiuae/Falcon-H1-0.5B-Instruct",
        {},
        id="mamba-hybrid",
    ),
    pytest.param(
        "allenai/OLMoE-1B-7B-0924",
        "allenai/OLMoE-1B-7B-0924-Instruct",
        {"quantization": "fp8_per_tensor"},
        id="online-fp8-moe",
        marks=[
            pytest.mark.slow_test,
            pytest.mark.xfail(
                strict=True, reason="online FP8 MoE rebuilds its kernel on reload"
            ),
        ],
    ),
]


def _diff_ptrs(before: dict, after: dict) -> dict[str, list[str]]:
    out = {}
    for kind in ("registered", "off_registry"):
        b, a = before[kind], after[kind]
        out[kind] = sorted(
            [k for k in b if k in a and a[k][0] != b[k][0]]
            + [k for k in a if k not in b]
        )
    return out


def _diff_checksums(got: dict, expected: dict) -> dict[str, list[str]]:
    return {
        kind: sorted(
            k
            for k in set(got[kind]) | set(expected[kind])
            if got[kind].get(k) != expected[kind].get(k)
        )
        for kind in ("registered", "off_registry")
    }


@pytest.mark.parametrize("model_a,model_b,kwargs", CASES)
def test_refit_equals_fresh(vllm_runner, model_a, model_b, kwargs):
    if not current_platform.is_cuda():
        pytest.skip("CUDA graphs and device-tensor streaming")
    if "FP8" in model_a and _fp8_reload_unsupported():
        pytest.skip("Requires FP8 support")
    _check_refit_equals_fresh(vllm_runner, model_a, model_b, kwargs)


def _check_refit_equals_fresh(vllm_runner, model_a, model_b, kwargs):
    kwargs = dict(kwargs)
    common = dict(
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        worker_extension_cls=HARNESS,
        # graphs on; autotuning off keeps kernel choice stable across engines
        kernel_config={
            "enable_flashinfer_autotune": False,
            **kwargs.pop("kernel_config", {}),
        },
        **kwargs,
    )
    with vllm_runner(model_b, **common) as fresh:
        expected = fresh.collective_rpc("mw_checksums")

    with vllm_runner(model_a, **common) as llm:
        before = llm.collective_rpc("mw_ptr_snapshot")
        for _ in range(2):  # the second reload runs against built-once kernels
            stats = llm.collective_rpc(
                "mw_reload",
                kwargs=dict(path=model_b),
            )
        after = llm.collective_rpc("mw_ptr_snapshot")
        got = llm.collective_rpc("mw_checksums")

    for rank, (g, e, b, a, st) in enumerate(zip(got, expected, before, after, stats)):
        assert _diff_checksums(g, e) == {"registered": [], "off_registry": []}, (
            f"rank {rank}: tensors differ from a fresh load"
        )
        assert _diff_ptrs(b, a) == {"registered": [], "off_registry": []}, (
            f"rank {rank}: tensors moved (captured graphs read the old ones)"
        )
        assert st["end_over_base"] == 0, f"rank {rank}: reload left memory held"


# ---------------------------------------------------------------------------
# One case per MoE backend that declares `reload_safe` (SM100), plus FP8 KV.
# B is derived from A (`make_variant`), so each case only needs A on the Hub. Generated
# checkpoints go to VLLM_TEST_RELOAD_CKPT_DIR (reused across runs) or a
# session temp dir; the 30B cases need ~35 GB each.
# ---------------------------------------------------------------------------

QWEN3_MOE_FP8 = "Qwen/Qwen3-30B-A3B-FP8"
OLMOE = "allenai/OLMoE-1B-7B-0924"
QWEN3_MOE = "Qwen/Qwen3-30B-A3B"
DSV3_CT_FP8 = "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC"
NVFP4_MODELOPT = "nvidia/Qwen3-30B-A3B-NVFP4"
NVFP4_CT = "RedHatAI/Qwen3-30B-A3B-NVFP4"


def _backend(name, recipe, source, moe_backend, **kwargs):
    kernel_config = {"moe_backend": moe_backend} if moe_backend else {}
    return pytest.param(
        recipe,
        source,
        {"kernel_config": kernel_config, **kwargs},
        id=f"{name}-{moe_backend or 'default'}",
        marks=[pytest.mark.slow_test],
    )


# Cases are added by the PRs that declare each backend reload-safe.
BACKEND_CASES: list = []


@pytest.fixture(scope="session")
def reload_ckpt_dir(tmp_path_factory):
    root = os.environ.get("VLLM_TEST_RELOAD_CKPT_DIR")
    return root or str(tmp_path_factory.mktemp("reload_ckpts"))


@pytest.mark.parametrize("recipe,source,kwargs", BACKEND_CASES)
def test_moe_backend_refit_equals_fresh(
    vllm_runner, reload_ckpt_dir, recipe, source, kwargs
):
    if not current_platform.is_cuda():
        pytest.skip("CUDA graphs and device-tensor streaming")
    if not current_platform.is_device_capability_family(100):
        pytest.skip("backend matrix is for SM100")
    slug = source.replace("/", "--")
    if recipe == "fp8_static":
        model_a = make_fp8_static(source, os.path.join(reload_ckpt_dir, f"{slug}-fp8s"))
    else:
        model_a = source
    model_b = make_variant(model_a, os.path.join(reload_ckpt_dir, f"{slug}-{recipe}-B"))
    _check_refit_equals_fresh(vllm_runner, model_a, model_b, kwargs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_DONE = ".reload_checkpoint_done"


def _source_dir(src: str) -> str:
    return src if os.path.isdir(src) else hf_api().snapshot_download(src)


def _copy_non_weights(src: str, dst: str, skip=()) -> None:
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f not in skip:
            shutil.copy(p, os.path.join(dst, f))


def make_variant(src: str, dst: str, eps: float = 0.02) -> str:
    """A perturbed copy of `src` in the same format: float weights get
    multiplicative noise, float scales x1.1 (input scales x1.25), and 1-byte
    tensors (FP8, packed FP4, e8m0 scales) are rolled by one row (one expert
    for stacked [E, ...] tensors), so every quantized weight changes too.
    Noise is seeded per tensor name, and a tied `lm_head` gets the embedding's
    noise, so tied checkpoints stay consistent."""
    if os.path.exists(os.path.join(dst, _DONE)):
        return dst
    src = _source_dir(src)
    _copy_non_weights(src, dst)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    tied = cfg.get("tie_word_embeddings", False)
    for fn in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        out = {}
        for k, t in load_file(fn).items():
            if t.is_floating_point() and t.element_size() >= 2:
                if "input_scale" in k:
                    t = t * 1.25
                elif "scale" in k:
                    t = t * 1.1
                else:
                    name = (
                        "model.embed_tokens.weight"
                        if tied and k == "lm_head.weight"
                        else k
                    )
                    g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                    n = torch.randn(t.shape, generator=g, dtype=torch.float32)
                    t = (t.float() * (1 + eps * n)).to(t.dtype)
            elif t.element_size() == 1 and t.dim() >= 1 and t.shape[0] > 1:
                # FP8 / packed FP4 / e8m0: any permutation is still valid data
                t = torch.roll(t, 1, dims=0)
            out[k] = t.contiguous()
        save_file(out, os.path.join(dst, os.path.basename(fn)), {"format": "pt"})
    _mark_done(src, dst)
    return dst


def _mark_done(src: str, dst: str) -> None:
    missing = [
        f
        for f in os.listdir(src)
        if f.endswith(".safetensors") and not os.path.exists(os.path.join(dst, f))
    ]
    assert not missing, f"{dst}: weight files not written: {missing}"
    open(os.path.join(dst, _DONE), "w").close()


_FP8_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def make_fp8_static(src: str, dst: str) -> str:
    """Quantize a BF16 checkpoint to serialized per-tensor FP8 with static
    activation scales (`quant_method: fp8`, `activation_scheme: static`), the
    `Fp8MoEMethod` per-tensor path. Activation scales are synthetic
    (deterministic, per module), which is enough for refit-equals-fresh."""
    if os.path.exists(os.path.join(dst, _DONE)):
        return dst
    src = _source_dir(src)
    _copy_non_weights(src, dst, skip=("model.safetensors.index.json",))
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # MoE routers stay unquantized: vLLM would quantize them without scales
    routers = sorted(
        k[: -len(".weight")]
        for fn in glob.glob(os.path.join(src, "*.safetensors"))
        for k in safe_open(fn, "pt").keys()  # noqa: SIM118
        if k.endswith(".mlp.gate.weight")
    )
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "static",
        "ignored_layers": ["lm_head", *routers],
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_map = {}
    for fn in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        out = {}
        for k, t in load_file(fn).items():
            if (
                k.endswith(".weight")
                and t.dim() == 2
                and any(f".{p}." in k for p in _FP8_PROJECTIONS)
            ):
                w = t.float()
                scale = w.abs().max().clamp(min=1e-8) / fp8_max
                base = k[: -len(".weight")]
                out[k] = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                out[base + ".weight_scale"] = scale.reshape(()).float()
                out[base + ".input_scale"] = (0.01 + 0.05 * w.abs().mean()).reshape(())
            else:
                out[k] = t.contiguous()
        name = os.path.basename(fn)
        save_file(out, os.path.join(dst, name), {"format": "pt"})
        weight_map.update(dict.fromkeys(out, name))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    open(os.path.join(dst, _DONE), "w").close()
    return dst
