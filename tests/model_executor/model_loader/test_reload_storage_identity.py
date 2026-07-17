# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Storage Identity Tests for Weight Reload (Issue #48312 Category 1)

Verifies that ALL CUDA tensors reachable from model layers — including
unmanaged tensors not registered as nn.Parameter or nn.Buffer — preserve
their device addresses (data_ptr) across weight reload.

Covers:
  - MLA derived weights: W_UV, W_UK_T (#48251)
  - Marlin workspace + sort indices (#48438)
  - CUTLASS MoE stride descriptors (#41670)
  - Machete act_perm (#48539)
  - FlashInfer CUTLASS MoE constants
  - All other unmanaged CUDA tensors on layer attributes
"""
import functools
import types
from collections import defaultdict

import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform


# ---------------------------------------------------------------------------
# Tensor census utilities (must be top-level for pickling via apply_model)
# ---------------------------------------------------------------------------

def _walk_for_census(path, obj, visited, result, depth, max_depth):
    if depth > max_depth:
        return
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda and obj.numel() > 0:
            result[path] = obj.data_ptr()
        return
    if isinstance(obj, nn.Module):
        return
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is not None:
                _walk_for_census(f"{path}[{k!r}]", v, visited, result,
                                 depth + 1, max_depth)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if v is not None:
                _walk_for_census(f"{path}[{i}]", v, visited, result,
                                 depth + 1, max_depth)
        return
    if isinstance(obj, functools.partial):
        for i, arg in enumerate(obj.args):
            _walk_for_census(f"{path}.args[{i}]", arg, visited, result,
                             depth + 1, max_depth)
        for k, v in obj.keywords.items():
            _walk_for_census(f"{path}.keywords[{k!r}]", v, visited, result,
                             depth + 1, max_depth)
        return
    if isinstance(obj, types.FunctionType) and obj.__closure__:
        for i, cell in enumerate(obj.__closure__):
            try:
                cell_val = cell.cell_contents
            except ValueError:
                continue
            _walk_for_census(f"{path}.__closure__[{i}]", cell_val, visited,
                             result, depth + 1, max_depth)
        return
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is None or isinstance(obj, type):
        return
    for attr_name in list(obj_dict):
        if attr_name.startswith("__"):
            continue
        val = obj_dict.get(attr_name)
        if val is not None:
            _walk_for_census(f"{path}.{attr_name}", val, visited, result,
                             depth + 1, max_depth)


def census_all_cuda_tensors(model):
    """Walk entire module tree, return {dotted_path: data_ptr}."""
    result = {}
    for layer_name, layer in model.named_modules():
        for pname, param in layer._parameters.items():
            if param is not None and param.is_cuda:
                key = f"{layer_name}.{pname}" if layer_name else pname
                result[key] = param.data_ptr()
        for bname, buf in layer._buffers.items():
            if buf is not None and buf.is_cuda:
                key = f"{layer_name}.{bname}" if layer_name else bname
                result[key] = buf.data_ptr()
        skip_names = set()
        skip_names.update(layer._parameters.keys())
        skip_names.update(layer._buffers.keys())
        skip_names.update(layer._modules.keys())
        visited = set()
        for attr_name in list(vars(layer)):
            if attr_name.startswith("_") or attr_name in skip_names:
                continue
            val = getattr(layer, attr_name, None)
            if val is None:
                continue
            prefix = f"{layer_name}.{attr_name}" if layer_name else attr_name
            _walk_for_census(prefix, val, visited, result, 0, 8)
    return result


def get_managed_ptrs(model):
    """Return set of data_ptrs for managed params/buffers."""
    managed = set()
    for layer in model.modules():
        for p in layer._parameters.values():
            if p is not None and p.is_cuda:
                managed.add(p.data_ptr())
        for b in layer._buffers.values():
            if b is not None and b.is_cuda:
                managed.add(b.data_ptr())
    return managed


def classify_tensor(path):
    pl = path.lower()
    if "w_uv" in pl or "w_uk_t" in pl:
        return "MLA_derived"
    if "workspace" in pl:
        return "Marlin_workspace"
    if "sort_indices" in pl or "g_idx" in pl:
        return "Marlin_sort_indices"
    if "strides" in pl:
        return "CUTLASS_MoE_strides"
    if "act_perm" in pl:
        return "Machete_act_perm"
    if "gemm1_alpha" in pl or "gemm1_beta" in pl or "clamp_limit" in pl:
        return "FlashInfer_CUTLASS_MoE"
    return "other"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_storage_identity_check(vllm_runner, model, tp_size=1,
                                quantization=None, trust_remote_code=False):
    """Core storage identity check: census before/after reload."""
    extra = {}
    if quantization:
        extra["quantization"] = quantization

    with vllm_runner(
        model_name=model,
        tensor_parallel_size=tp_size,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
        **extra,
    ) as llm:
        before = llm.apply_model(census_all_cuda_tensors)[0]
        managed_ptrs = llm.apply_model(get_managed_ptrs)[0]

        llm.collective_rpc("reload_weights",
                           kwargs={"weights_path": model})

        after = llm.apply_model(census_all_cuda_tensors)[0]

    # Compute drifts
    unmanaged_drifts = []
    for path, old_ptr in before.items():
        if path not in after:
            continue
        if old_ptr != after[path] and old_ptr not in managed_ptrs:
            unmanaged_drifts.append((path, classify_tensor(path)))

    if unmanaged_drifts:
        by_cat = defaultdict(list)
        for path, cat in unmanaged_drifts:
            by_cat[cat].append(path)
        msg_parts = []
        for cat, paths in sorted(by_cat.items()):
            msg_parts.append(f"{cat}: {paths[:3]}")
        pytest.fail(
            f"{len(unmanaged_drifts)} unmanaged tensor(s) changed address "
            f"after reload: {'; '.join(msg_parts)}"
        )


def _fp8_reload_unsupported() -> bool:
    if not current_platform.supports_fp8():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a
        return on_gfx90a()
    return False


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("Qwen/Qwen3-0.6B"),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_storage_identity_bf16(model, vllm_runner):
    """Unmanaged tensor addresses must not drift after BF16 reload."""
    trust = "DeepSeek" in model
    _run_storage_identity_check(vllm_runner, model,
                                trust_remote_code=trust)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("Qwen/Qwen3-0.6B"),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_storage_identity_fp8(model, vllm_runner):
    """Unmanaged tensor addresses must not drift after FP8 reload."""
    if _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")
    trust = "DeepSeek" in model
    _run_storage_identity_check(vllm_runner, model, quantization="fp8",
                                trust_remote_code=trust)


@pytest.mark.slow_test
def test_dsv3_reload_perplexity(vllm_runner):
    """DeepSeek-V3 debug reload must change model behavior correctly."""
    base = "inference-optimization/DeepSeek-V3-debug-empty"
    mul_model = "inference-optimization/DeepSeek-V3-debug-multiply"
    add_model = "inference-optimization/DeepSeek-V3-debug-add"

    with vllm_runner(
        model_name=base,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights",
                           kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(
            ["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(
            ["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights",
                           kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(
            ["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(
            ["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp
