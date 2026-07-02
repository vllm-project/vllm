# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_GEMMA = REPO_ROOT / "vllm" / "model_executor" / "models" / "diffusion_gemma.py"
MODEL_RUNNER = REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    if isinstance(func, ast.Name):
        return func.id
    return ""


def test_dense_compiled_sampler_preserves_upstream_rng_draw_order():
    tree = _module(DIFFUSION_GEMMA)
    compiled = _function(tree, "_compiled_sample_step")

    arg_names = [arg.arg for arg in compiled.args.args]
    assert "random_tokens" not in arg_names

    calls = [
        (_call_name(node), node.lineno)
        for node in ast.walk(compiled)
        if isinstance(node, ast.Call)
    ]
    rand_like_line = min(line for name, line in calls if name == "torch.rand_like")
    randint_line = min(line for name, line in calls if name == "torch.randint")
    assert rand_like_line < randint_line


def test_dense_tp_fallback_syncs_canvas_after_rank_local_rng():
    source = DIFFUSION_GEMMA.read_text()
    compiled_call = source.index("scaled = _compiled_sample_step(")
    sync_call = source.index(
        "states.canvas[decode_slots] = _tp_broadcast_from_rank0(",
        compiled_call,
    )
    draft_sync = source.index(
        "self.req_states.draft_tokens[all_slots, :CL] = states.canvas[",
        sync_call,
    )
    assert sync_call > compiled_call
    assert draft_sync > sync_call


def test_diffusion_gemma_local_logits_are_context_gated():
    tree = _module(DIFFUSION_GEMMA)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DiffusionGemmaForConditionalGeneration"
    )
    compute_logits = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_logits"
    )
    source = ast.unparse(compute_logits)
    assert "_allow_local_vocab_logits" in source
    assert "_DIFFUSION_GEMMA_LOCAL_VOCAB_SAMPLER and self._allow_local_vocab_logits" in source


def test_model_runner_only_enables_local_logits_without_full_vocab_consumers():
    source = MODEL_RUNNER.read_text()
    assert "enable_local_vocab_logits" in source
    assert "grammar_output is None and self.rejection_sampler is None" in source


def test_local_vocab_argmax_empty_shard_returns_losing_sentinel():
    tree = _module(DIFFUSION_GEMMA)
    helper = _function(tree, "_local_vocab_argmax_tokens")
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(DIFFUSION_GEMMA), "exec"), namespace)

    values, indices = namespace["_local_vocab_argmax_tokens"](
        torch.empty(2, 0),
        vocab_start_index=128,
        local_vocab_width=0,
    )

    assert values.shape == (2,)
    assert indices.shape == (2,)
    assert torch.isneginf(values).all()
    assert torch.equal(indices, torch.full((2,), torch.iinfo(torch.int64).max))
