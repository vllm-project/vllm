# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter gate must stay a pure function of TP-rank-identical state.

_input_fits_in_drafter decides whether the drafter (and its TP collectives)
runs. Its inputs derive from scheduler-broadcast state; the rank-local
acceptance correction is applied only to the GPU num_computed_tokens buffer.
If a rank-local value ever feeds the gate, TP ranks can disagree within
num_spec_tokens of the max_model_len ceiling and launch mismatched drafter
collectives (the vllm-project/vllm#49027 wedge / corrupt-and-continue class).
These tests pin the two halves of that invariant at the source level, the
same way the sparse short-extend tiering tests pin their call sites."""

import ast
import inspect
import textwrap

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _method_ast(name: str) -> ast.AST:
    src = textwrap.dedent(inspect.getsource(getattr(GPUModelRunner, name)))
    return ast.parse(src)


def _attribute_names(tree: ast.AST) -> set[str]:
    return {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def test_gate_reads_only_rank_identical_state():
    """The gate may read scheduler-derived metadata and static config only."""
    tree = _method_ast("_input_fits_in_drafter")
    attrs = _attribute_names(tree)
    forbidden = {
        # rank-local sampler/acceptance state
        "valid_sampled_token_count_gpu",
        "num_accepted_tokens",
        "sampled_token_ids",
        "valid_sampled_tokens_count",
        # GPU-side corrected buffer (rank-local under async spec decode)
        "num_computed_tokens",
    }
    leaked = attrs & forbidden
    assert not leaked, (
        f"_input_fits_in_drafter reads rank-local state {leaked}; TP ranks "
        "can now disagree near the ceiling and wedge (see #49027 class)"
    )


def test_cpu_computed_tokens_never_take_the_gpu_correction():
    """_prepare_inputs must not write the rank-local acceptance correction
    back into num_computed_tokens_cpu_tensor (the gate's input)."""
    tree = _method_ast("_prepare_inputs")
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr == "num_computed_tokens_cpu_tensor"
                ):
                    raise AssertionError(
                        "_prepare_inputs assigns to num_computed_tokens_cpu_"
                        "tensor; the GPU-only correction invariant is broken"
                    )
        # .copy_() / in-place mutation calls on the cpu tensor
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy_", "add_", "sub_", "index_copy_"}
        ):
            for sub in ast.walk(node.func.value):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr == "num_computed_tokens_cpu_tensor"
                ):
                    raise AssertionError(
                        "_prepare_inputs mutates num_computed_tokens_cpu_"
                        "tensor in place; gate determinism is broken"
                    )
