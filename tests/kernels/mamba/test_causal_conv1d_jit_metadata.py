# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def _get_do_not_specialize_on_alignment_args(
    source_path: Path, func_name: str
) -> set[str]:
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "jit"
                ):
                    for keyword in decorator.keywords:
                        if keyword.arg != "do_not_specialize_on_alignment":
                            continue
                        if not isinstance(keyword.value, ast.List):
                            raise AssertionError(
                                "Expected do_not_specialize_on_alignment to be a list"
                            )
                        return {
                            elt.value
                            for elt in keyword.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        }
    raise AssertionError(f"Could not find {func_name} triton.jit decorator")


def test_causal_conv1d_metadata_ptrs_not_alignment_specialized() -> None:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
    )

    fwd_alignment_exempt = _get_do_not_specialize_on_alignment_args(
        source_path, "_causal_conv1d_fwd_kernel"
    )
    assert {
        "cache_indices_ptr",
        "has_initial_states_ptr",
        "query_start_loc_ptr",
        "batch_ptr",
        "token_chunk_offset_ptr",
        "block_idx_first_scheduled_token",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "num_computed_tokens",
    }.issubset(fwd_alignment_exempt)

    update_alignment_exempt = _get_do_not_specialize_on_alignment_args(
        source_path, "_causal_conv1d_update_kernel"
    )
    assert {
        "conv_state_indices_ptr",
        "num_accepted_tokens_ptr",
        "query_start_loc_ptr",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
    }.issubset(update_alignment_exempt)
