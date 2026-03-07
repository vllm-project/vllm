# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Qwen3.5 LoRA support.

Validates that the packed_modules_mapping and MergedColumnParallelLinear
output_sizes are aligned for Qwen3.5's GDN (Gated Delta Network) layers,
which use fused projections (in_proj_qkvz and in_proj_ba).

Regression test for: IndexError in set_lora when output_sizes had 4 slices
but packed_modules_mapping only had 2 entries.
"""

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5MoeForCausalLM,
)


def test_qwen3_5_packed_modules_output_sizes_alignment():
    """Verify packed_modules_mapping entries match output_sizes slices count.

    The LoRA system requires that len(packed_modules_mapping[key]) equals
    len(output_sizes) for the corresponding MergedColumnParallelLinear.
    A mismatch causes IndexError in set_lora.
    """
    mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping

    # in_proj_qkvz should map to 2 sub-modules: in_proj_qkv, in_proj_z
    assert "in_proj_qkvz" in mapping
    assert mapping["in_proj_qkvz"] == ["in_proj_qkv", "in_proj_z"]

    # in_proj_ba should map to 2 sub-modules: in_proj_b, in_proj_a
    assert "in_proj_ba" in mapping
    assert mapping["in_proj_ba"] == ["in_proj_b", "in_proj_a"]


def test_qwen3_5_create_qkvz_proj_output_sizes():
    """Verify create_qkvz_proj produces output_sizes with 2 slices.

    The key_dim * 2 + value_dim formula represents the combined Q+K+V
    projection (in_proj_qkv in the HuggingFace checkpoint), and value_dim
    represents the Z projection (in_proj_z).

    For the Qwen3.5-9B model: key_dim=2048, value_dim=4096.
    The HF checkpoint has in_proj_qkv.weight shape [8192, 4096] and
    in_proj_z.weight shape [4096, 4096].
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(Qwen3_5GatedDeltaNet.create_qkvz_proj)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Find the output_sizes keyword argument in MergedColumnParallelLinear()
    output_sizes_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "output_sizes":
            output_sizes_node = node.value
            break

    assert output_sizes_node is not None, "Could not find output_sizes kwarg"
    assert isinstance(output_sizes_node, ast.List), (
        "output_sizes should be a list literal"
    )

    # Must have exactly 2 elements to match packed_modules_mapping
    num_slices = len(output_sizes_node.elts)
    num_packed = len(Qwen3_5ForCausalLMBase.packed_modules_mapping["in_proj_qkvz"])
    assert num_slices == num_packed, (
        f"output_sizes has {num_slices} slices but packed_modules_mapping "
        f"has {num_packed} entries for in_proj_qkvz. These must match."
    )

    # Verify with concrete values: key_dim=2048, value_dim=4096
    key_dim, value_dim = 2048, 4096
    expected_sizes = [key_dim * 2 + value_dim, value_dim]
    assert expected_sizes == [8192, 4096]
    assert sum(expected_sizes) == key_dim * 2 + value_dim * 2


def test_qwen3_5_conditional_generation_packed_mapping():
    """Verify multimodal variant also has correct GDN packed mapping."""
    mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    # Should include the GDN mappings
    assert mapping["in_proj_qkvz"] == ["in_proj_qkv", "in_proj_z"]
    assert mapping["in_proj_ba"] == ["in_proj_b", "in_proj_a"]

    # Should also include standard attention and MLP mappings
    assert "qkv_proj" in mapping
    assert "gate_up_proj" in mapping


def test_qwen3_5_moe_inherits_packed_mapping():
    """Verify MoE variant inherits the same packed_modules_mapping."""
    dense_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping
    moe_mapping = Qwen3_5MoeForCausalLM.packed_modules_mapping

    # MoE should have identical GDN mappings as dense
    assert moe_mapping["in_proj_qkvz"] == dense_mapping["in_proj_qkvz"]
    assert moe_mapping["in_proj_ba"] == dense_mapping["in_proj_ba"]


def test_qwen3_5_stacked_params_shard_ids():
    """Verify load_weights stacked_params_mapping uses correct shard IDs.

    With output_sizes=[qkv_size, z_size], shard IDs must be integers 0 and 1.
    Tuple shard IDs like (0, 1, 2) would indicate a mismatch with output_sizes.
    """
    import ast
    import inspect
    import textwrap

    # stacked_params_mapping is in Qwen3_5Model.load_weights (the inner model)
    source = inspect.getsource(Qwen3_5Model.load_weights)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Find the stacked_params_mapping list assignment
    stacked_mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "stacked_params_mapping"
                ):
                    stacked_mapping = ast.literal_eval(node.value)
                    break

    assert stacked_mapping is not None, "Could not find stacked_params_mapping"

    # Find the in_proj_qkvz entries
    qkvz_entries = [e for e in stacked_mapping if e[0] == "in_proj_qkvz"]
    assert len(qkvz_entries) == 2, (
        f"Expected 2 in_proj_qkvz entries, got {qkvz_entries}"
    )

    # Shard IDs must be simple integers 0 and 1 (not tuples)
    shard_ids = {e[2] for e in qkvz_entries}
    assert shard_ids == {0, 1}, (
        f"in_proj_qkvz shard IDs should be {{0, 1}}, got {shard_ids}. "
        "Tuple shard IDs like (0,1,2) indicate a mismatch with output_sizes."
    )

    # Verify in_proj_ba shard IDs
    ba_entries = [e for e in stacked_mapping if e[0] == "in_proj_ba"]
    assert len(ba_entries) == 2
    ba_shard_ids = {e[2] for e in ba_entries}
    assert ba_shard_ids == {0, 1}
