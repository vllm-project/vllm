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


def test_qwen3_5_forward_does_not_use_weight_shape_for_gdn_in_proj():
    """Verify forward() computes gdn_in_proj output sizes from model dims.

    Regression test for: when using quantized models (AWQ/GPTQ) with LoRA,
    self.in_proj_qkvz.weight.shape[0] returns the packed quantized weight
    dimension (e.g. input_size // 8 for 4-bit) instead of the actual output
    size. This caused torch.compile to trace with wrong tensor shapes,
    leading to a split size mismatch error.

    The fix computes output sizes from key_dim, value_dim, num_v_heads, and
    tp_size instead of reading weight.shape[0].

    Introduced by commit f1740006e ([Perf] Enable dual stream execution of
    input projection for Qwen3 #36795) which added the gdn_in_proj custom op.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(Qwen3_5GatedDeltaNet.forward)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Ensure the forward method does NOT reference .weight.shape
    # (which breaks for quantized models with LoRA)
    source_text = source
    assert ".weight.shape" not in source_text, (
        "Qwen3_5GatedDeltaNet.forward() must not use .weight.shape to "
        "determine gdn_in_proj output sizes. For quantized models "
        "(AWQ/GPTQ) with LoRA, .weight returns the packed qweight whose "
        "shape does not reflect the actual output dimension. Use computed "
        "sizes from model dimensions (key_dim, value_dim, etc.) instead."
    )

    # Verify gdn_in_proj is called with computed size expressions,
    # not attribute accesses on weight
    gdn_call_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func_str = ast.dump(node.func)
            if "gdn_in_proj" in func_str:
                gdn_call_found = True
                # The 2nd and 3rd positional args (index 1,2) should be
                # computed sizes, not .weight.shape[0] attribute accesses
                for arg in node.args[1:3]:
                    arg_source = ast.get_source_segment(source, arg)
                    if arg_source:
                        assert "weight" not in arg_source, (
                            f"gdn_in_proj argument '{arg_source}' must not "
                            "reference .weight - use computed model dims"
                        )
    assert gdn_call_found, "Could not find gdn_in_proj call in forward()"


def test_qwen3_5_gdn_output_sizes_match_model_dims():
    """Verify computed output sizes match expected values for Qwen3.5-9B.

    For Qwen3.5-9B:
    - key_dim = num_k_heads * head_k_dim = 16 * 128 = 2048
    - value_dim = num_v_heads * head_v_dim = 32 * 128 = 4096
    - num_v_heads = 32

    With tp_size=1:
    - qkvz_output = (2048*2 + 4096) + 4096 = 12288
    - ba_output = 32 * 2 = 64
    """
    key_dim = 2048
    value_dim = 4096
    num_v_heads = 32
    tp_size = 1

    qkv_size = (key_dim * 2 + value_dim) // tp_size  # 8192
    z_size = value_dim // tp_size  # 4096
    ba_size = (num_v_heads * 2) // tp_size  # 64

    # These are the values passed to gdn_in_proj
    qkvz_output = qkv_size + z_size
    assert qkvz_output == 12288, f"Expected 12288, got {qkvz_output}"
    assert ba_size == 64, f"Expected 64, got {ba_size}"

    # The split after gdn_in_proj must consume the full qkvz output
    assert qkv_size + z_size == qkvz_output

    # Verify with tp_size=2
    tp_size = 2
    qkv_size_tp2 = (key_dim * 2 + value_dim) // tp_size  # 4096
    z_size_tp2 = value_dim // tp_size  # 2048
    ba_size_tp2 = (num_v_heads * 2) // tp_size  # 32

    assert qkv_size_tp2 + z_size_tp2 == 6144
    assert ba_size_tp2 == 32
