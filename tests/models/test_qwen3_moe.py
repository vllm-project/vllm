# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP


def test_qwen3_moe_mlp_threads_is_sequence_parallel_to_disable_tp() -> None:
    mlp = Qwen3MoeMLP(
        hidden_size=128,  # Arbitrary; only wiring is checked
        intermediate_size=256,  # Arbitrary; only wiring is checked
        hidden_act="silu",  # Placeholder; only value the ctor accepts
        # With `reduce_results` left at its default of `True`, the `down_proj` layer
        # would all-reduce and misses the bug we're covering here (that an un-reduced
        # partial-sum output is the input to the offending all_gather)
        reduce_results=False,
        # This test covers that this SP flag propagates to disable TP on both
        # of the MLP's linear layers (`gate_up_proj` and `down_proj`)
        is_sequence_parallel=True,
    )

    assert mlp.gate_up_proj.disable_tp, (
        "gate_up_proj is still tensor-parallel under is_sequence_parallel: "
        "weights would be column-sharded across TP and the shared-expert "
        "output for each SP chunk would be wrong. With TP disabled on the "
        "layer, each rank instead holds the full replicated weights and "
        "processes its sequence-parallel chunk independently."
    )
    assert mlp.down_proj.disable_tp, (
        "down_proj is still tensor-parallel under is_sequence_parallel: "
        "weights would be row-sharded and, with reduce_results=False, each "
        "rank would emit a 1/TP partial sum. With TP disabled on the layer, "
        "each rank instead holds the full replicated weights and processes "
        "its sequence-parallel chunk independently."
    )
