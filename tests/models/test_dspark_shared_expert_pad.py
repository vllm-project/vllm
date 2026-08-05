# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark loader's shared-expert padding must match the convention its own
MoE is built with.

DSpark draft layers construct ``DeepseekV4MoE`` on the non-sequence-parallel
convention unconditionally, but the loader used to gate padding on
``parallel_config.use_sequence_parallel_moe``. Whenever that flag was set the
loader skipped padding block-quantized shared-expert weights that the runtime
layer expected padded, so the standard TP loaders sliced them on the wrong
block boundary — garbled draft output on exactly the EP+TP configurations that
set it.
"""

import ast
import inspect

from vllm.models.deepseek_v4.nvidia import dspark


def _moe_construction_sp_argument() -> ast.expr | None:
    """The use_sequence_parallel argument DSpark passes to DeepseekV4MoE."""
    tree = ast.parse(inspect.getsource(dspark.DeepSeekV4DSparkLayer))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "DeepseekV4MoE":
            continue
        for kw in node.keywords:
            if kw.arg == "use_sequence_parallel":
                return kw.value
    return None


def test_pad_guard_and_moe_construction_read_the_same_constant():
    sp_arg = _moe_construction_sp_argument()
    assert sp_arg is not None, (
        "DSpark builds DeepseekV4MoE without an explicit use_sequence_parallel; "
        "the loader's pad guard then has nothing to stay consistent with"
    )
    assert isinstance(sp_arg, ast.Name)
    assert sp_arg.id == "_DSPARK_USE_SEQUENCE_PARALLEL"

    guard = inspect.getsource(dspark.DSparkDeepseekV4ForCausalLM.__init__)
    assert "_DSPARK_USE_SEQUENCE_PARALLEL" in guard, (
        "pad_shared_expert must derive from the same constant the MoE is "
        "constructed with, not from a parallel-config flag"
    )
    assert "use_sequence_parallel_moe" not in guard


def test_dspark_moe_runs_the_unpadded_sequence_parallel_convention():
    """Pinning the value: padding is required precisely because DSpark's MoE is
    not sequence-parallel."""
    assert dspark._DSPARK_USE_SEQUENCE_PARALLEL is False
