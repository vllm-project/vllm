# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every DSpark model class the V2 speculator can load must expose the
sampling hooks it calls. The greedy branch of DSparkSpeculator._sample_logits
calls model.map_draft_to_target on every profile_run (draft_logits is None
during profiling), so a missing hook is a cold-boot blocker, not an edge case
-- upstream #49969 added the hooks to amd/, xpu/, kimi_k3 and qwen3_dspark
but not the nvidia deepseek_v4 class (reported by alexbi29 in
vllm-project/vllm#41834)."""

import inspect

from vllm.models.deepseek_v4.nvidia import dspark as nvidia_dspark


def test_nvidia_dspark_exposes_v2_speculator_hooks():
    cls = nvidia_dspark.DSparkDeepseekV4ForCausalLM
    for hook in ("map_draft_to_target", "compute_draft_logits"):
        assert hasattr(cls, hook), (
            f"{cls.__name__} lacks {hook}; the V2 DSpark speculator dies in "
            "profile_run on the first cold boot"
        )


def test_map_draft_to_target_is_identity_for_full_vocab():
    src = inspect.getsource(
        nvidia_dspark.DSparkDeepseekV4ForCausalLM.map_draft_to_target
    )
    assert "return draft_ids" in src
