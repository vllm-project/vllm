# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.speculative import SpeculativeConfig


def make_config(**kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=4,
        prompt_lookup_max=3,
        prompt_lookup_min=2,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Selects a different runner, and EAGLE models register an extra
        # buffer and branch in forward when it is set.
        ("parallel_drafting", True),
        # Builds the draft's own ParallelConfig; tensor parallel size is
        # graph-relevant for ParallelConfig.
        ("draft_tensor_parallel_size", 2),
        # Cloned into the draft's sub-configs by apply_draft_overrides, so the
        # hashes VllmConfig folds in never observe them.
        ("moe_backend", "triton"),
        ("kv_cache_dtype", "fp8"),
        # Allocates a per-request draft logits buffer in the speculator.
        ("draft_sample_method", "probabilistic"),
        # Only supported by the dspark path, and rejected for Model Runner V1.
        ("enable_adaptive_verification", True),
    ],
)
def test_draft_execution_fields_change_compilation_hash(field: str, value):
    base = make_config()
    changed = make_config(**{field: value})

    assert getattr(changed, field) == value, "field did not take effect"
    assert base.compute_hash() != changed.compute_hash(), (
        f"{field} changes how the draft executes but not the config hash"
    )


def test_draft_attention_backend_override_is_configurable():
    """`attention_backend` is part of _DRAFT_VLLM_CONFIG_OVERRIDES; kept as a
    separate case because its value space is an enum rather than a string."""
    base = make_config()
    changed = make_config(attention_backend="FLASH_ATTN")

    assert base.compute_hash() != changed.compute_hash()
