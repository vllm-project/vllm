# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement compatibility policy: own-KV speculators with satisfiable
feature requirements are accepted, everything else is rejected with an
explanatory reason."""

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
    placement_incompatibilities,
    remote_draft_config_incompatibilities,
)

EAGLE3_FEATURES = frozenset(
    {
        TargetFeatureKind.TOKEN_IDS,
        TargetFeatureKind.POSITIONS,
        TargetFeatureKind.QUERY_LENS,
        TargetFeatureKind.AUX_HIDDEN_STATES,
    }
)


def make_capabilities(**overrides) -> SpeculatorPlacementCapabilities:
    defaults = dict(
        state_dependency="own_kv",
        required_features=tuple(sorted(EAGLE3_FEATURES)),
        standalone_weights="materializable",
        supports_parallel_drafting=True,
    )
    defaults.update(overrides)
    return SpeculatorPlacementCapabilities(**defaults)


def test_own_kv_speculator_accepted():
    assert (
        placement_incompatibilities(
            make_capabilities(),
            parallel_drafting=True,
            draft_sample_method="greedy",
            provided_features=EAGLE3_FEATURES,
        )
        == []
    )


def test_target_kv_sharing_rejected():
    reasons = placement_incompatibilities(
        make_capabilities(state_dependency="target_kv"),
        parallel_drafting=False,
        draft_sample_method="greedy",
        provided_features=EAGLE3_FEATURES,
    )
    assert len(reasons) == 1
    assert "target-owned state" in reasons[0]


def test_missing_standalone_weights_rejected():
    reasons = placement_incompatibilities(
        make_capabilities(standalone_weights="missing"),
        parallel_drafting=False,
        draft_sample_method="greedy",
        provided_features=EAGLE3_FEATURES,
    )
    assert any("materialize" in r for r in reasons)


def test_unprovidable_features_reported_by_name():
    reasons = placement_incompatibilities(
        make_capabilities(
            required_features=(
                TargetFeatureKind.TOKEN_IDS,
                TargetFeatureKind.NEXT_PREFILL_TOKENS,
            )
        ),
        parallel_drafting=False,
        draft_sample_method="greedy",
        provided_features={TargetFeatureKind.TOKEN_IDS},
    )
    assert len(reasons) == 1
    assert "next_prefill_tokens" in reasons[0]


def test_unknown_future_feature_kind_rejected_by_name():
    # A newer-minor server may require a kind this build does not know;
    # it must surface as an unprovidable feature, not a decode error.
    reasons = placement_incompatibilities(
        make_capabilities(
            required_features=(TargetFeatureKind.TOKEN_IDS, "rope_scales")
        ),
        parallel_drafting=False,
        draft_sample_method="greedy",
        provided_features={TargetFeatureKind.TOKEN_IDS},
    )
    assert len(reasons) == 1
    assert "rope_scales" in reasons[0]


def test_multi_module_requires_checkpoint_support():
    reasons = placement_incompatibilities(
        make_capabilities(supports_multi_module=False),
        parallel_drafting=False,
        draft_sample_method="greedy",
        provided_features=EAGLE3_FEATURES,
        uses_multi_module=True,
    )
    assert any("multi-module" in r for r in reasons)


def test_parallel_drafting_requires_checkpoint_support():
    reasons = placement_incompatibilities(
        make_capabilities(supports_parallel_drafting=False),
        parallel_drafting=True,
        draft_sample_method="greedy",
        provided_features=EAGLE3_FEATURES,
    )
    assert any("parallel_drafting" in r for r in reasons)


def test_probabilistic_draft_requires_probability_contract():
    reasons = placement_incompatibilities(
        make_capabilities(),
        parallel_drafting=False,
        draft_sample_method="probabilistic",
        provided_features=EAGLE3_FEATURES,
    )
    assert any("probability contract" in r for r in reasons)


def test_config_time_supported_methods():
    for method in ("eagle", "eagle3", "mtp"):
        assert (
            remote_draft_config_incompatibilities(
                method=method,
                draft_tensor_parallel_size=1,
                draft_sample_method="greedy",
                uses_target_kv=False,
            )
            == []
        )


def test_config_time_rejections_are_cumulative():
    reasons = remote_draft_config_incompatibilities(
        method="dflash",
        draft_tensor_parallel_size=2,
        draft_sample_method="probabilistic",
        uses_target_kv=True,
    )
    assert len(reasons) == 4
    joined = "\n".join(reasons)
    assert "dflash" in joined
    assert "target KV" in joined
    assert "draft_tensor_parallel_size" in joined
    assert "greedy" in joined


def test_config_time_unresolved_draft_tp_allowed():
    # draft_tensor_parallel_size may still be None at validation time.
    assert (
        remote_draft_config_incompatibilities(
            method="mtp",
            draft_tensor_parallel_size=None,
            draft_sample_method="greedy",
            uses_target_kv=False,
        )
        == []
    )


def test_config_time_unknown_kv_sharing_fails_closed():
    # If the MTP variant cannot be determined, remote placement is refused
    # rather than assuming the draft keeps its own KV.
    reasons = remote_draft_config_incompatibilities(
        method="mtp",
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        uses_target_kv=None,
    )
    assert len(reasons) == 1
    assert "cannot determine" in reasons[0]
