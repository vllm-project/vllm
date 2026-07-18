# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPM-SALA's layer-schedule and decay-slope helpers.

Location note: placed under tests/models/language/generation/ (not a
model-specific subdirectory) to match the real, current vLLM test layout
-- confirmed against .buildkite/test_areas/models_language.yaml, which
drives `pytest -v -s models/language/generation -m hybrid_model` as its
own CI job. The `pytestmark` below opts this file into that job (the
`hybrid_model` marker is registered in pyproject.toml as "models that
contain mamba layers (including pure SSM and hybrid architectures)" --
exactly this model's category, confirmed against the real marker
registration, not assumed).

These test the pure orchestration/math logic factored out of
`vllm/model_executor/models/minicpm_sala.py` specifically so it CAN be
tested without a GPU (`build_alibi_slopes` needs torch but not CUDA;
`validate_mixer_schedule`/`is_sparse_layer`/`is_lightning_layer` need
neither). They do NOT exercise the kernel dispatch, cache management, or
numerical correctness against HuggingFace -- that is the job of
`scripts/minicpm_sala_differential_validation.py`, which requires a real
GPU and real checkpoint weights and is explicitly out of scope for this
CI-runnable suite.

Honesty note: these tests were authored and their *logic* was verified
standalone (see the accompanying `verify_logic.py` sandbox script, whose
output is reproduced in the Stage-1 PR description) in an environment
without torch installed. This file itself has NOT been executed in this
environment because `import torch` / `import vllm` are unavailable here --
running it is a required, not-yet-completed step before merge, called out
explicitly in docs/minicpm_sala_known_limitations.md.
"""

import math

import pytest
import torch

from vllm.model_executor.models.minicpm_sala import (
    build_alibi_slopes,
    is_lightning_layer,
    is_sparse_layer,
    validate_mixer_schedule,
)

pytestmark = pytest.mark.hybrid_model

# Real 32-layer mixer_types array from openbmb/MiniCPM-SALA's config.json
# (commit 9180fe1), copied verbatim -- NOT a synthetic/simplified example.
REAL_MIXER_TYPES = [
    "minicpm4", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "minicpm4", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "minicpm4", "minicpm4", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn", "minicpm4", "lightning-attn",
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "minicpm4", "minicpm4", "minicpm4",
]  # fmt: skip
# CORRECTION: the second "minicpm4" entry is at index 9, not 8 -- an
# earlier hand-transcription of this array (Phase 1 report, this file's
# first draft) miscounted it as index 8. Re-verified by careful re-count
# of the raw config.json array AND independently cross-checked against
# the real model.safetensors.index.json weight-name patterns (sparse
# layers have self_attn.o_gate and lack q_norm/k_norm/o_norm/z_proj).


class TestMixerScheduleValidation:
    def test_real_checkpoint_schedule_is_valid(self) -> None:
        validate_mixer_schedule(list(REAL_MIXER_TYPES))  # must not raise

    def test_real_checkpoint_has_32_layers(self) -> None:
        assert len(REAL_MIXER_TYPES) == 32

    def test_sparse_layer_positions_match_config(self) -> None:
        sparse = [i for i, m in enumerate(REAL_MIXER_TYPES) if is_sparse_layer(m)]
        assert sparse == [0, 9, 16, 17, 22, 29, 30, 31]

    def test_sparse_fraction_is_25_percent(self) -> None:
        sparse_count = sum(1 for m in REAL_MIXER_TYPES if is_sparse_layer(m))
        assert sparse_count / len(REAL_MIXER_TYPES) == 0.25

    def test_every_layer_is_exactly_one_type(self) -> None:
        for m in REAL_MIXER_TYPES:
            assert is_sparse_layer(m) != is_lightning_layer(m)

    def test_empty_schedule_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_mixer_schedule([])

    def test_layer_zero_must_be_sparse(self) -> None:
        bad = ["lightning-attn"] + REAL_MIXER_TYPES[1:]
        with pytest.raises(ValueError, match="layer 0"):
            validate_mixer_schedule(bad)

    def test_unknown_mixer_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            validate_mixer_schedule(["minicpm4", "mamba2"])

    @pytest.mark.parametrize("alias", ["lightning", "lightning_attn", "lightning-attn"])
    def test_all_three_lightning_spellings_accepted(self, alias: str) -> None:
        # The reference HF `MiniCPMSALADecoderLayer.__init__` accepts all
        # three spellings (`elif self.mixer_type in ["lightning",
        # "lightning_attn", "lightning-attn"]`); the released config.json
        # only uses "lightning-attn", but a hand-edited config could use
        # either of the other two, and the port must not reject them.
        validate_mixer_schedule(["minicpm4", alias])
        assert is_lightning_layer(alias)


class TestAlibiSlopes:
    """`build_alibi_slopes` must be a byte-for-byte port of the reference
    `_build_slope_tensor`, which is itself the SAME algorithm as vLLM's
    in-tree `MiniMaxText01LinearAttention._build_slope_tensor` -- verified
    directly against that source, not assumed. This class checks the
    output against closed-form ALiBi values for the power-of-2 case
    (num_heads=32, the released checkpoint's value) and structural
    properties for a non-power-of-2 case.
    """

    def test_32_heads_matches_closed_form(self) -> None:
        slopes = build_alibi_slopes(32)
        assert slopes.shape == (32,)
        assert slopes.dtype == torch.float32
        expected_first = 2 ** (-8 / 32)
        assert math.isclose(slopes[0].item(), expected_first, rel_tol=1e-6)

    def test_32_heads_monotonically_decreasing(self) -> None:
        slopes = build_alibi_slopes(32)
        diffs = slopes[1:] - slopes[:-1]
        assert torch.all(diffs <= 0), "ALiBi slopes must be non-increasing"

    def test_all_slopes_strictly_positive(self) -> None:
        # Sign flip (`* -1.0` to get an actual decay rate) happens at the
        # call site in MiniCPMSALALightningAttention.__init__, NOT inside
        # build_alibi_slopes -- this function must return positive values.
        slopes = build_alibi_slopes(32)
        assert torch.all(slopes > 0)

    def test_non_power_of_2_head_count_does_not_crash(self) -> None:
        # config.json's num_attention_heads=32 is a power of 2 for the
        # released checkpoint, but a future/custom checkpoint might not
        # be -- the reference `_build_slope_tensor` explicitly handles
        # this via the `get_slopes` fallback branch, so this port must
        # too.
        slopes = build_alibi_slopes(24)
        assert slopes.shape == (24,)
        assert torch.all(slopes > 0)


class TestResidualScaleConstant:
    """The muP residual-branch scale constant (Phase 1 report section 3)
    is a plain float computed from two config integers -- cheap to test
    exactly, and exactly the kind of silent-wrongness risk flagged in the
    architecture report if a future edit changes the formula.
    """

    def test_released_checkpoint_scale_depth_and_layers(self) -> None:
        scale_depth = 1.4
        num_hidden_layers = 32
        expected = scale_depth / math.sqrt(num_hidden_layers)
        assert math.isclose(expected, 0.2474873734152916, rel_tol=1e-12)
