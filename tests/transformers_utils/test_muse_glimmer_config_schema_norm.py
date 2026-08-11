# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MuseGlimmer attention config-schema normalization (native flat vs modular).

Regression test for the modular-config degenerate-output bug: the modular HF
text_config OMITS use_qk_norm / use_attn_output_gate (read as None) and ships a
PRE-FOLDED qk_scale_factor (43.784/sqrt(128)=3.87). vLLM must treat missing
flags as True (MuseGlimmer always applies QK-norm + output gate) and normalize the
query pre-scale so native (raw 43.784) and modular (folded 3.87) converge.
"""

import math

from vllm.model_executor.models.muse_glimmer import (
    _muse_glimmer_query_prescale,
    _muse_glimmer_use_attn_output_gate,
    _muse_glimmer_use_qk_norm,
)

HEAD_DIM = 128
SQRT_HD = math.sqrt(HEAD_DIM)
NATIVE = 43.7840518911
FOLDED = NATIVE / SQRT_HD  # 3.8700...


class Cfg:
    def __init__(self, **kw):
        self.head_dim = HEAD_DIM
        for k, v in kw.items():
            setattr(self, k, v)


def test_qk_norm_missing_defaults_true():
    assert _muse_glimmer_use_qk_norm(Cfg(use_qk_norm=None)) is True  # modular
    assert _muse_glimmer_use_qk_norm(Cfg()) is True  # absent
    assert _muse_glimmer_use_qk_norm(Cfg(use_qk_norm=True)) is True  # native
    assert _muse_glimmer_use_qk_norm(Cfg(use_qk_norm=False)) is False  # explicit off


def test_output_gate_missing_defaults_true():
    assert _muse_glimmer_use_attn_output_gate(Cfg(use_attn_output_gate=None)) is True
    assert _muse_glimmer_use_attn_output_gate(Cfg()) is True
    assert _muse_glimmer_use_attn_output_gate(Cfg(use_attn_output_gate=False)) is False


def test_query_prescale_native_and_modular_converge():
    # Both schemas must yield the SAME final scale_query_by (~3.87).
    assert (
        abs(_muse_glimmer_query_prescale(Cfg(qk_scale_factor=NATIVE)) - FOLDED) < 1e-9
    )
    assert (
        abs(_muse_glimmer_query_prescale(Cfg(qk_scale_factor=FOLDED)) - FOLDED) < 1e-9
    )


def test_query_prescale_explicit_wins():
    c = Cfg(scale_query_by=FOLDED, qk_scale_factor=NATIVE)
    assert abs(_muse_glimmer_query_prescale(c) - FOLDED) < 1e-9


if __name__ == "__main__":
    test_qk_norm_missing_defaults_true()
    test_output_gate_missing_defaults_true()
    test_query_prescale_native_and_modular_converge()
    test_query_prescale_explicit_wins()
    print("ALL MUSE_GLIMMER CONFIG-SCHEMA NORM TESTS PASSED")
