# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A reload must not silently change host-scalar attention scales that a FULL
decode CUDA graph has captured: those checks raise instead."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flashinfer import FlashInferImpl


class TestFlashInferHostScales:
    @pytest.mark.parametrize(
        "full,dcp,old,new,raises",
        [
            (True, False, (0.1, 1.0, 2.0), (0.1, 1.0, 2.0), False),
            (True, False, (0.1, 1.0, 2.0), (0.1, 1.0, 4.0), True),
            (True, False, (0.1, 1.0, None), (0.2, 0.5, None), False),
            (True, True, (0.1, 1.0, None), (0.2, 1.0, None), True),
            (False, True, (0.1, 1.0, 2.0), (0.2, 0.5, 4.0), False),
            (True, False, (None, None, None), (0.1, 1.0, 2.0), False),
        ],
        ids=[
            "same",
            "o-sf-changed",
            "bmm-on-device",
            "dcp-bmm-changed",
            "no-full-graph",
            "cold-load",
        ],
    )
    def test_reload(self, full, dcp, old, new, raises):
        bmm1, bmm2, o_sf = new
        impl = SimpleNamespace(
            _full_decode_graphs=full,
            _dcp=dcp,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            o_sf_scale=o_sf,
        )
        layer = SimpleNamespace(layer_name="layers.0.attn")
        if raises:
            with pytest.raises(RuntimeError, match="host scalar"):
                FlashInferImpl._check_host_scales(impl, layer, old)
        else:
            FlashInferImpl._check_host_scales(impl, layer, old)

    def test_post_load_keeps_scales_for_the_check(self):
        """The check compares with the previous load's scales, so the impl's
        post-load hook (run before the refresh) must not clear them."""
        impl = object.__new__(FlashInferImpl)
        impl.bmm1_scale, impl.bmm2_scale, impl._sinks_source = 0.1, 1.0, None
        impl.process_weights_after_loading(torch.bfloat16)
        assert (impl.bmm1_scale, impl.bmm2_scale) == (0.1, 1.0)
