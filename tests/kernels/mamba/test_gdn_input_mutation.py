# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN input mutations must survive graph export and functionalization."""

import pytest
import torch

# Register the real GDN custom ops; export uses their registered fake kernels.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn  # noqa: F401


class GDNExport(torch.nn.Module):
    def __init__(self, packed: bool):
        super().__init__()
        self.packed = packed

    def forward(self, qkv, b, a, out):
        if self.packed:
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                qkv, b, out, "layers.0"
            )
        else:
            torch.ops.vllm.qwen_gdn_attention_core(qkv, b, a, out, "layers.0")
        return qkv + 1, out + 1


@pytest.mark.parametrize("packed", [False, True])
def test_export_preserves_qkv_mutation(packed):
    inputs = tuple(torch.zeros(2, 4) for _ in range(4))
    exported = torch.export.export(GDNExport(packed), inputs).run_decompositions()
    assert "qkv" in exported.graph_signature.user_inputs_to_mutate.values()
