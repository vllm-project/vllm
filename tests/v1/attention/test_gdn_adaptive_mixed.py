# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN metadata builder under adaptive verification trimming.

`supports_device_cpu_query_lens_mismatch()` declares that the builder accepts a
device-side `query_start_loc` that adaptive verification rewrote below the host
plan (device total < host total) — exactly what
`AdaptiveVerificationManager.reallocate_drafts()` produces when the draft budget
binds. Building a batch that mixes a speculative verify row with a prefill row
used to size `torch.repeat_interleave` with the HOST token total while the
repeats sum to the DEVICE total, tripping the Repeat.cu device-side assert and
poisoning the CUDA context: a hard engine crash, not a recoverable error.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
PREFIX = "model.layers.0.linear_attn"

if not (
    current_platform.is_rocm()
    or (current_platform.is_cuda() and current_platform.has_device_capability(80))
):
    pytest.skip(
        reason="GDN metadata build requires ROCm or CUDA compute capability 8.0+.",
        allow_module_level=True,
    )


def _build_with_device_query_start_loc(
    device_query_start_loc: torch.Tensor,
):
    """Build metadata for one verify row (1 bonus + 3 drafts = 4 scheduled
    tokens, host plan [0, 4, 132]) plus one prefill row (128 tokens), with the
    device-side `query_start_loc` rewritten by adaptive verification."""
    device = torch.device("cuda")
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": 128},
    )
    vllm_config.additional_config = {
        "gdn_prefill_backend": "triton" if current_platform.is_rocm() else "cutedsl"
    }
    # "none" keeps the block table a plain (#requests, 1 + num_speculative_blocks)
    # view; the default "align" mode needs the precomputed aligned state indices
    # (see test_update_block_table_matches_build), which is orthogonal here.
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=NUM_SPEC,
    )

    batch = BatchSpec(seq_lens=[65, 200], query_lens=[4, 128])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common = common.replace(query_start_loc=device_query_start_loc.to(device))
    common.block_table_tensor.add_(1)

    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    with set_current_vllm_config(vllm_config):
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=device),
            num_decode_draft_tokens_cpu=torch.tensor([3, -1], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "device_query_start_loc",
    [
        # Adaptive verification trimmed the verify row 4 -> 2 (1 admitted draft
        # + 1 bonus); the prefill row keeps its scheduled window. Device total
        # (130) < host total (132).
        pytest.param(
            torch.tensor([0, 2, 130], dtype=torch.int32), id="adaptive-trimmed"
        ),
        # Control: device == host plan; must keep building as before.
        pytest.param(torch.tensor([0, 4, 132], dtype=torch.int32), id="no-trim"),
    ],
)
def test_gdn_adaptive_trimmed_mixed_batch_builds(device_query_start_loc):
    meta = _build_with_device_query_start_loc(device_query_start_loc)
    assert meta.num_spec_decodes == 1
    assert meta.num_prefills == 1
