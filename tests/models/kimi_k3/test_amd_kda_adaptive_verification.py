# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 KDA metadata under adaptive verification.

Adaptive verification splits the draft budget evenly on the host, then trims
each verify request on device. The KDA spec path must therefore plan from the
device offsets, never from host lengths that merely look uniform.
"""

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.kimi_k3.amd.kda_metadata import (
    KimiK3ROCmKDABackend,
    KimiK3ROCmKDAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
NUM_SPEC = 3


def _config(adaptive: bool):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=NUM_SPEC,
        enable_adaptive_verification=adaptive,
    )
    vllm_config.cache_config.mamba_cache_mode = "none"
    return vllm_config


def _spec() -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=NUM_SPEC,
    )


def _builder(adaptive: bool) -> KimiK3ROCmKDAMetadataBuilder:
    return KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=_spec(),
        layer_names=["layer.0"],
        vllm_config=_config(adaptive),
        device=DEVICE,
    )


def _trimmed_verify_batch(builder: KimiK3ROCmKDAMetadataBuilder):
    """Host sees [3, 3]; the device was trimmed to [4, 2] (same total)."""
    batch = BatchSpec(seq_lens=[64, 64], query_lens=[4, 2])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    common = common.replace(
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32)
    )
    accepted = torch.ones(2, dtype=torch.int32, device=DEVICE)
    num_decode_draft_tokens_cpu = torch.tensor([NUM_SPEC, NUM_SPEC], dtype=torch.int32)
    return builder.build(0, common, accepted, num_decode_draft_tokens_cpu)


def test_backend_and_builder_opt_in_only_under_adaptive_verification():
    assert KimiK3ROCmKDABackend.supports_device_cpu_query_lens_mismatch()
    bound = KimiK3ROCmKDAMetadataBuilder.get_varlen_cudagraph_max_query_len
    assert bound(_config(adaptive=True), _spec()) == NUM_SPEC + 1
    assert bound(_config(adaptive=False), _spec()) is None


def test_adaptive_verification_drops_the_host_uniform_length():
    """Equal host lengths would select the fixed-length recurrent kernel,
    which places sequence i at i * L and ignores the device cu_seqlens."""
    meta = _trimmed_verify_batch(_builder(adaptive=True))
    assert meta.num_spec_decodes == 2
    assert meta.uniform_spec_sequence_length is None
    assert torch.equal(
        meta.spec_query_start_loc.cpu(), torch.tensor([0, 4, 6], dtype=torch.int32)
    )
    # Without adaptive verification the host lengths are exact and kept.
    meta = _trimmed_verify_batch(_builder(adaptive=False))
    assert meta.uniform_spec_sequence_length == 3
