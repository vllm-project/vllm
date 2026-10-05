# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PLE short-conv state (tp-replicated MambaSpec) aliases the first page
of the block-outer HMA layout together with layer-0 GDN, the QSA ring and the
MLA indexer cache, so it is discovered in their region. Its descriptors must
still cover the whole PLE page, not the region's (smaller) block_len."""

from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import NixlAgentMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)

from .utils import create_vllm_config


def _make_csa_linear_kv_cache_config(num_blocks: int = 4):
    mla = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    gdn = MambaSpec(
        block_size=16,
        shapes=((8, 3), (1, 4, 4)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=8, head_size_v=0, dtype=torch.float16
    )
    # Short-conv state wider than the MLA page: (channels, window).
    ple = MambaSpec(
        block_size=16,
        shapes=((64, 6),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        tp_replicated=True,
    )
    assert ple.page_size_bytes > mla.page_size_bytes
    block_stride = max(
        mla.page_size_bytes,
        gdn.page_size_bytes,
        ring.page_size_bytes,
        ple.page_size_bytes,
    )
    return (
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=num_blocks * block_stride,
                    layers=["mla.0", "gdn.0", "ring.0", "ple.0"],
                    layer_stride=num_blocks * block_stride,
                    block_stride=block_stride,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["mla.0"], mla),
                KVCacheGroupSpec(["gdn.0"], gdn),
                KVCacheGroupSpec(["ring.0"], ring),
                KVCacheGroupSpec(["ple.0"], ple),
            ],
        ),
        {"mla": mla, "gdn": gdn, "ring": ring, "ple": ple},
        block_stride,
    )


@pytest.mark.cpu_test
def test_register_kv_caches_ple_descriptors_cover_the_ple_page():
    kv_cache_config, specs, block_stride = _make_csa_linear_kv_cache_config()
    num_blocks = kv_cache_config.num_blocks
    vllm_config = create_vllm_config(block_size=16)
    vllm_config.kv_transfer_config.kv_buffer_device = "cuda"

    fake_backend = MagicMock()
    fake_backend.get_supported_kernel_block_sizes.return_value = [16]
    fake_backend.get_name.return_value = "FLASHMLA"
    fake_backend.full_cls_name.return_value = "fake.FLASHMLA"
    fake_platform = MagicMock()
    fake_platform.device_type = "cuda"
    fake_platform.get_nixl_memory_type.return_value = "VRAM"

    with (
        patch.object(bw, "NixlWrapper"),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[fake_backend]),
        patch.object(bw, "current_platform", fake_platform),
        patch(
            "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
            return_value="DS",
        ),
        set_current_vllm_config(vllm_config),
    ):
        worker = NixlConnectorWorker(vllm_config, "test-engine", kv_cache_config)
        worker.use_mla = True
        worker.nixl_wrapper.get_agent_metadata.return_value = b"fake-agent-metadata"

        # Block-outer layout: every group's page starts at byte 0 of the block.
        backing = torch.zeros(num_blocks, block_stride, dtype=torch.uint8)
        worker.register_kv_caches(
            {
                "mla.0": backing[:, : specs["mla"].page_size_bytes],
                "gdn.0": backing[:, : specs["gdn"].page_size_bytes],
                "ring.0": backing[:, : specs["ring"].page_size_bytes],
                "ple.0": backing[:, : specs["ple"].page_size_bytes],
            }
        )

    # The PLE page is discovered in the region it aliases (shared with the MLA
    # page registered first) ...
    assert worker._ple_region_index == 0
    assert worker.block_len_per_layer[0] == specs["mla"].page_size_bytes
    # ... but its descriptors must cover the whole PLE page, not the region's.
    bases = worker.kv_caches_base_addr[worker.engine_id][0]
    ple_descs = worker._build_mamba_local(bases)[-num_blocks:]
    assert ple_descs[:, 1].tolist() == [specs["ple"].page_size_bytes] * num_blocks
    assert ple_descs[:, 0].tolist() == [
        backing.data_ptr() + block * block_stride for block in range(num_blocks)
    ]
    assert worker._ple_block_len == specs["ple"].page_size_bytes
    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert metadata.ple_block_len == specs["ple"].page_size_bytes
