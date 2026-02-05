# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec


def test_mamba_get_block_table_tensor_align_maps_null_block_to_pad():
    # Block id 0 is reserved for BlockPool.null_block (placeholder) and must not
    # reach Mamba kernels. Mamba kernels use PAD_SLOT_ID (-1) for padding.
    spec = MambaSpec(block_size=4, shapes=((1,),), dtypes=(torch.float16,))

    # 2 requests, 4 blocks each.
    block_table = torch.tensor(
        [
            [5, 6, 7, 8],
            [1, 2, 0, 4],  # last block index will gather the `0`
        ],
        dtype=torch.int32,
    )
    # For block_size=4: start_indices = (seq_lens - 1) // 4 -> [0, 2]
    seq_lens = torch.tensor([1, 9], dtype=torch.int32)

    out = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    assert out.shape == (2, 1)
    assert int(out[0, 0]) == 5
    assert int(out[1, 0]) == PAD_SLOT_ID


def test_mamba_get_block_table_tensor_all_none_maps_null_block_to_pad():
    spec = MambaSpec(block_size=4, shapes=((1,),), dtypes=(torch.float16,))

    block_table = torch.tensor([[0, 2], [3, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([1, 1], dtype=torch.int32)

    out_all = mamba_get_block_table_tensor(block_table, seq_lens, spec, "all")
    assert out_all.tolist() == [[PAD_SLOT_ID, 2], [3, PAD_SLOT_ID]]

    out_none = mamba_get_block_table_tensor(block_table, seq_lens, spec, "none")
    assert out_none.tolist() == [[PAD_SLOT_ID, 2], [3, PAD_SLOT_ID]]

