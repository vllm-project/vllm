import itertools as it

import pytest
import torch
import torch.nn.functional as F

from vllm.attention.layer import Attention
from vllm.attention.prefill_only.abstract import AttentionType
from vllm.attention.prefill_only.selector import (AttentionImpls, AttnBackend,
                                                  _Backend)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE


def compare_embeddings(embeddings1, embeddings2):
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


SEQ_LENS = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("attn_type", ["DECODER", "ENCODER"])
@pytest.mark.parametrize("n_seqs", list(range(1, len(SEQ_LENS))))
def test_basic_correctness(head_dim: int, num_heads: int, num_kv_heads: int,
                           attn_type: str, dtype: str, n_seqs: int):
    assert num_heads % num_kv_heads == 0

    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    attention_impls = AttentionImpls[dtype]

    seq_lens = SEQ_LENS[:n_seqs]
    batchsize = sum(seq_lens)

    query = torch.rand((batchsize, num_heads, head_dim),
                       dtype=torch_dtype,
                       device="cuda:0").view((batchsize, -1))
    key = torch.rand((batchsize, num_kv_heads, head_dim),
                     dtype=torch_dtype,
                     device="cuda:0").view((batchsize, -1))
    value = torch.rand((batchsize, num_kv_heads, head_dim),
                       dtype=torch_dtype,
                       device="cuda:0").view((batchsize, -1))

    impl_outputs_list = []

    for attention_impl in attention_impls:
        selected_backend = _Backend.backend_name_to_enum(attention_impl)
        backend_cls = AttnBackend.get_backend_cls(selected_backend)

        attn_type_enum = AttentionType.attn_type_name_to_enum(attn_type)

        attn_backend = backend_cls(attn_type_enum)
        scaling = head_dim**-0.5

        attn = Attention(num_heads,
                         head_dim,
                         scale=scaling,
                         num_kv_heads=num_kv_heads,
                         attn_backend=attn_backend)

        metadata_builder = attn_backend.make_metadata_builder()
        attn_metadata = metadata_builder(seq_lens=seq_lens)
        attn_metadata = attn_metadata.to("cuda:0")

        outputs = attn.forward(query,
                               key,
                               value,
                               kv_cache=None,
                               attn_metadata=attn_metadata)

        impl_outputs_list.append((attention_impl, outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all(
            (all_similarities <= 1.0 + tolerance)
            & (all_similarities >= 1.0 - tolerance)
        ), f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
