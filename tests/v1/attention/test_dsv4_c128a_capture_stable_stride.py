# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression gate: the C128A decode topk row stride must be capture-stable.

The C128A builder lays decode rows out in one flat persistent buffer at
``active_topk_width`` ints per row, and the decode consumers (FlashMLA /
FlashInfer SM120 ``_forward_decode``) run inside FULL cudagraphs, which bake
the row stride they saw at capture time. Capture builds metadata with
``max_seq_len = max_model_len``. If the runtime build derives the stride from
the batch's ``max_seq_len`` instead, the builder writes rows at a narrower
stride than the captured kernels read: row 0 still lines up at offset 0, but
every later decode row is read from stale bytes -- in a mixed batch, prefill
row 0 is written at exactly the offset the captured kernels read as decode
row 1. Observed in production as one request per batch (never the first)
emitting NaN-logits/BOS bursts or multilingual token salad under concurrent
long-context load, while ``--enforce-eager`` stays clean.

Asserted by inspecting the assignment, in the spirit of the tiering gate: a
gate on a value the builder no longer computes cannot fail.
"""

import ast
import inspect


def _active_topk_width_assignments() -> list[ast.Assign]:
    from vllm.models.deepseek_v4 import sparse_mla

    tree = ast.parse(inspect.getsource(sparse_mla))
    build_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_c128a_metadata"
    )
    return [
        node
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "active_topk_width"
            for t in node.targets
        )
    ]


def test_c128a_decode_stride_is_batch_independent():
    assigns = _active_topk_width_assignments()
    assert assigns, (
        "_build_c128a_metadata no longer assigns active_topk_width; "
        "re-point this gate at wherever the C128A row stride is computed"
    )
    for node in assigns:
        for sub in ast.walk(node.value):
            attr = sub.attr if isinstance(sub, ast.Attribute) else None
            name = sub.id if isinstance(sub, ast.Name) else None
            assert (attr or name) != "max_seq_len", (
                "C128A row stride is derived from the batch's max_seq_len: "
                f"`{ast.unparse(node)}`. FULL-cudagraph decode kernels bake "
                "the capture-time stride (max_seq_len = max_model_len), so a "
                "batch-dependent stride desynchronizes the builder's layout "
                "from the captured readers for every decode row after row 0."
            )


def test_c128a_build_kernel_iterates_the_same_stride():
    """The build kernel's per-row iteration bound must be the row stride.

    ``build_c128a_topk_metadata`` writes rows at ``max_compressed_tokens``
    ints per row; passing anything other than ``active_topk_width`` would
    desynchronize producer and consumer inside a single step.
    """
    from vllm.models.deepseek_v4 import sparse_mla

    tree = ast.parse(inspect.getsource(sparse_mla))
    build_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_c128a_metadata"
    )
    kws = [
        kw
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "max_compressed_tokens"
    ]
    assert kws, "build_c128a_topk_metadata call lost its max_compressed_tokens kwarg"
    assert ast.unparse(kws[0].value) == "active_topk_width"


def test_c128a_decode_row_addresses_survive_batch_width_changes():
    """GPU reproduction of the corruption: decode row addresses must not move.

    FULL-cudagraph capture builds metadata with ``max_seq_len =
    max_model_len``, and the captured decode kernels bake the resulting row
    addresses. Every later build must therefore lay decode rows out at those
    same addresses, or the captured kernels read rows r >= 1 from bytes the
    builder no longer writes. On the unfixed tree this fails concretely: a
    build at batch ``max_seq_len=300`` puts row 1 at stride 128 while the
    capture-time build put it at stride 8192, so row 1's address moves by
    (8192 - 128) * 4 bytes and the replayed kernel reads stale memory.
    """
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from types import SimpleNamespace

    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder

    device = torch.device("cuda")
    max_tokens = 64
    width_cap = 8192

    b = DeepseekV4FlashMLAMetadataBuilder.__new__(DeepseekV4FlashMLAMetadataBuilder)
    b.reorder_batch_threshold = 1
    b.compress_ratio = 128
    b.c128a_max_compressed = width_cap
    b.kv_cache_spec = SimpleNamespace(block_size=256)
    b.c128a_topk_buffer = torch.full(
        (max_tokens, width_cap), -1, dtype=torch.int32, device=device
    )
    b.c128a_decode_lens_buffer = torch.zeros(
        max_tokens, dtype=torch.int32, device=device
    )

    def build(max_seq_len):
        # Two single-token decode requests -- the smallest batch that has a
        # row 1. The split helper's pure-decode path only needs the CPU
        # bookkeeping fields plus is_prefilling for the tiering gate.
        cm = SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            num_reqs=2,
            max_query_len=1,
            num_actual_tokens=2,
            max_seq_len=max_seq_len,
            is_prefilling=torch.tensor([False, False]),
            positions=torch.tensor([256, 299], dtype=torch.int64, device=device),
            block_table_tensor=torch.zeros((2, 8), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
        )
        req_id = torch.tensor([0, 1], dtype=torch.int32, device=device)
        fields = b._build_c128a_metadata(cm, req_id)
        return fields["c128a_global_decode_topk_indices"]

    # Capture-time build (max_seq_len = max_model_len), then a runtime build
    # for a small batch.
    wide = build(1_048_576)
    narrow = build(300)

    assert narrow.shape[-1] == wide.shape[-1], (
        f"decode topk row stride changed with the batch's max_seq_len "
        f"({wide.shape[-1]} at capture vs {narrow.shape[-1]} at runtime): "
        "FULL-cudagraph consumers baked the capture-time stride and will "
        "read every decode row after row 0 from stale bytes"
    )
    assert narrow[1].data_ptr() == wide[1].data_ptr(), (
        "decode row 1 moved between builds; captured kernels read it at the "
        "capture-time address"
    )
