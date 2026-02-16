# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from tests.v1.attention.utils import (
    create_standard_kv_cache_spec,
    create_vllm_config,
    try_backend_includes_kv_cache_update,
    try_get_attention_backend,
)
from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if not is_flash_attn_varlen_func_available():
    pytest.skip(
        "This test requires flash_attn_varlen_func, but it's not available.",
        allow_module_level=True,
    )

# --------------------------------------------------------------------------- #
#  KV cache layout adaptation
# --------------------------------------------------------------------------- #
# Two KV cache layouts exist across backends:
#
#   Flash layout: (2, num_blocks, block_size, num_kv_heads, head_size)
#     - dim 0 separates key (index 0) and value (index 1)
#     - Used by: FLASH_ATTN, TREE_ATTN, ROCM_AITER_FA, ROCM_ATTN
#
#   Block layout: (num_blocks, 2, block_size, num_kv_heads, head_size)
#     - dim 1 separates key (index 0) and value (index 1)
#     - Used by: TRITON_ATTN
#
# The test creates KV caches in flash layout (the canonical format used by
# tree attention). When a reference backend needs block layout we transpose
# dims 0 and 1.
#
# Note: ROCM_ATTN uses flash layout for storage but its forward path calls
# PagedAttention.split_kv_cache which reinterprets the raw memory as paged
# layout (num_blocks, num_kv_heads, head_size//x, block_size, x). This is
# a view-level incompatibility, not a transpose - see the TODO in
# _get_available_reference_backends for details.
#
# TODO: Replace this mapping with a `KV_CACHE_LAYOUT` class attribute on each
# AttentionImpl so the layout is self-documented by the backend itself, e.g.:
#     class TritonAttentionImpl(AttentionImpl):
#         KV_CACHE_LAYOUT = "block"
# --------------------------------------------------------------------------- #

_BLOCK_KV_LAYOUT_BACKENDS = frozenset(
    {
        AttentionBackendEnum.TRITON_ATTN,
    }
)

# Backends whose do_kv_cache_update requires engine-level state (e.g.
# ForwardContext) that is not available in this test harness, but whose
# KV cache is flash layout and can be written with reshape_and_cache_flash.
# When a backend is listed here, forward_attention() bypasses
# do_kv_cache_update and writes directly to the cache.
_NEEDS_DIRECT_CACHE_UPDATE = frozenset(
    {
        AttentionBackendEnum.ROCM_AITER_FA,
    }
)

# Backends with known test-harness incompatibilities - see the TODOs
# inside _get_available_reference_backends for details.
_INCOMPATIBLE_REFERENCE_BACKENDS = frozenset(
    {
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.ROCM_ATTN,
    }
)


def _adapt_kv_cache_for_backend(
    kv_cache: torch.Tensor,
    backend: AttentionBackendEnum,
) -> torch.Tensor:
    """Convert kv_cache from flash layout ``(2, num_blocks, ...)`` to block
    layout ``(num_blocks, 2, ...)`` if the backend requires it.  Returns the
    original tensor unchanged when no conversion is needed."""
    if backend in _BLOCK_KV_LAYOUT_BACKENDS:
        return kv_cache.transpose(0, 1).contiguous()
    return kv_cache


def _get_platform_default_backend() -> AttentionBackendEnum:
    """Ask the platform what backend it would auto-select at runtime."""
    from vllm.v1.attention.selector import AttentionSelectorConfig

    config = AttentionSelectorConfig(
        block_size=32,
        kv_cache_dtype="auto",
        use_mla=False,
        use_sparse=False,
        head_size=128,
        dtype=torch.bfloat16,
    )
    backend_path = current_platform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=config,
    )
    for backend in AttentionBackendEnum:
        try:
            if backend.get_path() == backend_path:
                return backend
        except ValueError:
            continue
    raise RuntimeError(
        f"Platform returned backend path '{backend_path}' "
        f"that doesn't match any AttentionBackendEnum member."
    )


def _get_available_reference_backends() -> list[AttentionBackendEnum]:
    """Collect all reference backends the current platform can run.

    On CUDA this is just FLASH_ATTN. On ROCm this includes the platform
    default plus every backend the hardware supports, so the test validates
    tree attention against all of them.
    """
    if current_platform.is_rocm():
        backends: list[AttentionBackendEnum] = []

        # 1. Whatever the platform would auto-select at runtime.
        default_backend = _get_platform_default_backend()
        if default_backend not in _INCOMPATIBLE_REFERENCE_BACKENDS:
            backends.append(default_backend)

        # 2. TRITON_ATTN - always available on ROCm.
        if AttentionBackendEnum.TRITON_ATTN not in backends:
            backends.append(AttentionBackendEnum.TRITON_ATTN)

        # TODO: Enable ROCM_ATTN. Its forward path uses
        # PagedAttention.split_kv_cache which reinterprets the raw
        # cache memory as paged layout:
        #   key:   (num_blocks, num_kv_heads, head_size//x, block_size, x)
        #   value: (num_blocks, num_kv_heads, head_size, block_size)
        # Tree attention writes prefix data in NHD flash layout, so the
        # same bytes produce completely different values when read in
        # paged format. Supporting ROCM_ATTN would require writing
        # prefix data via PagedAttention.write_to_paged_cache into a
        # separate paged-format KV cache.

        # TODO: Enable ROCM_AITER_FA. Its metadata builder reads head
        # counts from the model config at construction time and
        # allocates extend_workspace with those dimensions. The test
        # uses independent head count parameters (num_heads=2/4,
        # num_kv_heads=2) that don't match the model config
        # (Llama-3-8B: 32 q heads, 8 kv heads), causing a head count
        # mismatch in flash_attn_varlen_func during extend_forward.
        # Fixing this requires either matching test head counts to the
        # model config or decoupling the builder from model config
        # head geometry. The direct cache update path
        # (_NEEDS_DIRECT_CACHE_UPDATE) is already in place for when
        # this is resolved.

        return backends

    # CUDA: flash attention.
    return [AttentionBackendEnum.FLASH_ATTN]


class MockAttentionLayer(torch.nn.Module):
    _q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    _k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    _v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    layer_name = "mock_layer"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def forward_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seqlen_k: int,
    backend: AttentionBackendEnum,
    spec_token_tree: str | None = None,
    num_spec_tokens: int = 0,
) -> torch.Tensor:
    """Run a single attention forward pass through the given backend.

    ``kv_cache`` is expected in **flash layout**
    ``(2, num_blocks, block_size, num_kv_heads, head_size)``.
    It is automatically converted when the target backend needs a
    different layout.
    """
    batch_size, q_len, num_heads, dim_per_head = q.shape
    num_kv_heads = k.shape[-2]
    # Initialize the query and KV sequence lengths.
    query_start_loc = q_len * torch.arange(
        batch_size + 1, device=q.device, dtype=torch.int32
    )
    query_lens = torch.diff(query_start_loc)
    seq_lens = torch.full(
        (batch_size,),
        seqlen_k,
        device=q.device,
        dtype=torch.int32,
    )
    context_lens = seq_lens - query_lens
    max_seq_len = int(seq_lens.max())
    max_query_len = q_len
    num_actual_tokens = query_start_loc[-1]

    softmax_scale = q.shape[-1] ** (-0.5)
    layer = MockAttentionLayer()

    # Build common metadata.
    model_name = "meta-llama/Meta-Llama-3-8B"
    builder_cls, impl_cls = try_get_attention_backend(backend)
    vllm_config = create_vllm_config(model_name=model_name, max_model_len=max(seq_lens))
    if spec_token_tree is not None:
        # Create speculative config if token tree is specified.
        vllm_config.speculative_config = SpeculativeConfig(
            target_model_config=vllm_config.model_config,
            target_parallel_config=ParallelConfig(),
            model=model_name,
            method="eagle",
            num_speculative_tokens=num_spec_tokens,
            speculative_token_tree=spec_token_tree,
        )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = builder_cls(kv_cache_spec, [], vllm_config, q.device)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=context_lens.cpu(),
        num_reqs=batch_size,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )

    # Build attention metadata.
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Initialize the backend implementation.
    instance = impl_cls(
        num_heads=num_heads,
        head_size=dim_per_head,
        scale=softmax_scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Adapt KV cache layout for this backend.
    adapted_kv_cache = _adapt_kv_cache_for_backend(kv_cache, backend)

    # Run forward pass and return output.
    query = q.view(-1, num_heads, dim_per_head)
    key = k.view(-1, num_kv_heads, dim_per_head)
    value = v.view(-1, num_kv_heads, dim_per_head)
    output = torch.empty_like(query)
    if not try_backend_includes_kv_cache_update(backend):
        if backend in _NEEDS_DIRECT_CACHE_UPDATE:
            # This backend's do_kv_cache_update requires engine-level
            # ForwardContext that isn't available in this test harness.
            # Write directly using reshape_and_cache_flash since the
            # KV cache layout is identical (flash layout, unbind on dim 0).
            key_cache, value_cache = adapted_kv_cache.unbind(0)
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                "auto",
                layer._k_scale,
                layer._v_scale,
            )
        else:
            instance.do_kv_cache_update(
                layer=layer,
                key=key,
                value=value,
                kv_cache=adapted_kv_cache,
                slot_mapping=attn_metadata.slot_mapping,
            )
    return instance.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=adapted_kv_cache.clone(),
        attn_metadata=attn_metadata,
        output=output,
    )


@pytest.mark.parametrize(
    "reference_backend",
    _get_available_reference_backends(),
    ids=lambda b: b.name,
)
def test_tree_attn_correctness(
    reference_backend: AttentionBackendEnum,
) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda"
    tree_attn_masks = {
        # Chain.
        "[(0,), (0, 0), (0, 0, 0)]": torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            device=device,
            dtype=torch.int32,
        ),
        # Tree.
        "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]": torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.int32,
        ),
    }

    dim_per_head = 128
    num_kv_heads = 2
    block_size = 32
    max_sequence_length = 8192
    randomize_blocks = True
    for batch_size in [1, 16, 32]:
        for num_heads in [2, 4]:
            for sequence_position in [16, 1024, 2048]:
                for spec_token_tree, tree_attn_mask in tree_attn_masks.items():
                    # Assert that the number of heads is divisible
                    # by the number of KV heads.
                    assert num_heads % num_kv_heads == 0

                    # Initialize q, k, and v.
                    tree_size_q = tree_attn_mask.shape[0]
                    seqlen_k = sequence_position + tree_size_q
                    q = torch.randn(
                        (batch_size, tree_size_q, num_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    k = torch.randn(
                        (batch_size, tree_size_q, num_kv_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    v = torch.randn(
                        (batch_size, tree_size_q, num_kv_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )

                    # KV cache in flash layout - the canonical format for
                    # tree attention. forward_attention() handles conversion
                    # when needed.
                    assert max_sequence_length % block_size == 0
                    max_blocks_per_batch = max_sequence_length // block_size
                    kv_cache = torch.randn(
                        (
                            2,
                            batch_size * max_blocks_per_batch,
                            block_size,
                            num_kv_heads,
                            dim_per_head,
                        ),
                        device=q.device,
                        dtype=torch.bfloat16,
                    )
                    num_alloc_blocks_per_batch = math.ceil(seqlen_k / block_size)
                    block_table = torch.zeros(
                        (batch_size, max_blocks_per_batch),
                        device=q.device,
                        dtype=torch.int32,
                    )
                    block_ids = torch.arange(
                        0,
                        batch_size * num_alloc_blocks_per_batch,
                        device=q.device,
                        dtype=torch.int32,
                    )
                    if randomize_blocks:
                        # Randomize the block ids.
                        block_ids = block_ids[torch.randperm(block_ids.numel())]
                    block_table[:, :num_alloc_blocks_per_batch] = block_ids.view(
                        -1, num_alloc_blocks_per_batch
                    )

                    # Set up the slot mapping for the input KVs.
                    tree_positions = sequence_position + torch.arange(
                        0,
                        tree_size_q,
                        device=q.device,
                        dtype=torch.int64,
                    ).repeat(batch_size, 1)
                    tree_slot_mapping = _gen_slot_mapping(
                        tree_positions, block_table, block_size
                    )

                    # Compute attention for the tree.
                    tree_attn_output = forward_attention(
                        q=q,
                        k=k,
                        v=v,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        slot_mapping=tree_slot_mapping,
                        seqlen_k=seqlen_k,
                        backend=AttentionBackendEnum.TREE_ATTN,
                        spec_token_tree=spec_token_tree,
                        num_spec_tokens=tree_size_q - 1,
                    ).view(batch_size, -1, num_heads, dim_per_head)

                    # Verify each branch against the reference backend.
                    for q_index in range(tree_size_q):
                        # Get the q, k, and v for the branch.
                        branch_mask = tree_attn_mask[q_index, :]
                        branch_indices = torch.nonzero(branch_mask, as_tuple=True)[0]
                        q_len = branch_indices.shape[0]
                        q_branch = q[:, branch_indices]
                        k_branch = k[:, branch_indices]
                        v_branch = v[:, branch_indices]

                        # Setup slot mapping for the branch.
                        branch_positions = sequence_position + torch.arange(
                            0,
                            q_len,
                            device=q.device,
                            dtype=torch.int64,
                        ).repeat(batch_size, 1)
                        branch_slot_mapping = _gen_slot_mapping(
                            branch_positions, block_table, block_size
                        )

                        # Reference attention for this branch.
                        ref_output = forward_attention(
                            q=q_branch,
                            k=k_branch,
                            v=v_branch,
                            kv_cache=kv_cache,
                            block_table=block_table,
                            slot_mapping=branch_slot_mapping,
                            seqlen_k=sequence_position + q_len,
                            backend=reference_backend,
                        ).view(batch_size, -1, num_heads, dim_per_head)

                        # Compare the outputs.
                        assert torch.allclose(
                            tree_attn_output[:, branch_indices],
                            ref_output,
                            atol=7.81e-3,
                        ), (
                            f"outputs are not close for "
                            f"reference_backend: {reference_backend.name}, "
                            f"batch_size: {batch_size}, "
                            f"num_heads: {num_heads}, "
                            f"sequence_position: {sequence_position}, "
                            f"tree_attn_mask: {tree_attn_mask}, "
                            f"q_index: {q_index}."
                        )


def _gen_slot_mapping(
    positions: torch.Tensor, block_table: torch.Tensor, block_size: int
):
    block_indices = positions // block_size
    blocks = block_table.gather(dim=1, index=block_indices)
    return (blocks * block_size + positions % block_size).view(-1)
