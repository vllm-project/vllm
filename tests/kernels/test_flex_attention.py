# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for FlexAttention backend vs default backend."""

import pytest
import torch

from tests.utils import set_random_seed
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    registered_attention_layers,
)
from vllm.config import AttentionConfig
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.flex_attention import (
    BlockSparsityHint,
    FlexAttentionMetadataBuilder,
    physical_to_logical_mapping,
)

from ..models.utils import check_embeddings_close, check_logprobs_close


@pytest.mark.parametrize(
    ("uses_paged_kv", "expected"),
    [
        (True, (16, 16)),
        (False, (128, 128)),
    ],
)
def test_flex_attention_default_block_sizes(
    uses_paged_kv: bool,
    expected: tuple[int, int],
):
    block_sizes = FlexAttentionMetadataBuilder._get_block_sizes(
        AttentionConfig(),
        cache_block_size=16,
        uses_paged_kv=uses_paged_kv,
    )
    assert block_sizes == expected


def test_flex_attention_explicit_block_sizes_override_encoder_defaults():
    block_sizes = FlexAttentionMetadataBuilder._get_block_sizes(
        AttentionConfig(
            flex_attn_q_block_size=64,
            flex_attn_kv_block_size=32,
        ),
        cache_block_size=16,
        uses_paged_kv=False,
    )
    assert block_sizes == (64, 32)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_flex_attention_full_cudagraphs(vllm_runner):
    """Test the numerics for flex attention full cudagraphs support."""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    num_logprobs = 5
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention eager
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        output_eager = llm_flex.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    # Run with flex attention compiled
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_default:
        output_compile = llm_default.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=output_eager,
        outputs_1_lst=output_compile,
        name_0="eager",
        name_1="compile",
    )


def windowed_causal_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx <= q_idx) & (q_idx - kv_idx < 4)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_flex_attention_custom_mask_full_cudagraphs(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(
        Attention,
        "logical_mask_mod",
        staticmethod(windowed_causal_mask_mod),
        raising=False,
    )

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    num_logprobs = 5
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_eager:
        output_eager = llm_eager.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [4],
        },
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_cudagraph:
        output_cudagraph = llm_cudagraph.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=output_eager,
        outputs_1_lst=output_cudagraph,
        name_0="eager",
        name_1="cudagraph",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend, ensuring they are similar when using the same seed.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    num_logprobs = 5
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        output_flex = llm_flex.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    # Run with default backend
    set_random_seed(seed)
    with vllm_runner(
        model_name,
        runner="generate",
        tensor_parallel_size=1,
        num_gpu_blocks_override=128,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    ) as llm_default:
        output_default = llm_default.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=output_flex,
        outputs_1_lst=output_default,
        name_0="flex",
        name_1="default",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_encoder_flex_attention_vs_default_backend(vllm_runner):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend for encoder models.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    # Exercise packed sequence boundaries inside 128-token FlexAttention
    # blocks, including sequences that span more than one block.
    prompts = [
        "hello " * 120,
        "world " * 130,
        "attention " * 254,
    ]

    # Run with flex attention
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=384,
        enforce_eager=True,
        attention_config={"backend": "FLEX_ATTENTION"},
    ) as llm_flex:
        flex_outputs = llm_flex.embed(prompts)

    # Run with default backend
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=384,
        enforce_eager=True,
    ) as llm_default:
        default_outputs = llm_default.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=flex_outputs,
        embeddings_1_lst=default_outputs,
        name_0="flex",
        name_1="default",
        tol=1e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_block_mask_direct_vs_slow_path():
    """Test that direct path block mask is a superset of slow path.

    The direct path may include extra blocks for performance (over-estimation),
    but must include all blocks that the slow path determines are necessary.
    """
    device = torch.device("cuda")

    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen2.5-1.5B-Instruct", block_size=16, max_model_len=1024
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # Use a mixed batch that will create groups spanning multiple sequences
    batch_spec = BatchSpec(
        seq_lens=[35, 64, 128, 256], query_lens=[33, 5, 32, 64], name="test_mixed_batch"
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    with registered_attention_layers(vllm_config, ["layer"]):
        builder = FlexAttentionMetadataBuilder(
            kv_cache_spec, ["layer"], vllm_config, device
        )

    metadata_direct = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    builder.direct_build = False
    metadata_slow = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    assert metadata_direct.block_mask is not None
    assert metadata_slow.block_mask is not None

    # Extract block indices for comparison, B, H are the same
    direct_indices = metadata_direct.block_mask.kv_indices[0, 0]
    slow_indices = metadata_slow.block_mask.kv_indices[0, 0]
    direct_num = metadata_direct.block_mask.kv_num_blocks[0, 0]
    slow_num = metadata_slow.block_mask.kv_num_blocks[0, 0]

    # main test: every block needed by slow path must be in direct path
    num_groups = direct_num.shape[0]
    all_contained = True
    missing_details = []

    for group_idx in range(num_groups):
        direct_blocks = set(direct_indices[group_idx, : direct_num[group_idx]].tolist())
        slow_blocks = set(slow_indices[group_idx, : slow_num[group_idx]].tolist())

        missing_blocks = slow_blocks - direct_blocks
        if missing_blocks:
            all_contained = False
            missing_details.append(
                f"Group {group_idx}: missing {sorted(missing_blocks)}"
            )

    assert all_contained, (
        "Direct path is missing blocks required by slow path:\n"
        + "\n".join(missing_details)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
@pytest.mark.parametrize("direct_build", [True, False])
@torch.inference_mode()
def test_flex_attention_request_count_changes_reuse_compiled_graph(direct_build):
    """Request-count changes preserve attention results and compiled graph reuse."""
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch.nn.attention.flex_attention import flex_attention

    from vllm.v1.attention.backends.flex_attention import get_kernel_options

    torch.manual_seed(42)
    device = torch.device("cuda")
    config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B",
        max_model_len=128,
        num_gpu_blocks=40,
        max_num_seqs=8,
        max_num_batched_tokens=128,
    )
    with registered_attention_layers(config, ["layer"]):
        builder = FlexAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config), ["layer"], config, device
        )
    builder.direct_build = direct_build
    query = torch.randn(1, 2, 32, 64, device=device)
    key = torch.randn(1, 2, 640, 64, device=device)
    value = torch.randn_like(key)
    kernel_options = get_kernel_options(query, 16, 16, direct_build)
    counter = CompileCounterWithBackend("inductor")
    compiled_attention = torch.compile(flex_attention, backend=counter, fullgraph=True)

    def build_metadata(num_reqs):
        # Keep Q/K tensor and block-mask sizes fixed to isolate request counts.
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[64] * num_reqs, query_lens=[32 // num_reqs] * num_reqs),
            16,
            device,
            max_block_idx=40,
        )
        common.block_table_tensor.copy_(
            torch.arange(1, num_reqs * 4 + 1, device=device).view(num_reqs, 4)
        )
        return builder.build(0, common)

    def attend(metadata):
        return compiled_attention(
            query,
            key,
            value,
            block_mask=metadata.block_mask,
            kernel_options=kernel_options,
        )

    def reference(num_reqs):
        query_len = 32 // num_reqs
        q_idx = torch.arange(32, device=device)[:, None]
        kv_idx = torch.arange(640, device=device)[None, :]
        request = q_idx // query_len
        logical_q = q_idx % query_len + 64 - query_len
        logical_kv = kv_idx - (request * 4 + 1) * 16
        mask = (logical_kv >= 0) & (logical_kv <= logical_q)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )

    for num_reqs in (4, 2, 1, 8, 1):
        metadata = build_metadata(num_reqs)
        torch.testing.assert_close(
            attend(metadata), reference(num_reqs), atol=1e-4, rtol=1e-4
        )
    assert counter.frame_count == 1

    if direct_build:
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = attend(metadata)
        for num_reqs in (8, 2, 1):
            metadata = build_metadata(num_reqs)
            graph.replay()
            torch.testing.assert_close(
                output, reference(num_reqs), atol=1e-4, rtol=1e-4
            )


def test_physical_to_logical_mapping_handles_reused_blocks():
    """Regression test: reused physical blocks map to the latest logical block.

    For sliding-window / hybrid attention layers, physical KV-cache blocks can be
    reused over time. The inverse mapping must therefore select the latest
    logical block index for a physical block id.
    """
    # Padding should not make physical block 0 look live.
    block_table = torch.tensor([[6, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([1 * 16], dtype=torch.int32)  # only 1 block valid
    out = physical_to_logical_mapping(
        block_table=block_table, seq_lens=seq_lens, block_size=16, total_blocks=10
    )
    assert out[0, 0].item() == -1
    assert out[0, 6].item() == 0

    # If a physical block id appears multiple times (block reuse), mapping should
    # point to the latest logical block index.
    block_table2 = torch.tensor([[2, 2, 5]], dtype=torch.int32)
    seq_lens2 = torch.tensor([3 * 16], dtype=torch.int32)
    out2 = physical_to_logical_mapping(
        block_table=block_table2, seq_lens=seq_lens2, block_size=16, total_blocks=8
    )
    assert out2[0, 2].item() == 1


def test_page_ownership_tables():
    """Per-page tables encode exactly the ownership in `physical_to_logical`.

    Page 1 is shared by requests 0 and 65 but not 1, so a `req % 64` bit
    alone would wrongly give it to request 1. Request 63 uses the sign bit.
    """
    from vllm.v1.attention.backends.flex_attention import page_ownership_tables

    num_reqs, num_words = 70, 3
    block_table = torch.arange(1, 2 * num_reqs + 1).view(num_reqs, 2)
    block_table[65, 0] = 1
    seq_lens = torch.full((num_reqs,), 32)
    p2l = physical_to_logical_mapping(
        block_table, seq_lens, block_size=16, total_blocks=2 * num_reqs + 1
    )

    page_to_logical, page_owner_bits = page_ownership_tables(p2l, num_words)

    assert page_owner_bits.shape == (num_words, p2l.shape[1])
    assert torch.equal(page_to_logical, p2l.amax(dim=0))
    reqs = torch.arange(num_reqs)[:, None]
    owner_bit = (
        page_owner_bits[reqs // 64, torch.arange(p2l.shape[1])] >> (reqs % 64)
    ) & 1
    assert torch.equal(owner_bit.bool(), p2l >= 0)
    assert not owner_bit[1, 1]
    assert page_owner_bits[2].eq(0).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < DIRECT_BUILD_VERSION,
    reason="CUDA not available or PyTorch version < 2.9",
)
@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(
            seq_lens=[200, 150, 64, 97, 33, 120, 81, 256],
            query_lens=[200, 150, 1, 97, 17, 1, 81, 64],
            name="mixed",
        ),
        BatchSpec(
            seq_lens=[(17, 40, 64)[i % 3] for i in range(70)],
            query_lens=[1, 1, 1] + [(17, 40, 64)[i % 3] for i in range(3, 70)],
            name="over_64_requests",
        ),
    ],
    ids=lambda spec: spec.name,
)
@torch.inference_mode()
def test_flex_attention_paged_mask_batch_invariant(monkeypatch, batch_spec):
    """Each request's output must not depend on the rest of the batch.

    Regression test for https://github.com/vllm-project/vllm/issues/66460:
    on gfx950 a request-indexed page lookup in the paged mask_mod changed
    the kernel's reduction order once a batch held several requests.
    """
    from vllm.v1.attention.backends.flex_attention import (
        flex_attention_compiled,
        get_kernel_options,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(0)
    device = torch.device("cuda")
    block_size, num_blocks = 16, 1024
    num_heads, num_kv_heads, head_size = 16, 8, 128
    config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B",
        max_model_len=512,
        num_gpu_blocks=num_blocks,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
    )
    with registered_attention_layers(config, ["layer"]):
        builder = FlexAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config), ["layer"], config, device
        )

    # Requests share leading blocks with an earlier one (prefix caching); the
    # rest are private and shuffled. From request 64 on, the source is 63
    # requests back, so e.g. request 65 shares request 2's prefix while
    # request 1 (bit 1 too, same q-block as request 2) must not see it.
    pages = (torch.randperm(num_blocks - 1) + 1).tolist()
    seq_lens = batch_spec.seq_lens
    max_blocks = cdiv(max(seq_lens), block_size)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    for r, seq_len in enumerate(seq_lens):
        shared = src = 0
        if r >= 3:
            src = r - 63 if r >= 64 else r - 1
            shared = min(seq_lens[src], seq_len - 1) // block_size // 2
        for j in range(cdiv(seq_len, block_size)):
            block_table[r, j] = block_table[src, j] if j < shared else pages.pop()

    kv_shape = (1, num_kv_heads, num_blocks * block_size, head_size)
    key = torch.randn(kv_shape, device=device, dtype=torch.bfloat16)
    value = torch.randn(kv_shape, device=device, dtype=torch.bfloat16)
    query = torch.randn(
        1,
        num_heads,
        sum(batch_spec.query_lens),
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )

    def attend(spec, rows, q):
        common = create_common_attn_metadata(spec, block_size, device)
        common.block_table_tensor.copy_(
            block_table[rows, : common.block_table_tensor.shape[1]]
        )
        metadata = builder.build(0, common)
        kernel_options = get_kernel_options(q, block_size, block_size, True)
        kernel_options["IS_DIVISIBLE"] = False
        return flex_attention_compiled(
            q,
            key,
            value,
            block_mask=metadata.block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )

    batched = attend(batch_spec, list(range(len(seq_lens))), query)
    start = 0
    for r, (seq_len, query_len) in enumerate(zip(seq_lens, batch_spec.query_lens)):
        q = query[:, :, start : start + query_len]
        alone = attend(BatchSpec(seq_lens=[seq_len], query_lens=[query_len]), [r], q)
        assert torch.equal(batched[:, :, start : start + query_len], alone), (
            f"request {r} differs when batched"
        )
        start += query_len


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_block_sparsity_hint_prunes_blocks():
    """Test that BlockSparsityHint prunes KV blocks from the direct build path.

    Uses a hint that only keeps the diagonal (q_block == kv_block) to verify
    that off-diagonal blocks are excluded from the resulting BlockMask.
    """
    device = torch.device("cuda")

    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m",
        block_size=16,
        max_model_len=1024,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    batch_spec = BatchSpec(
        seq_lens=[256],
        query_lens=[256],
        name="test_sparsity_hint",
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    with registered_attention_layers(vllm_config, ["layer"]):
        builder = FlexAttentionMetadataBuilder(
            kv_cache_spec, ["layer"], vllm_config, device
        )

    metadata_no_hint = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    metadata_no_hint.block_mask = metadata_no_hint._build_block_mask_direct()
    assert metadata_no_hint.block_mask.kv_num_blocks.max().item() > 1

    def diagonal_hint(q_block_idx, kv_block_idx, block_size):
        return q_block_idx == kv_block_idx

    metadata_with_hint = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    metadata_with_hint.block_sparsity_hint = BlockSparsityHint(
        hint_fn=diagonal_hint,
    )
    metadata_with_hint.block_mask = metadata_with_hint._build_block_mask_direct()
    assert metadata_with_hint.block_mask.kv_num_blocks.max().item() <= 1


if __name__ == "__main__":
    pytest.main([__file__])
