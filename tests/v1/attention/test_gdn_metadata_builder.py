# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    device: torch.device = DEVICE,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "query_lens,draft_counts",
    [([3, 3], [2, 2]), ([3, 0], [2, -1]), ([3, 1], [2, -1]), ([1, 1], [-1, -1])],
)
def test_shared_gdn_metadata_matches_independent_groups(
    device, full_cuda_graph, query_lens, draft_counts
):
    """Sharing preserves all fields, including each group's distinct state IDs."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    device = torch.device(device)
    builders = [_create_gdn_builder(2, full_cuda_graph, device) for _ in range(2)]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[80, 96], query_lens=query_lens), BLOCK_SIZE, device
    )
    tables = [common.block_table_tensor, common.block_table_tensor + 1000]
    # A new ModelSpecificAttnMetadata per step must refresh accepted counts.
    for accepted in ([1, 2], [3, 1]):
        model_metadata = MambaHybridAttnMetadata(
            is_prefilling=torch.zeros(2, dtype=torch.bool, device=device),
            num_accepted_tokens=torch.tensor(
                accepted, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_counts, dtype=torch.int32),
        )
        actual = _build_groups(builders, common, tables, model_metadata)
        for i, builder in enumerate(builders):
            expected = builder.build(
                0,
                replace(common, block_table_tensor=tables[i]),
                num_accepted_tokens=model_metadata.num_accepted_tokens,
                num_decode_draft_tokens_cpu=model_metadata.num_decode_draft_tokens_cpu,
            )
            for field in fields(expected):
                torch.testing.assert_close(
                    getattr(actual[str(i)], field.name),
                    getattr(expected, field.name),
                    rtol=0,
                    atol=0,
                )
        if draft_counts[0] >= 0 and query_lens[1] != 1:
            for name in (
                "spec_sequence_masks",
                "spec_query_start_loc",
                "spec_token_indx",
                "num_accepted_tokens",
            ):
                assert getattr(actual["0"], name) is getattr(actual["1"], name)
            assert (
                actual["0"].spec_state_indices_tensor
                is not actual["1"].spec_state_indices_tensor
            )
        else:
            assert not model_metadata.metadata_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_shared_gdn_capture_replay_refreshes_both_groups():
    """A captured consumer sees updated shared values and separate state IDs."""
    device = torch.device("cuda")
    builders = [_create_gdn_builder(2, True, device) for _ in range(2)]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[80, 96], query_lens=[3, 3]), BLOCK_SIZE, device
    )
    tables = [common.block_table_tensor, common.block_table_tensor + 1000]
    captured = _build_groups(
        builders,
        common,
        tables,
        MambaHybridAttnMetadata(is_prefilling=torch.zeros(2, device=device)),
        capture=True,
    )
    names = (
        "spec_sequence_masks",
        "spec_query_start_loc",
        "spec_token_indx",
        "num_accepted_tokens",
        "spec_state_indices_tensor",
    )
    sources = [getattr(captured[str(i)], name) for i in range(2) for name in names]
    outputs = [torch.empty_like(t) for t in sources]
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for output, source in zip(outputs, sources):
            output.copy_(source)

    for accepted in ([1, 2], [2, 1]):
        for table in tables:
            table.add_(7)
        replayed = _build_groups(
            builders,
            common,
            tables,
            MambaHybridAttnMetadata(
                is_prefilling=torch.zeros(2, device=device),
                num_accepted_tokens=torch.tensor(
                    accepted, dtype=torch.int32, device=device
                ),
                num_decode_draft_tokens_cpu=torch.tensor([2, 2], dtype=torch.int32),
            ),
        )
        graph.replay()
        for i, source in enumerate(sources):
            expected = getattr(replayed[str(i // len(names))], names[i % len(names)])
            assert source.data_ptr() == expected.data_ptr()
            torch.testing.assert_close(outputs[i], expected, rtol=0, atol=0)
        assert outputs[3].tolist() == accepted
        assert not torch.equal(outputs[4], outputs[9])


def _build_groups(builders, common, tables, model_metadata, capture=False):
    groups = [
        [SimpleNamespace(layer_names=[str(i)], get_metadata_builder=lambda _, b=b: b)]
        for i, b in enumerate(builders)
    ]
    return build_attn_metadata(
        attn_groups=groups,
        num_reqs=common.num_reqs,
        num_tokens=common.num_actual_tokens,
        query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        max_query_len=common.max_query_len,
        seq_lens=common.seq_lens,
        max_seq_len=common.max_seq_len,
        block_tables=tables,
        slot_mappings=torch.stack([common.slot_mapping] * len(builders)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        model_specific_attn_metadata=model_metadata,
        for_cudagraph_capture=capture,
    )
