# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields
from unittest.mock import Mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAAttentionBackend,
    KimiK3KDAMetadata,
    KimiK3KDAMetadataBuilder,
    _mamba_get_block_table_tensor,
    stage_spec_decode_metadata,
)
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
PRUNED_METADATA_FIELDS = {
    "chunk_indices",
    "chunk_offsets",
    "prefill_query_start_loc",
    "prefill_state_indices",
    "prefill_has_initial_state",
    "spec_sequence_masks",
}


def _assert_matches_shared_gdn(
    reference: GDNAttentionMetadata, actual: KimiK3KDAMetadata
):
    assert actual.recoverssm_commit is None
    assert actual.recoverssm_context is None
    for field in fields(GDNAttentionMetadata):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(reference, field.name)
        if field.name in PRUNED_METADATA_FIELDS:
            assert actual_value is None
            continue
        if (
            field.name in {"spec_token_indx", "non_spec_token_indx"}
            and actual.num_spec_decodes > 0
            and actual.num_prefills == 0
            and actual.num_decodes == 0
        ):
            assert actual_value is None
            continue
        if isinstance(actual_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value)
        elif field.name == "nums_dict":
            assert (actual_value is None) == (expected_value is None)
            if actual_value is not None:
                assert actual_value[8]["tot"] == expected_value[8]["tot"]
                torch.testing.assert_close(
                    actual_value[8]["nums"], expected_value[8]["nums"]
                )
        else:
            assert actual_value == expected_value


def _make_builder(
    builder_cls: type[AttentionMetadataBuilder],
    num_speculative_tokens: int,
    full_cuda_graph: bool,
    device: torch.device = DEVICE,
    mamba_cache_mode: str = "none",
    use_recoverssm: bool = False,
) -> AttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if num_speculative_tokens:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if full_cuda_graph else CUDAGraphMode.NONE
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.cache_config.use_replayssm = use_recoverssm
    vllm_config.cache_config.use_kda_recoverssm = use_recoverssm
    builder = builder_cls(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=(0 if use_recoverssm else num_speculative_tokens),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )
    if use_recoverssm:
        assert isinstance(builder, KimiK3KDAMetadataBuilder)
        builder.recoverssm_context = Mock()
    return builder


@pytest.mark.parametrize(
    (
        "batch",
        "num_decode_draft_tokens",
        "num_speculative_tokens",
        "full_cuda_graph",
        "is_prefilling",
    ),
    [
        pytest.param(
            BatchSpec(seq_lens=[50, 30], query_lens=[3, 3]),
            [2, 2],
            2,
            False,
            [False, False],
            id="pure-spec-decode",
        ),
        pytest.param(
            BatchSpec(seq_lens=[100, 65, 20], query_lens=[50, 1, 3]),
            [-1, -1, 2],
            2,
            False,
            [True, False, False],
            id="mixed-prefill-and-spec-decode",
        ),
        pytest.param(
            BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]),
            None,
            0,
            False,
            [False, False],
            id="regular-decode",
        ),
        pytest.param(
            BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]),
            [0, 0],
            2,
            False,
            [False, False],
            id="no-scheduled-draft-tokens",
        ),
    ],
)
def test_kimi_k3_kda_metadata_matches_shared_gdn(
    batch: BatchSpec,
    num_decode_draft_tokens: list[int] | None,
    num_speculative_tokens: int,
    full_cuda_graph: bool,
    is_prefilling: list[bool],
):
    kwargs: dict[str, torch.Tensor] = {}
    if num_decode_draft_tokens is not None:
        kwargs = {
            "num_decode_draft_tokens_cpu": torch.tensor(
                num_decode_draft_tokens, dtype=torch.int32
            ),
            "num_accepted_tokens": torch.ones(
                batch.batch_size, dtype=torch.int32, device=DEVICE
            ),
        }

    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool))
    reference = _make_builder(
        GDNAttentionMetadataBuilder,
        num_speculative_tokens,
        full_cuda_graph,
    ).build(
        0,
        common_attn_metadata,
        **kwargs,
    )
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens,
        full_cuda_graph,
    ).build(0, common_attn_metadata, **kwargs)

    assert isinstance(actual, KimiK3KDAMetadata)
    _assert_matches_shared_gdn(reference, actual)


def test_mixed_regular_and_spec_decode_uses_packed_decode_metadata():
    batch = BatchSpec(seq_lens=[100, 65, 20], query_lens=[1, 1, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([False, False, False]))
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=False,
    ).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device=DEVICE),
    )

    # The K3 layer dispatches the non-spec subgroup to packed decode whenever
    # it contains no prefill request.
    assert actual.num_decodes == 2
    assert actual.num_decode_tokens == 2
    assert actual.num_prefills == 0
    assert actual.num_prefill_tokens == 0
    assert actual.has_initial_state is None
    assert actual.nums_dict is None
    assert actual.non_spec_query_start_loc is None
    torch.testing.assert_close(actual.non_spec_token_indx, torch.tensor([0, 1]))
    torch.testing.assert_close(actual.spec_token_indx, torch.tensor([2, 3, 4]))
    torch.testing.assert_close(
        actual.spec_query_start_loc,
        torch.tensor([0, 3], dtype=torch.int32),
    )


def test_mixed_regular_and_spec_decode_excludes_request_padding():
    batch = BatchSpec(seq_lens=[16, 65, 20], query_lens=[0, 1, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([False, False, False]))
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=False,
    ).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device=DEVICE),
    )

    assert actual.num_decodes == 1
    assert actual.non_spec_state_indices_tensor is not None
    assert actual.non_spec_state_indices_tensor.shape == (1,)
    torch.testing.assert_close(actual.non_spec_token_indx, torch.tensor([0]))
    torch.testing.assert_close(actual.spec_token_indx, torch.tensor([1, 2, 3]))


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_recoverssm_spec_uses_one_state_slot_and_current_window(
    mamba_cache_mode: str,
):
    if mamba_cache_mode == "align" and not torch.cuda.is_available():
        pytest.skip("align metadata construction requires CUDA")
    device = torch.device("cuda") if mamba_cache_mode == "align" else DEVICE
    batch = BatchSpec(seq_lens=[100, 65, 20], query_lens=[1, 1, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, device
    ).replace(is_prefilling=torch.tensor([True, True, False]))
    builder = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=False,
        device=device,
        mamba_cache_mode=mamba_cache_mode,
        use_recoverssm=True,
    )
    assert isinstance(builder, KimiK3KDAMetadataBuilder)
    context = builder.recoverssm_context
    assert context is not None
    actual = builder.build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([3, 2, 2], dtype=torch.int32, device=device),
    )

    assert actual.spec_state_indices_tensor is not None
    assert actual.spec_state_indices_tensor.shape == (1, 1)
    torch.testing.assert_close(
        actual.num_accepted_tokens,
        torch.ones(1, dtype=torch.int32, device=device),
    )
    commit_metadata = actual.recoverssm_commit
    assert commit_metadata is not None
    torch.testing.assert_close(
        commit_metadata.request_indices,
        torch.tensor([2], dtype=torch.int32, device=device),
    )
    assert actual.recoverssm_context is context
    num_accepted_tokens = torch.tensor([3, 2, 1], dtype=torch.int32, device=device)

    postprocess = actual.commit_recoverssm_state(num_accepted_tokens)

    if mamba_cache_mode == "none":
        assert commit_metadata.align is None
        assert postprocess is None
    else:
        assert isinstance(postprocess, RecoverSSMPostprocessMetadata)
        assert postprocess.num_spec_decodes == 1
        assert postprocess.request_indices is commit_metadata.request_indices
        assert postprocess.block_table is common_attn_metadata.block_table_tensor
        assert (
            postprocess.num_computed_tokens
            is common_attn_metadata.compute_num_computed_tokens()
        )
        assert postprocess.block_size == BLOCK_SIZE
    args = context.commit.call_args.args
    assert args[0] is num_accepted_tokens
    torch.testing.assert_close(args[1], commit_metadata.state_indices[:, 0])
    torch.testing.assert_close(args[2], commit_metadata.query_start_loc)


def test_recoverssm_distinguishes_draftless_decode_from_one_token_prefill():
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[1, 1])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([False, True]))
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=False,
        use_recoverssm=True,
    ).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=torch.full((2,), -1, dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )

    assert actual.num_spec_decodes == 1
    assert actual.num_decodes == 0
    assert actual.num_prefills == 1
    assert actual.spec_state_indices_tensor is not None
    assert actual.spec_state_indices_tensor.shape == (1, 1)
    torch.testing.assert_close(
        actual.spec_query_start_loc,
        torch.tensor([0, 1], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("seq_len", "expected_has_initial_state"),
    [
        pytest.param(1, False, id="first-token-prefill"),
        pytest.param(65, True, id="final-one-token-prefill-chunk"),
    ],
)
def test_mixed_one_token_prefill_and_spec_decode_uses_prefill_metadata(
    seq_len: int,
    expected_has_initial_state: bool,
):
    batch = BatchSpec(seq_lens=[seq_len, 20], query_lens=[1, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([True, False]))
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=False,
    ).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=torch.tensor([-1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=DEVICE),
    )

    assert actual.num_prefills == 1
    assert actual.num_prefill_tokens == 1
    assert actual.num_decodes == 0
    assert actual.num_decode_tokens == 0
    assert actual.has_initial_state is not None
    assert actual.has_initial_state.tolist() == [expected_has_initial_state]
    assert actual.non_spec_query_start_loc is not None
    torch.testing.assert_close(
        actual.non_spec_query_start_loc,
        torch.tensor([0, 1], dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_k3_kda_cudagraph_capture_matches_shared_gdn():
    device = torch.device("cuda")
    batch = BatchSpec(seq_lens=[50, 30], query_lens=[3, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, device
    ).replace(is_prefilling=torch.tensor([False, False]))
    reference = _make_builder(
        GDNAttentionMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=True,
        device=device,
    ).build_for_cudagraph_capture(common_attn_metadata)
    actual = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=True,
        device=device,
    ).build_for_cudagraph_capture(common_attn_metadata)

    assert isinstance(actual, KimiK3KDAMetadata)
    _assert_matches_shared_gdn(reference, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_recoverssm_spec_cudagraph_stages_one_checkpoint_per_request():
    device = torch.device("cuda")
    batch = BatchSpec(seq_lens=[50, 30], query_lens=[3, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, device
    ).replace(is_prefilling=torch.tensor([False, False]))
    builder = _make_builder(
        KimiK3KDAMetadataBuilder,
        num_speculative_tokens=2,
        full_cuda_graph=True,
        device=device,
        use_recoverssm=True,
    )
    assert isinstance(builder, KimiK3KDAMetadataBuilder)
    assert builder.spec_state_indices_tensor.shape == (
        builder.vllm_config.scheduler_config.max_num_seqs,
        1,
    )
    actual = builder.build_for_cudagraph_capture(common_attn_metadata)

    assert actual.spec_state_indices_tensor is not None
    assert actual.spec_state_indices_tensor.shape == (batch.batch_size, 1)
    assert actual.num_accepted_tokens is not None
    torch.testing.assert_close(
        actual.num_accepted_tokens,
        torch.ones(batch.batch_size, dtype=torch.int32, device=device),
    )
    assert actual.recoverssm_commit is not None
    assert actual.recoverssm_commit.request_indices is None


def test_kimi_k3_kda_backend_uses_private_metadata_builder():
    assert KimiK3KDAAttentionBackend.get_builder_cls() is KimiK3KDAMetadataBuilder
    assert KimiK3KDAAttentionBackend.is_ssm()
    assert issubclass(KimiK3KDAAttentionBackend, GDNAttentionBackend)
    assert issubclass(KimiK3KDAMetadata, GDNAttentionMetadata)
    assert issubclass(KimiK3KDAMetadataBuilder, GDNAttentionMetadataBuilder)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_spec_decode_metadata_matches_pytorch():
    device = torch.device("cuda")
    num_spec_decodes = 33
    batch_size = 65
    num_state_slots = 3
    state_indices = torch.arange(
        num_spec_decodes * 32,
        dtype=torch.int32,
        device=device,
    ).reshape(num_spec_decodes, 32)[:, :num_state_slots]
    query_start_loc = (
        torch.arange(num_spec_decodes + 1, dtype=torch.int32, device=device)
        * num_state_slots
    )
    num_accepted_tokens = (
        torch.arange(num_spec_decodes, dtype=torch.int32, device=device)
        % num_state_slots
        + 1
    )

    staged_state_indices = torch.empty(
        (batch_size, num_state_slots), dtype=torch.int32, device=device
    )
    staged_query_start_loc = torch.empty(
        batch_size + 1, dtype=torch.int32, device=device
    )
    staged_num_accepted_tokens = torch.empty(
        batch_size, dtype=torch.int32, device=device
    )
    stage_spec_decode_metadata(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        staged_state_indices,
        staged_query_start_loc,
        staged_num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
    )

    expected_state_indices = torch.full_like(staged_state_indices, NULL_BLOCK_ID)
    expected_state_indices[:num_spec_decodes] = state_indices
    expected_query_start_loc = torch.full(
        (batch_size + 1,),
        query_start_loc[-1],
        dtype=torch.int32,
        device=device,
    )
    expected_query_start_loc[: num_spec_decodes + 1] = query_start_loc
    expected_num_accepted_tokens = torch.ones(
        batch_size, dtype=torch.int32, device=device
    )
    expected_num_accepted_tokens[:num_spec_decodes] = num_accepted_tokens

    torch.testing.assert_close(staged_state_indices, expected_state_indices)
    torch.testing.assert_close(staged_query_start_loc, expected_query_start_loc)
    torch.testing.assert_close(staged_num_accepted_tokens, expected_num_accepted_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_aligned_block_table_matches_shared_gdn():
    device = torch.device("cuda")
    seq_lens = torch.tensor(
        [0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 513],
        dtype=torch.int32,
        device=device,
    ).repeat(6)[:65]
    block_table_storage = torch.arange(
        seq_lens.numel() * 128,
        dtype=torch.int32,
        device=device,
    ).reshape(seq_lens.numel(), 128)
    block_table = block_table_storage[:, ::2]
    kv_cache_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=2,
    )

    expected = mamba_get_block_table_tensor(
        block_table,
        seq_lens,
        kv_cache_spec,
        "align",
    )
    actual = _mamba_get_block_table_tensor(
        block_table,
        seq_lens,
        kv_cache_spec,
        "align",
    )

    torch.testing.assert_close(actual, expected)
