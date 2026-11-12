# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest
import torch

from vllm import SamplingParams
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.model_executor.models.pixtral import (
    _apply_rotary_emb_vit_complex,
    _apply_rotary_emb_vit_real,
    _make_merge_indices,
    _make_packed_sequence_metadata,
    _merge_patches_by_index,
    _patch_grid_positions,
    get_sub_grids,
    pack_image_patches,
    precompute_freqs_cis_2d,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ....utils import VLLM_PATH, large_gpu_test
from ...utils import check_logprobs_close

if TYPE_CHECKING:
    from _typeshed import StrPath

PIXTRAL_ID = "mistralai/Pixtral-12B-2409"
MISTRAL_SMALL_3_1_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MINISTRAL_3B_ID = "mistralai/Ministral-3-3B-Instruct-2512"

MODELS = [PIXTRAL_ID, MISTRAL_SMALL_3_1_ID]

IMG_URLS = [
    "237-400x300.jpg",  # "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/237-400x300.jpg",
    "231-200x300.jpg",  # "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/237-400x300.jpg",
    "27-500x500.jpg",  # "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/237-400x300.jpg",
    "17-150x600.jpg",  # "https://huggingface.co/datasets/Isotr0py/mistral-test-images/resolve/main/237-400x300.jpg",
]
PROMPT = "Describe each image in one short sentence."


def _create_msg_format(urls: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT,
                }
            ]
            + [{"type": "image_url", "image_url": {"url": url}} for url in urls],
        }
    ]


SAMPLING_PARAMS = SamplingParams(max_tokens=512, temperature=0.0, logprobs=5)
LIMIT_MM_PER_PROMPT = dict(image=4)

MAX_MODEL_LEN = [8192, 65536]

FIXTURES_PATH = VLLM_PATH / "tests/models/fixtures"
assert FIXTURES_PATH.exists()

FIXTURE_LOGPROBS_CHAT = {
    PIXTRAL_ID: FIXTURES_PATH / "pixtral_chat.json",
    MISTRAL_SMALL_3_1_ID: FIXTURES_PATH / "mistral_small_3_chat.json",
    MINISTRAL_3B_ID: FIXTURES_PATH / "ministral_3b_chat.json",
}

OutputsLogprobs = list[tuple[list[int], str, SampleLogprobs | None]]


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TORCH_SDPA,
    ],
)
def test_packed_sequence_metadata(backend: AttentionBackendEnum) -> None:
    cu_seqlens, max_seqlen, sequence_lengths = _make_packed_sequence_metadata(
        [4, 6],
        backend,
        hidden_size=64,
        tp_size=1,
        device=torch.device("cpu"),
    )

    assert cu_seqlens.dtype == torch.int32
    assert max_seqlen.dtype == torch.int32
    if backend == AttentionBackendEnum.FLASHINFER:
        assert max_seqlen.item() >= 6
        assert sequence_lengths is not None
        assert sequence_lengths.dtype == torch.int32
        assert len(cu_seqlens) % 2 == 0
    else:
        expected_max_seqlen = 6 if backend == AttentionBackendEnum.FLASH_ATTN else 0
        assert max_seqlen.item() == expected_max_seqlen
        assert cu_seqlens.tolist() == [0, 4, 10]
        assert sequence_lengths is None


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
    ],
)
def test_packed_sequence_metadata_zero_length_padding(
    backend: AttentionBackendEnum,
) -> None:
    """Padding to a fixed sequence count keeps encoder CUDA graph buffer shapes
    independent of the number of images."""

    def make(lengths: list[int]):
        return _make_packed_sequence_metadata(
            lengths,
            backend,
            hidden_size=64,
            tp_size=1,
            device=torch.device("cpu"),
            num_sequences=4,
        )

    capture = make([64])
    replay = make([4, 6])
    assert replay[0].shape == capture[0].shape
    if backend == AttentionBackendEnum.FLASHINFER:
        assert replay[2] is not None and capture[2] is not None
        assert replay[2].shape == capture[2].shape
    else:
        assert replay[0].tolist() == [0, 4, 10, 10, 10]
        assert replay[1].item() == 6


def test_packed_sequence_metadata_fp8_layout() -> None:
    backend = AttentionBackendEnum.FLASHINFER
    bf16, _, _ = _make_packed_sequence_metadata(
        [4, 6], backend, hidden_size=64, tp_size=1, device=torch.device("cpu")
    )
    fp8, _, _ = _make_packed_sequence_metadata(
        [4, 6],
        backend,
        hidden_size=64,
        tp_size=1,
        device=torch.device("cpu"),
        fp8_padded_hidden_size=80,
    )
    # FP8 Q/K/V are separate contiguous tensors with padded strides.
    assert fp8.max().item() == 10 * 80
    assert bf16.max().item() == 10 * 64 * 3


def test_patch_grid_positions() -> None:
    positions = _patch_grid_positions([(2, 3), (1, 2)], torch.device("cpu"))
    assert positions.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [0, 0],
        [0, 1],
    ]


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_merge_indices_match_unfold(spatial_merge_size: int) -> None:
    m = spatial_merge_size
    grid_sizes = [(2 * m, 3 * m), (m, m), (4 * m, 2 * m)]
    features = torch.randn(sum(h * w for h, w in grid_sizes), 8)

    expected = torch.cat(
        [
            grid.view(-1, grid.shape[-1]).t()
            for grid in get_sub_grids(features, grid_sizes, m)
        ]
    )
    merge_indices = _make_merge_indices(grid_sizes, m, torch.device("cpu"))
    torch.testing.assert_close(
        _merge_patches_by_index(features, merge_indices), expected
    )


def test_pack_image_patches_matches_conv() -> None:
    patch_size = 4
    images = [torch.randn(3, 8, 12), torch.randn(3, 4, 4)]
    weight = torch.randn(16, 3, patch_size, patch_size)

    patches, grid_sizes = pack_image_patches(images, (patch_size, patch_size))

    assert grid_sizes == [(2, 3), (1, 1)]
    expected = torch.cat(
        [
            torch.nn.functional.conv2d(image[None], weight, stride=patch_size)
            .flatten(2)[0]
            .t()
            for image in images
        ]
    )
    torch.testing.assert_close(patches @ weight.reshape(16, -1).t(), expected)


def test_real_rotary_emb_vit_matches_complex() -> None:
    """The real RoPE used under torch.compile must match the eager one."""
    num_patches, num_heads, head_dim = 12, 4, 64
    freqs_cis = precompute_freqs_cis_2d(head_dim, height=3, width=4, theta=1e4)
    freqs_cis = torch.view_as_real(freqs_cis.reshape(num_patches, head_dim // 2))
    q, k = torch.randn(2, 1, num_patches, num_heads, head_dim).unbind(0)

    torch.testing.assert_close(
        _apply_rotary_emb_vit_real(q, k, freqs_cis),
        _apply_rotary_emb_vit_complex(q, k, freqs_cis),
    )


# For the test author to store golden output in JSON
def _dump_outputs_w_logprobs(
    outputs: OutputsLogprobs,
    filename: "StrPath",
) -> None:
    json_data = [
        (
            tokens,
            text,
            [
                {k: asdict(v) for k, v in token_logprobs.items()}
                for token_logprobs in (logprobs or [])
            ],
        )
        for tokens, text, logprobs in outputs
    ]

    with open(filename, "w") as f:
        json.dump(json_data, f)


def load_outputs_w_logprobs(filename: "StrPath") -> OutputsLogprobs:
    with open(filename, "rb") as f:
        json_data = json.load(f)

    return [
        (
            tokens,
            text,
            [
                {int(k): Logprob(**v) for k, v in token_logprobs.items()}
                for token_logprobs in logprobs
            ],
        )
        for tokens, text, logprobs in json_data
    ]


@large_gpu_test(min_gb=80)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_model_len", MAX_MODEL_LEN)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_chat(
    vllm_runner, max_model_len: int, model: str, dtype: str, local_asset_server
) -> None:
    if (
        model == MISTRAL_SMALL_3_1_ID
        and max_model_len == 65536
        and current_platform.is_rocm()
    ):
        pytest.skip(
            "OOM on ROCm: 24B model with 65536 context length exceeds GPU memory"
        )

    EXPECTED_CHAT_LOGPROBS = load_outputs_w_logprobs(FIXTURE_LOGPROBS_CHAT[model])
    with vllm_runner(
        model,
        dtype=dtype,
        tokenizer_mode="mistral",
        load_format="mistral",
        config_format="mistral",
        max_model_len=max_model_len,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = []

        urls_all = [local_asset_server.url_for(u) for u in IMG_URLS]
        msgs = [
            _create_msg_format(urls_all[:1]),
            _create_msg_format(urls_all[:2]),
            _create_msg_format(urls_all),
        ]
        for msg in msgs:
            output = vllm_model.llm.chat(msg, sampling_params=SAMPLING_PARAMS)

            outputs.extend(output)

    logprobs = vllm_runner._final_steps_generate_w_logprobs(outputs)
    # Remove last `None` prompt_logprobs to compare with fixture
    for i in range(len(logprobs)):
        assert logprobs[i][-1] is None
        logprobs[i] = logprobs[i][:-1]
    check_logprobs_close(
        outputs_0_lst=EXPECTED_CHAT_LOGPROBS,
        outputs_1_lst=logprobs,
        name_0="h100_ref",
        name_1="output",
    )


@large_gpu_test(min_gb=16)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_chat_consolidated(vllm_runner, dtype: str, local_asset_server) -> None:
    EXPECTED_CHAT_LOGPROBS = load_outputs_w_logprobs(
        FIXTURE_LOGPROBS_CHAT[MINISTRAL_3B_ID]
    )
    with vllm_runner(
        MINISTRAL_3B_ID,
        dtype=dtype,
        tokenizer_mode="mistral",
        load_format="mistral",
        config_format="mistral",
        max_model_len=8192,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = []
        urls_all = [local_asset_server.url_for(u) for u in IMG_URLS]
        msgs = [
            _create_msg_format(urls_all[:1]),
            _create_msg_format(urls_all[:2]),
            _create_msg_format(urls_all),
        ]
        for msg in msgs:
            output = vllm_model.llm.chat(msg, sampling_params=SAMPLING_PARAMS)
            outputs.extend(output)

    logprobs = vllm_runner._final_steps_generate_w_logprobs(outputs)
    for i in range(len(logprobs)):
        assert logprobs[i][-1] is None
        logprobs[i] = logprobs[i][:-1]
    check_logprobs_close(
        outputs_0_lst=EXPECTED_CHAT_LOGPROBS,
        outputs_1_lst=logprobs,
        name_0="h100_ref",
        name_1="output",
    )
