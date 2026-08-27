# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest
import torch
from mistral_common.multimodal import download_image
from mistral_common.protocol.instruct.chunk import ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.multimodal import image_from_chunk
from transformers import AutoProcessor

from vllm import SamplingParams, TextPrompt, TokensPrompt
from vllm.inputs import MultiModalDataBuiltins
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.model_executor.models.pixtral import (
    PatchMerger,
    PixtralForConditionalGeneration,
    _flatten_pixtral_image_patches,
    _make_packed_sequence_metadata,
    _pad_pixtral_cu_seqlens,
    _pad_pixtral_sequence_lengths,
    get_sub_grids,
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


def _create_msg_format_hf(urls: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "content": PROMPT,
                },
                *({"type": "image", "image": download_image(url)} for url in urls),
            ],
        }
    ]


def _create_engine_inputs(urls: list[str]) -> TokensPrompt:
    msg = _create_msg_format(urls)

    tokenizer = MistralTokenizer.from_model("pixtral")

    request = ChatCompletionRequest(messages=msg)  # type: ignore[type-var]
    tokenized = tokenizer.encode_chat_completion(request)

    engine_inputs = TokensPrompt(prompt_token_ids=tokenized.tokens)

    images = []
    for chunk in request.messages[0].content:
        if isinstance(chunk, ImageURLChunk):
            images.append(image_from_chunk(chunk))

    mm_data = MultiModalDataBuiltins(image=images)
    engine_inputs["multi_modal_data"] = mm_data

    return engine_inputs


def _create_engine_inputs_hf(urls: list[str]) -> TextPrompt:
    msg = _create_msg_format_hf(urls)

    tokenizer = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
    prompt = tokenizer.apply_chat_template(msg)

    images = []
    for chunk in msg[0]["content"]:
        if chunk["type"] == "image":
            images.append(chunk["image"])

    mm_data = MultiModalDataBuiltins(image=images)
    engine_inputs = TextPrompt(prompt=prompt, multi_modal_data=mm_data)

    return engine_inputs


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


def test_pixtral_encoder_cudagraph_patch_layout() -> None:
    patch_size = 4
    images = [torch.randn(3, 8, 12), torch.randn(3, 4, 8)]
    patch_conv = torch.nn.Conv2d(3, 5, patch_size, stride=patch_size, bias=False)

    expected = torch.cat(
        [
            patch_conv(image.unsqueeze(0)).flatten(2).permute(0, 2, 1)
            for image in images
        ],
        dim=1,
    ).squeeze(0)
    patches = _flatten_pixtral_image_patches(images, patch_size)
    actual = patch_conv(patches).flatten(1)

    torch.testing.assert_close(actual, expected)


def test_pixtral_encoder_cudagraph_patch_merge_layout() -> None:
    grid_sizes = [(2, 4), (4, 2)]
    hidden_size = 3
    patch_merger = PatchMerger.__new__(PatchMerger)
    torch.nn.Module.__init__(patch_merger)
    patch_merger.spatial_merge_size = 2
    features = torch.arange(
        sum(height * width for height, width in grid_sizes) * hidden_size,
        dtype=torch.float32,
    ).view(-1, hidden_size)

    indices = patch_merger.make_merge_indices(grid_sizes, torch.device("cpu"))
    actual = features[indices].permute(0, 2, 1).reshape(indices.shape[0], -1)
    expected = torch.cat(
        [
            grid.view(-1, grid.shape[-1]).t()
            for grid in get_sub_grids(features, grid_sizes, spatial_merge_size=2)
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_pixtral_supports_encoder_cudagraph() -> None:
    assert supports_encoder_cudagraph(PixtralForConditionalGeneration)


def test_pixtral_encoder_cudagraph_pads_attention_tail() -> None:
    src_cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    dst_cu_seqlens = torch.empty(6, dtype=torch.int32)
    _pad_pixtral_cu_seqlens(
        dst_cu_seqlens,
        src_cu_seqlens,
        input_capacity=8,
        attn_backend=AttentionBackendEnum.FLASH_ATTN,
        flashinfer_offset_scale=1,
    )

    assert dst_cu_seqlens.tolist() == [0, 2, 5, 5, 5, 8]

    src_sequence_lengths = torch.tensor([2, 3, 0, 0], dtype=torch.int32)
    dst_sequence_lengths = torch.empty(8, dtype=torch.int32)
    _pad_pixtral_sequence_lengths(
        dst_sequence_lengths, src_sequence_lengths, input_capacity=8
    )

    assert dst_sequence_lengths.tolist() == [2, 3, 0, 0, 0, 0, 0, 3]


def test_pixtral_encoder_cudagraph_pads_flashinfer_offsets() -> None:
    src_qko = torch.tensor([0, 8] + [20] * 7, dtype=torch.int32)
    src_v = torch.tensor([0, 24] + [60] * 7, dtype=torch.int32)
    src_cu_seqlens = torch.cat((src_qko, src_v))
    dst_cu_seqlens = torch.empty_like(src_cu_seqlens)

    _pad_pixtral_cu_seqlens(
        dst_cu_seqlens,
        src_cu_seqlens,
        input_capacity=8,
        attn_backend=AttentionBackendEnum.FLASHINFER,
        flashinfer_offset_scale=4,
    )

    assert dst_cu_seqlens[:9].tolist() == [0, 8] + [20] * 6 + [32]
    assert dst_cu_seqlens[9:].tolist() == [0, 24] + [60] * 6 + [96]


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
def test_chat_consolidated(
    vllm_runner, dtype: str, local_asset_server, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
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
        compilation_config={
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": [4096],
            "encoder_cudagraph_max_vision_items_per_batch": 4,
        },
    ) as vllm_model:
        engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
        model_runner = engine_core.model_executor.driver_worker.worker.model_runner
        encoder_cudagraph_manager = model_runner.encoder_cudagraph_manager
        assert encoder_cudagraph_manager is not None

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

        assert encoder_cudagraph_manager.graph_hits > 0

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
