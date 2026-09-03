# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import asdict
from types import SimpleNamespace
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
from vllm.config.multimodal import MultiModalConfig
from vllm.inputs import MultiModalDataBuiltins
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.model_executor.models.pixtral import (
    PATCH_MERGE,
    PatchMerger,
    PixtralForConditionalGeneration,
    VisionEncoderArgs,
    VisionLanguageAdapter,
    VisionTransformer,
    _flatten_pixtral_image_patches,
    _make_packed_sequence_metadata,
    _pad_pixtral_cumulative_seqlens,
    _pad_pixtral_flashinfer_cu_seqlens,
    _pad_pixtral_sequence_lengths,
    get_sub_grids,
    position_meshgrid_from_sizes,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

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

_CUDA_GRAPH_TOKEN_BUDGET = 17
_CUDA_GRAPH_MAX_BATCH_SIZE = 4
_TINY_VISION_HIDDEN_SIZE = 128

_IS_BLACKWELL_OR_NEWER = (
    torch.cuda.is_available() and current_platform.has_device_capability(100)
)
_requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="Pixtral encoder CUDA graph coverage requires a CUDA GPU",
)

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


def test_position_meshgrid_from_sizes_matches_ij_meshgrid() -> None:
    grid_sizes = [(2, 3), (1, 4), (3, 1)]
    expected = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(height),
                    torch.arange(width),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for height, width in grid_sizes
        ]
    )
    actual = position_meshgrid_from_sizes(grid_sizes)
    torch.testing.assert_close(actual, expected)
    assert position_meshgrid_from_sizes([]).shape == (0, 2)


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
    _pad_pixtral_cumulative_seqlens(
        dst_cu_seqlens,
        src_cu_seqlens,
        input_capacity=8,
    )

    assert dst_cu_seqlens.tolist() == [0, 2, 5, 5, 5, 8]

    src_sequence_lengths = torch.tensor([2, 3, 0, 0], dtype=torch.int32)
    dst_sequence_lengths = torch.empty(8, dtype=torch.int32)
    _pad_pixtral_sequence_lengths(
        dst_sequence_lengths, src_sequence_lengths, input_capacity=8
    )

    assert dst_sequence_lengths.tolist() == [2, 3, 0, 0, 0, 0, 0, 3]

    full_src_cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    _pad_pixtral_cumulative_seqlens(
        dst_cu_seqlens,
        full_src_cu_seqlens,
        input_capacity=8,
    )
    assert dst_cu_seqlens.tolist() == [0, 3, 8, 8, 8, 8]

    full_src_sequence_lengths = torch.tensor([3, 5, 0, 0], dtype=torch.int32)
    _pad_pixtral_sequence_lengths(
        dst_sequence_lengths,
        full_src_sequence_lengths,
        input_capacity=8,
    )
    assert dst_sequence_lengths.tolist() == [3, 5, 0, 0, 0, 0, 0, 0]


def test_pixtral_encoder_cudagraph_pads_flashinfer_offsets() -> None:
    src_qko = torch.tensor([0, 8] + [20] * 7, dtype=torch.int32)
    src_v = torch.tensor([0, 24] + [60] * 7, dtype=torch.int32)
    src_cu_seqlens = torch.cat((src_qko, src_v))
    dst_cu_seqlens = torch.empty_like(src_cu_seqlens)

    _pad_pixtral_flashinfer_cu_seqlens(
        dst_cu_seqlens,
        src_cu_seqlens,
        input_capacity=8,
        flashinfer_offset_scale=4,
    )

    assert dst_cu_seqlens[:9].tolist() == [0, 8] + [20] * 6 + [32]
    assert dst_cu_seqlens[9:].tolist() == [0, 24] + [60] * 6 + [96]


def test_pixtral_encoder_cudagraph_pads_flashinfer_short_src() -> None:
    # Replay of 2 images pads to FlashInfer bucket 8 (9 offsets/section).
    # Capture with max_batch_size >= 8 pads to bucket 16 (17 offsets/section).
    src_qko = torch.tensor([0, 8] + [20] * 7, dtype=torch.int32)
    src_v = torch.tensor([0, 24] + [60] * 7, dtype=torch.int32)
    src_cu_seqlens = torch.cat((src_qko, src_v))
    dst_cu_seqlens = torch.empty(34, dtype=torch.int32)

    _pad_pixtral_flashinfer_cu_seqlens(
        dst_cu_seqlens,
        src_cu_seqlens,
        input_capacity=8,
        flashinfer_offset_scale=4,
    )

    assert dst_cu_seqlens[:17].tolist() == [0, 8] + [20] * 14 + [32]
    assert dst_cu_seqlens[17:].tolist() == [0, 24] + [60] * 14 + [96]


def _make_tiny_pixtral_encoder(
    backend: AttentionBackendEnum,
    merge_size: int,
    vllm_config,
) -> PixtralForConditionalGeneration:
    multimodal_config = MultiModalConfig(mm_encoder_attn_backend=backend)
    vllm_config.model_config = SimpleNamespace(
        multimodal_config=multimodal_config,
    )
    vision_args = VisionEncoderArgs(
        hidden_size=_TINY_VISION_HIDDEN_SIZE,
        num_channels=3,
        image_size=64,
        patch_size=4,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        rope_theta=10_000,
        image_token_id=10,
        adapter_bias=True,
        spatial_merge_size=merge_size,
        add_pre_mm_projector_layer_norm=merge_size > 1,
        mm_projector_id=PATCH_MERGE if merge_size > 1 else "",
    )

    model = PixtralForConditionalGeneration.__new__(PixtralForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=_TINY_VISION_HIDDEN_SIZE)
    )
    model.model_config = SimpleNamespace(max_model_len=8192)
    model.multimodal_config = multimodal_config
    model.vision_args = vision_args
    model._encoder_cudagraph_input_capacities = {}

    with set_default_torch_dtype(torch.bfloat16):
        model.vision_encoder = VisionTransformer(
            vision_args,
            prefix="vision_encoder",
        )
        model.pre_mm_projector_norm = (
            RMSNorm(vision_args.hidden_size, eps=1e-5) if merge_size > 1 else None
        )
        model.patch_merger = (
            PatchMerger(
                vision_encoder_dim=vision_args.hidden_size,
                spatial_merge_size=merge_size,
            )
            if merge_size > 1
            else None
        )
        model.vision_language_adapter = VisionLanguageAdapter(
            vision_args,
            dim=_TINY_VISION_HIDDEN_SIZE,
        )

    model.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.zero_()
            elif param.ndim == 1:
                param.fill_(1)
            else:
                torch.nn.init.normal_(param, mean=0, std=0.02)
    model.requires_grad_(False)
    model.eval()
    attention = model.vision_encoder.transformer.layers[0].attention.attn
    assert attention.attn_backend == backend
    return model


def _make_pixtral_encoder_cudagraph_manager(
    model: PixtralForConditionalGeneration,
    token_budgets: list[int] | None = None,
) -> EncoderCudaGraphManager:
    manager = object.__new__(EncoderCudaGraphManager)
    manager.token_budgets = sorted(token_budgets or [_CUDA_GRAPH_TOKEN_BUDGET])
    manager.path_token_budgets = {"default": manager.token_budgets}
    manager.max_batch_size = _CUDA_GRAPH_MAX_BATCH_SIZE
    manager.max_frames_per_batch = 0
    manager.use_dp = False
    manager.budget_graphs = {"default": {}}
    manager.graph_pool = None
    manager.graph_hits = 0
    manager.graph_misses = 0
    manager.log_stats_interval = 100
    manager.model = model
    manager.config = model.get_encoder_cudagraph_config()
    manager.device = torch.device("cuda")
    manager.dtype = torch.bfloat16
    return manager


def _make_pixtral_images(
    output_grid_sizes: list[tuple[int, int]],
    merge_size: int,
) -> list[torch.Tensor]:
    patch_size = 4
    return [
        torch.randn(
            3,
            height * merge_size * patch_size,
            width * merge_size * patch_size,
            device="cuda",
            dtype=torch.float32,
        )
        for height, width in output_grid_sizes
    ]


def _get_eager_pixtral_outputs(
    model: PixtralForConditionalGeneration,
    mm_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    output_sizes = [
        spec.output_tokens for spec in model.get_encoder_cudagraph_item_specs(mm_kwargs)
    ]
    with torch.inference_mode():
        output = model.encoder_eager_forward(mm_kwargs)
    return torch.split(output, output_sizes)


def _assert_pixtral_outputs_close(
    actual: list[torch.Tensor],
    expected: tuple[torch.Tensor, ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        assert actual_item.shape == expected_item.shape
        # Graph and eager run the same kernels on the same inputs, so they
        # should agree far more tightly than a generic bf16 tolerance.
        torch.testing.assert_close(
            actual_item,
            expected_item,
            atol=2e-3,
            rtol=2e-3,
        )


def _poison_cudagraph_padding(monkeypatch) -> dict[str, bool]:
    """Fill cudagraph input padding with NaN instead of zeros during replay.

    Swapping the manager's default padding logic lets ``manager.execute()`` run
    the real replay path with every float buffer's padded tail poisoned.
    ``cu_seqlens``/``sequence_lengths`` are deliberately untouched -- they have
    their own entries in ``padding_logics``, and it is those model-supplied
    paddings that must keep the poison away from real tokens.

    Integer buffers keep zero padding: ``merge_indices`` is used as
    ``image_features[merge_indices]``, so an out-of-range pad index would be an
    illegal memory access rather than a detectable NaN.

    Returns:
        Flags recording whether float and integer buffers were actually padded,
        so a caller can reject a batch that filled the budget exactly and made
        the poison a no-op.
    """
    padded = {"float": False, "int": False}

    def poisoning_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
        is_float = dst.is_floating_point() or dst.is_complex()
        if dst.shape[0] > src.shape[0]:
            padded["float" if is_float else "int"] = True
        if is_float:
            dst.fill_(torch.nan)
        else:
            dst.zero_()
        dst[: src.shape[0]].copy_(src)

    # _copy_padded_buffer is a staticmethod looked up via self, so the
    # replacement must be wrapped to avoid binding self as the first argument.
    monkeypatch.setattr(
        EncoderCudaGraphManager,
        "_copy_padded_buffer",
        staticmethod(poisoning_copy),
    )
    return padded


@_requires_cuda
@pytest.mark.parametrize(
    "backend",
    [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.FLASHINFER],
    ids=["flash-attn", "flashinfer"],
)
@pytest.mark.parametrize("merge_size", [1, 2], ids=["pixtral", "ministral"])
def test_pixtral_encoder_cudagraph_matches_eager(
    backend: AttentionBackendEnum,
    merge_size: int,
    default_vllm_config,
    dist_init,
) -> None:
    if backend == AttentionBackendEnum.FLASHINFER and not _IS_BLACKWELL_OR_NEWER:
        pytest.skip("Pixtral FlashInfer CUDA graph coverage requires SM100 or newer")

    torch.manual_seed(0)
    model = _make_tiny_pixtral_encoder(
        backend,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(model)

    expected_cu_seqlens_padding = (
        PixtralForConditionalGeneration._pad_encoder_cudagraph_flashinfer_cu_seqlens
        if backend == AttentionBackendEnum.FLASHINFER
        else PixtralForConditionalGeneration._pad_encoder_cudagraph_cumulative_seqlens
    )
    assert (
        manager.config.padding_logics["cu_seqlens"].__func__
        is expected_cu_seqlens_padding
    )

    manager.capture(graph_pool=current_platform.graph_pool_handle())

    graph = manager.budget_graphs["default"][_CUDA_GRAPH_TOKEN_BUDGET]
    output_capacity = 20
    assert graph.output_buffer.shape == (
        output_capacity,
        _TINY_VISION_HIDDEN_SIZE,
    )
    assert graph.input_buffers["pixel_values"].shape[0] == (
        output_capacity * merge_size**2
    )
    if merge_size > 1:
        assert graph.input_buffers["merge_indices"].shape == (
            output_capacity,
            merge_size**2,
        )
    else:
        assert "merge_indices" not in graph.input_buffers

    mm_kwargs = {
        "images": _make_pixtral_images(
            [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3)],
            merge_size,
        )
    }
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)
    actual = manager.execute(mm_kwargs)
    _assert_pixtral_outputs_close(actual, expected)
    assert manager.graph_hits == 5
    assert manager.graph_misses == 0

    replay_kwargs = {
        "images": _make_pixtral_images([(1, 1)], merge_size),
    }
    replay_expected = _get_eager_pixtral_outputs(model, replay_kwargs)
    replay_actual = manager.execute(replay_kwargs)
    _assert_pixtral_outputs_close(replay_actual, replay_expected)
    assert manager.graph_hits == 6
    assert manager.graph_misses == 0


@_requires_cuda
def test_pixtral_encoder_cudagraph_budget_boundary_and_fallback(
    default_vllm_config,
    dist_init,
) -> None:
    torch.manual_seed(0)
    merge_size = 2
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(model)
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    exact_budget_kwargs = {
        "images": _make_pixtral_images([(1, 1), (2, 4), (2, 4)], merge_size)
    }
    exact_budget_expected = _get_eager_pixtral_outputs(model, exact_budget_kwargs)
    exact_budget_actual = manager.execute(exact_budget_kwargs)
    _assert_pixtral_outputs_close(exact_budget_actual, exact_budget_expected)
    assert manager.graph_hits == 3
    assert manager.graph_misses == 0

    oversized_kwargs = {
        "images": _make_pixtral_images([(3, 6)], merge_size),
    }
    oversized_expected = _get_eager_pixtral_outputs(model, oversized_kwargs)
    oversized_actual = manager.execute(oversized_kwargs)
    _assert_pixtral_outputs_close(oversized_actual, oversized_expected)
    assert manager.graph_hits == 3
    assert manager.graph_misses == 1


@_requires_cuda
@pytest.mark.parametrize(
    "backend",
    [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.FLASHINFER],
    ids=["flash-attn", "flashinfer"],
)
@pytest.mark.parametrize("merge_size", [1, 2], ids=["pixtral", "ministral"])
def test_pixtral_encoder_cudagraph_ignores_poisoned_padding_tail(
    backend: AttentionBackendEnum,
    merge_size: int,
    default_vllm_config,
    dist_init,
    monkeypatch,
) -> None:
    """Padding must never reach real tokens, whatever is left in the tail.

    The attention metadata padding (``cu_seqlens``, and the two-section
    FlashInfer offsets plus ``sequence_lengths``) is what isolates the unused
    tail of the capture buffers. Poisoning that tail with NaN turns a leak into
    a hard failure instead of a plausible-looking number.
    """
    if backend == AttentionBackendEnum.FLASHINFER and not _IS_BLACKWELL_OR_NEWER:
        pytest.skip("Pixtral FlashInfer CUDA graph coverage requires SM100 or newer")

    torch.manual_seed(0)
    model = _make_tiny_pixtral_encoder(
        backend,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(model)
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    mm_kwargs = {
        "images": _make_pixtral_images([(1, 2), (2, 1)], merge_size),
    }
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)

    padded = _poison_cudagraph_padding(monkeypatch)
    actual = manager.execute(mm_kwargs)

    assert padded["float"], "batch filled the budget exactly; poison was a no-op"
    if merge_size > 1:
        assert padded["int"], "merge_indices was not padded; poison was a no-op"
    assert manager.graph_misses == 0

    # Assert on the post-processed items, not on the raw output buffer: for
    # merge_size=1 the padded output rows are legitimately NaN, and
    # scatter_output_slices never hands them to a caller.
    for item in actual:
        assert torch.isfinite(item.float()).all()
    _assert_pixtral_outputs_close(actual, expected)


@_requires_cuda
def test_pixtral_encoder_cudagraph_mixed_graph_and_eager(
    default_vllm_config,
    dist_init,
) -> None:
    torch.manual_seed(0)
    merge_size = 2
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(model)
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    mm_kwargs = {
        "images": _make_pixtral_images([(1, 1), (3, 6)], merge_size),
    }
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)
    actual = manager.execute(mm_kwargs)
    _assert_pixtral_outputs_close(actual, expected)
    assert manager.graph_hits == 1
    assert manager.graph_misses == 1


@_requires_cuda
def test_pixtral_encoder_cudagraph_splits_on_token_budget(
    default_vllm_config,
    dist_init,
) -> None:
    torch.manual_seed(0)
    merge_size = 2
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(model)
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    mm_kwargs = {
        "images": _make_pixtral_images([(2, 5), (2, 5)], merge_size),
    }
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)
    actual = manager.execute(mm_kwargs)
    _assert_pixtral_outputs_close(actual, expected)
    assert manager.graph_hits == 2
    assert manager.graph_misses == 0


@_requires_cuda
def test_pixtral_encoder_cudagraph_multi_budget_capacity_map(
    default_vllm_config,
    dist_init,
) -> None:
    torch.manual_seed(0)
    merge_size = 1
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        merge_size,
        default_vllm_config,
    )
    manager = _make_pixtral_encoder_cudagraph_manager(
        model, token_budgets=[8, _CUDA_GRAPH_TOKEN_BUDGET]
    )
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    assert set(manager.budget_graphs["default"]) == {8, _CUDA_GRAPH_TOKEN_BUDGET}
    capacity_ptrs = list(model._encoder_cudagraph_input_capacities)
    assert len(capacity_ptrs) >= 2
    assert len(set(capacity_ptrs)) == len(capacity_ptrs)
    assert manager._find_smallest_fitting_budget_given_tokens(4) == 8
    assert manager._find_smallest_fitting_budget_given_tokens(12) == (
        _CUDA_GRAPH_TOKEN_BUDGET
    )

    small_kwargs = {"images": _make_pixtral_images([(2, 2)], merge_size)}
    small_expected = _get_eager_pixtral_outputs(model, small_kwargs)
    small_actual = manager.execute(small_kwargs)
    _assert_pixtral_outputs_close(small_actual, small_expected)
    assert manager.graph_hits == 1
    assert manager.graph_misses == 0

    large_kwargs = {
        "images": _make_pixtral_images([(2, 2), (2, 4)], merge_size),
    }
    large_expected = _get_eager_pixtral_outputs(model, large_kwargs)
    large_actual = manager.execute(large_kwargs)
    _assert_pixtral_outputs_close(large_actual, large_expected)
    assert manager.graph_hits == 3
    assert manager.graph_misses == 0


@_requires_cuda
def test_pixtral_encoder_cudagraph_survives_recapture(
    default_vllm_config,
    dist_init,
) -> None:
    """Capture, release, capture again -- the sequence the model runner runs.

    Memory profiling captures encoder graphs with a throwaway manager and
    clears them, then the real manager captures again against the same model.
    ``_encoder_cudagraph_input_capacities`` is keyed by buffer address and is
    never cleared, so a second capture must re-register its own buffers before
    any replay reads them.
    """
    torch.manual_seed(0)
    merge_size = 2
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        merge_size,
        default_vllm_config,
    )

    profiling_manager = _make_pixtral_encoder_cudagraph_manager(model)
    profiling_manager.capture(graph_pool=current_platform.graph_pool_handle())
    profiling_manager.clear()

    manager = _make_pixtral_encoder_cudagraph_manager(model)
    manager.capture(graph_pool=current_platform.graph_pool_handle())

    graph_meta = manager.budget_graphs["default"][_CUDA_GRAPH_TOKEN_BUDGET]
    input_capacity = graph_meta.input_buffers["pixel_values"].shape[0]
    capacities = model._encoder_cudagraph_input_capacities
    for key in ("cu_seqlens", "sequence_lengths"):
        buf = graph_meta.input_buffers.get(key)
        if buf is not None:
            assert capacities[buf.data_ptr()] == input_capacity

    mm_kwargs = {
        "images": _make_pixtral_images([(1, 1), (1, 2)], merge_size),
    }
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)
    actual = manager.execute(mm_kwargs)
    _assert_pixtral_outputs_close(actual, expected)
    assert manager.graph_misses == 0


@_requires_cuda
def test_pixtral_encoder_cudagraph_manager_init_uses_budget_range(
    default_vllm_config,
    dist_init,
) -> None:
    torch.manual_seed(0)
    model = _make_tiny_pixtral_encoder(
        AttentionBackendEnum.FLASH_ATTN,
        1,
        default_vllm_config,
    )
    compilation_config = default_vllm_config.compilation_config
    compilation_config.encoder_cudagraph_token_budgets = [_CUDA_GRAPH_TOKEN_BUDGET]
    compilation_config.encoder_cudagraph_max_vision_items_per_batch = 0
    default_vllm_config.scheduler_config.max_num_batched_tokens = 8192

    min_budget, max_budget = model.get_encoder_cudagraph_budget_range(
        default_vllm_config
    )
    assert (min_budget, max_budget) == (3136, 8192)

    manager = EncoderCudaGraphManager(
        default_vllm_config,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        model=model,
    )
    assert manager.token_budgets == [_CUDA_GRAPH_TOKEN_BUDGET]
    expected_batch_size = min(max_budget // min_budget, _CUDA_GRAPH_TOKEN_BUDGET)
    assert manager.max_batch_size == expected_batch_size
    assert manager.config.modalities == ["image"]
    assert (
        manager.config.padding_logics["cu_seqlens"].__func__
        is PixtralForConditionalGeneration._pad_encoder_cudagraph_cumulative_seqlens
    )
    assert (
        manager.config.padding_logics["sequence_lengths"].__func__
        is PixtralForConditionalGeneration._pad_encoder_cudagraph_sequence_lengths
    )

    manager.capture(graph_pool=current_platform.graph_pool_handle())
    mm_kwargs = {"images": _make_pixtral_images([(1, 1)], merge_size=1)}
    expected = _get_eager_pixtral_outputs(model, mm_kwargs)
    actual = manager.execute(mm_kwargs)
    _assert_pixtral_outputs_close(actual, expected)
    assert manager.graph_hits == 1
    assert manager.graph_misses == 0


@pytest.mark.parametrize(
    "use_data_parallel,tp_size,expected_divisor",
    [
        (False, 1, 1),
        (False, 2, 2),
        # The ViT data-parallel path replicates the encoder, so TP does not
        # shard the hidden dimension and the offsets must not be divided.
        (True, 2, 1),
    ],
)
def test_pixtral_encoder_cudagraph_flashinfer_offset_scale_follows_tp(
    use_data_parallel: bool,
    tp_size: int,
    expected_divisor: int,
    monkeypatch,
) -> None:
    """FlashInfer offsets are byte-strides, so they must track the TP shard."""
    hidden_size = 128
    input_capacity = 8

    model = PixtralForConditionalGeneration.__new__(PixtralForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vision_args = SimpleNamespace(hidden_size=hidden_size)

    src_qko = torch.tensor([0, 8] + [20] * 7, dtype=torch.int32)
    src_v = torch.tensor([0, 24] + [60] * 7, dtype=torch.int32)
    src_cu_seqlens = torch.cat((src_qko, src_v))
    dst_cu_seqlens = torch.empty_like(src_cu_seqlens)
    model._encoder_cudagraph_input_capacities = {
        dst_cu_seqlens.data_ptr(): input_capacity
    }

    monkeypatch.setattr(
        "vllm.model_executor.models.pixtral.is_vit_use_data_parallel",
        lambda: use_data_parallel,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.pixtral.get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    model._pad_encoder_cudagraph_flashinfer_cu_seqlens(dst_cu_seqlens, src_cu_seqlens)

    expected = torch.empty_like(src_cu_seqlens)
    _pad_pixtral_flashinfer_cu_seqlens(
        expected,
        src_cu_seqlens,
        input_capacity,
        hidden_size // expected_divisor,
    )
    assert dst_cu_seqlens.tolist() == expected.tolist()


@pytest.mark.parametrize(
    "merge_size,scheduler_max,model_max,expected",
    [
        (1, 4096, 8192, (256, 4096)),
        (2, 4096, 8192, (64, 4096)),
        (1, 8192, 4096, (256, 4096)),
        (2, 32, 64, (32, 32)),
    ],
)
def test_pixtral_encoder_cudagraph_budget_range(
    merge_size: int,
    scheduler_max: int,
    model_max: int,
    expected: tuple[int, int],
) -> None:
    model = PixtralForConditionalGeneration.__new__(PixtralForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vision_args = SimpleNamespace(
        patch_size=14,
        spatial_merge_size=merge_size,
    )
    model.patch_merger = object() if merge_size > 1 else None
    model.model_config = SimpleNamespace(max_model_len=model_max)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=scheduler_max)
    )

    assert model.get_encoder_cudagraph_budget_range(vllm_config) == expected


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

        assert encoder_cudagraph_manager.graph_hits >= len(IMG_URLS)
        assert encoder_cudagraph_manager.graph_misses == 0

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
