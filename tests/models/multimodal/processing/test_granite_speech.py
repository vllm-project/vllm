# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import math

import numpy as np
import pytest
import torch

from vllm.config import DeviceConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.models.granite_speech import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechEncoderProjector,
    encode_audio,
    get_encoder_batch_size,
    get_encoder_windows,
    get_max_encoder_call_blocks,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget, get_mm_max_toks_per_item

from ...utils import build_model_context

MODEL = "ibm-granite/granite-speech-3.3-2b"


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL, limit_mm_per_prompt={"audio": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _process_audio(processor, seconds: float):
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    prompt = processor.dummy_inputs.get_dummy_text({"audio": 1})
    return processor(
        prompt, mm_items=processor.info.parse_mm_data({"audio": [(audio, 16000)]})
    )


def _num_features(processed) -> int:
    return processed["mm_kwargs"]["audio"][0]["input_features"].data.shape[0]


def test_audio_budget_covers_ten_minute_clips(processor):
    budget = get_mm_max_toks_per_item(processor, {"audio": 1}, None)["audio"]
    dummy = processor.get_dummy_mm_inputs({"audio": 1})
    ten_minutes = _process_audio(processor, 600)

    assert dummy["mm_placeholders"]["audio"][0].get_num_embeds() == budget
    assert ten_minutes["mm_placeholders"]["audio"][0].get_num_embeds() <= budget


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_num_seqs", "hint_seconds"),
    [
        (8192, 256, None),
        (12288, 256, None),
        (16384, 1, None),
        (16384, 256, None),
        (2048, 256, 30),
        (8192, 256, 120),
        (16384, 2, 30),
        (8192, 256, 0.05),
    ],
)
def test_encoder_calls_fit_the_profiled_call(
    max_num_batched_tokens, max_num_seqs, hint_seconds
):
    audio_limit = (
        1 if hint_seconds is None else {"count": 1, "length": int(hint_seconds * 16000)}
    )
    ctx = build_model_context(
        MODEL,
        model_config_kwargs={"max_model_len": 32768},
        limit_mm_per_prompt={"audio": audio_limit},
    )
    budget = MultiModalBudget(
        VllmConfig(
            model_config=ctx.model_config,
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=32768,
                is_encoder_decoder=False,
            ),
            device_config=DeviceConfig(device="cpu"),
        ),
        MULTIMODAL_REGISTRY,
        enable_cache=False,
    )
    processor = budget.processor
    config = processor.info.get_hf_config()
    context_size = config.encoder_config.context_size
    align = math.lcm(context_size, config.window_size)
    max_blocks = get_max_encoder_call_blocks(ctx.model_config)

    dummy_features = _num_features(processor.get_dummy_mm_inputs({"audio": 1}))
    assert len(budget.get_dummy_encoder_profile_inputs()) >= 1
    assert max_blocks == math.ceil(dummy_features / context_size)

    for features in range(1, dummy_features + 1, 25):
        batch_size = get_encoder_batch_size(config, features, max_blocks)
        assert batch_size * math.ceil(features / context_size) <= max_blocks

    longest = budget.encoder_cache_size // 10
    for seconds in [s for s in (2, 5, 60, 600) if s < longest] + [longest]:
        processed = _process_audio(processor, seconds)
        assert (
            processed["mm_placeholders"]["audio"][0].get_num_embeds()
            <= budget.encoder_cache_size
        )
        num_features = _num_features(processed)
        windows = get_encoder_windows(config, num_features, max_blocks)
        assert windows[0][1] == 0
        assert windows[-1][2] == num_features
        for (_, _, keep_end, _), (_, next_keep_begin, _, _) in zip(
            windows, windows[1:]
        ):
            assert keep_end == next_keep_begin
        for begin, keep_begin, keep_end, end in windows:
            assert begin <= keep_begin < keep_end <= end
            assert begin % context_size == 0
            assert keep_begin % align == 0
        call_blocks = [
            math.ceil((end - begin) / context_size) for begin, _, _, end in windows
        ]
        assert max(call_blocks) <= max_blocks


def _tower(processor):
    config = copy.deepcopy(processor.info.get_hf_config())
    config.encoder_config.num_layers = 2
    config.encoder_config.context_size = 20
    torch.manual_seed(0)
    encoder = GraniteSpeechCTCEncoder(config=config.encoder_config, prefix="encoder")
    projector = GraniteSpeechEncoderProjector(
        config=config, cache_config=None, prefix="projector"
    )
    for module in (encoder, projector):
        for param in module.parameters():
            torch.nn.init.normal_(param, std=0.1)
        for layer in module.modules():
            quant_method = getattr(layer, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(layer)
        module.eval()
    return config, encoder, projector


@torch.inference_mode()
def test_encode_audio_matches_one_call_per_clip(dist_init, processor):
    config, encoder, projector = _tower(processor)
    tokens_per_window = config.window_size // config.downsample_rate
    features = [torch.randn(n, 160) for n in (100, 100, 1000, 100, 100)]
    sizes = [
        math.ceil(feats.shape[0] / config.window_size) * tokens_per_window
        for feats in features
    ]

    outputs = encode_audio(encoder, projector, config, features, sizes, max_blocks=12)

    assert len(get_encoder_windows(config, 1000, 12)) > 1
    for feats, size, output in zip(features, sizes, outputs):
        expected = projector(encoder(feats.unsqueeze(0)))[0][:size]
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
