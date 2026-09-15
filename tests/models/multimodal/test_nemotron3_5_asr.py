# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingEncoderConfig,
)
from transformers import (
    Nemotron3_5AsrForRNNT as HFNemotron3_5AsrForRNNT,
)
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoder,
)

from vllm.model_executor.models.config import Nemotron3_5AsrForRNNTConfig
from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrAudioEncoder,
    Nemotron3_5AsrDecodeState,
    Nemotron3_5AsrForRNNT,
    _decode_next_token,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.nemotron3_5_asr import (
    Nemotron3_5AsrModelState,
)

pytestmark = pytest.mark.core_model


@pytest.mark.parametrize("tokenizer_mode", ["auto", "nemotron3_5_asr"])
def test_nemotron_asr_model_config(tokenizer_mode: str) -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(decoder_hidden_size=640),
        model_arch_config=SimpleNamespace(hidden_size=1024),
        tokenizer_mode=tokenizer_mode,
        max_logprobs=20,
    )

    Nemotron3_5AsrForRNNTConfig.verify_and_update_model_config(config)

    assert config.model_arch_config.hidden_size == 640
    assert config.tokenizer_mode == "nemotron3_5_asr"
    assert config.max_logprobs == 0


def test_nemotron_asr_rejects_unsupported_tokenizer_mode() -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(decoder_hidden_size=640),
        model_arch_config=SimpleNamespace(hidden_size=1024),
        tokenizer_mode="hf",
        max_logprobs=20,
    )

    with pytest.raises(ValueError, match="tokenizer mode"):
        Nemotron3_5AsrForRNNTConfig.verify_and_update_model_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        (None, "use_v2_model_runner", False, "Model Runner V2"),
        ("model_config", "enforce_eager", False, "enforce-eager"),
        ("parallel_config", "tensor_parallel_size", 2, "TP=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1"),
        ("model_config", "quantization", "fp8", "quantization"),
        (None, "speculative_config", object(), "speculative decoding"),
    ],
)
def test_nemotron_asr_rejects_unsupported_execution_config(
    section: str | None,
    field: str,
    value: object,
    error: str,
) -> None:
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True, quantization=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )
    target = getattr(config, section) if section else config
    setattr(target, field, value)

    with pytest.raises(ValueError, match=error):
        Nemotron3_5AsrForRNNTConfig.verify_and_update_config(config)


def _get_tiny_config() -> Nemotron3_5AsrConfig:
    encoder_config = NemotronAsrStreamingEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=32,
        scale_input=False,
        sliding_window=9,
        default_num_lookahead_tokens=3,
    )
    return Nemotron3_5AsrConfig(
        encoder_config=encoder_config,
        decoder_hidden_size=8,
        num_prompts=8,
        prompt_intermediate_size=16,
        default_prompt_id=3,
    )


def _get_vllm_config(config: Nemotron3_5AsrConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
    )


@pytest.mark.parametrize(
    ("num_mel_frames", "physical_frames", "valid_frames"),
    [(26, 5, 4), (128, 17, 17)],
)
def test_nemotron_audio_encoder_preserves_batch_and_valid_lengths(
    num_mel_frames: int,
    physical_frames: int,
    valid_frames: int,
) -> None:
    torch.manual_seed(0)
    config = _get_tiny_config()
    config.encoder_config.num_hidden_layers = 2
    model = Nemotron3_5AsrAudioEncoder(config).eval()
    input_features = torch.randn(2, num_mel_frames, 8)
    attention_mask = torch.zeros(2, num_mel_frames, dtype=torch.bool)
    attention_mask[0, : num_mel_frames - 1] = True
    attention_mask[1, :17] = True

    with torch.inference_mode():
        output, output_mask = model(
            input_features,
            attention_mask,
            prompt_ids=torch.tensor([2, 3]),
        )
        unpadded_output, unpadded_mask = model(
            input_features[1:2, :17],
            torch.ones(1, 17, dtype=torch.bool),
            prompt_ids=torch.tensor([3]),
        )

    assert output.shape == (2, physical_frames, 8)
    assert output_mask is not None
    assert output_mask.sum(-1).tolist() == [valid_frames, 3]
    assert torch.isfinite(output).all()
    assert unpadded_mask is not None
    assert unpadded_mask.sum().item() == 3
    torch.testing.assert_close(output[1, :3], unpadded_output[0, :3])


class _ScriptedJoint(torch.nn.Module):
    def __init__(self, token_ids: list[int], vocab_size: int):
        super().__init__()
        self.tokens = iter(token_ids)
        self.vocab_size = vocab_size

    def forward(self, decoder_hidden_states, encoder_hidden_states):
        logits = torch.full((1, self.vocab_size), -1.0)
        logits[0, next(self.tokens)] = 1.0
        return logits


def test_nemotron_rnnt_advances_on_blank_and_symbol_limit() -> None:
    config = _get_tiny_config()
    config.max_symbols_per_step = 2
    blank = config.blank_token_id
    state = Nemotron3_5AsrDecodeState(
        encoder_frames=torch.zeros(3, config.decoder_hidden_size),
        decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
        last_token_id=blank,
    )
    decoder = Nemotron3_5AsrRNNTDecoder(config).eval()
    joint = _ScriptedJoint([2, blank, blank, 3, 0], config.vocab_size)

    with torch.inference_mode():
        assert (
            _decode_next_token(
                state,
                decoder,
                joint,
                blank_token_id=blank,
                max_symbols_per_step=2,
            )
            == 2
        )
        assert (state.frame_idx, state.symbols_at_frame) == (0, 1)
        assert (
            _decode_next_token(
                state,
                decoder,
                joint,
                blank_token_id=blank,
                max_symbols_per_step=2,
            )
            == 3
        )
        assert (state.frame_idx, state.symbols_at_frame) == (2, 1)
        assert (
            _decode_next_token(
                state,
                decoder,
                joint,
                blank_token_id=blank,
                max_symbols_per_step=2,
            )
            == 0
        )
        assert (state.frame_idx, state.symbols_at_frame) == (3, 0)
        assert (
            _decode_next_token(
                state,
                decoder,
                joint,
                blank_token_id=blank,
                max_symbols_per_step=2,
            )
            is None
        )


def test_nemotron_eager_dummy_prepare_profiles_decoder_without_live_mutation() -> None:
    config = _get_tiny_config()
    config.max_symbols_per_step = 1
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    state = Nemotron3_5AsrModelState.__new__(Nemotron3_5AsrModelState)
    state.model = model
    state.device = torch.device("cpu")
    state.dtype = torch.float32
    state.request_indices = {"live": 0}
    live_state = Nemotron3_5AsrDecodeState(
        encoder_frames=torch.zeros(1, config.decoder_hidden_size),
        decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
        last_token_id=config.blank_token_id,
    )
    state.decode_states = {"live": live_state}
    batch = InputBatch.make_dummy(
        4,
        10,
        InputBuffers(max_num_reqs=4, max_num_tokens=10, device=torch.device("cpu")),
    )

    inputs = state.prepare_inputs(batch, req_states=None)
    assert inputs["query_end_positions"] == batch.query_start_loc_np[1:].tolist()
    assert all(
        decode_state is not live_state for decode_state in inputs["decode_states"]
    )

    with (
        torch.inference_mode(),
        patch.object(model.decoder, "forward", wraps=model.decoder.forward) as decode,
        patch.object(model.joint, "forward", wraps=model.joint.forward) as join,
    ):
        model.forward(batch.input_ids, batch.positions, **inputs)

    assert decode.call_count == batch.num_reqs
    assert join.call_count == batch.num_reqs
    assert state.decode_states == {"live": live_state}
    assert live_state.frame_idx == 0
    assert not live_state.decoder_cache.is_initialized


@pytest.mark.parametrize("num_tokens_to_replay", [0, 3])
def test_nemotron_greedy_tokens_match_transformers(num_tokens_to_replay: int) -> None:
    torch.manual_seed(0)
    config = _get_tiny_config()
    config.max_symbols_per_step = 2
    hf_model = HFNemotron3_5AsrForRNNT(config).eval()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    model.load_weights(hf_model.state_dict().items())
    features = torch.randn(1, 26, config.encoder_config.num_mel_bins)
    mask = torch.zeros(1, 26, dtype=torch.bool)
    mask[:, :25] = True
    prompt_ids = torch.tensor([2])

    with torch.inference_mode():
        encoded, output_mask = model.audio_encoder(features, mask, prompt_ids)
        assert output_mask is not None
        state = Nemotron3_5AsrDecodeState(
            encoder_frames=encoded[0][output_mask[0]],
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
            last_token_id=config.blank_token_id,
            num_tokens_to_replay=num_tokens_to_replay,
        )
        token_ids = []
        for _ in range(64):
            hidden_states = model.forward(
                input_ids=torch.tensor([config.blank_token_id]),
                positions=torch.tensor([0]),
                decode_states=[state],
                query_end_positions=[1],
                decode_ready=[True],
            )
            token_id = int(model.compute_logits(hidden_states).argmax(dim=-1).item())
            if token_id == config.blank_token_id:
                break
            token_ids.append(token_id)
        assert state.frame_idx == state.encoder_frames.shape[0]

        reference = hf_model.generate(
            input_features=features,
            attention_mask=mask,
            prompt_ids=prompt_ids,
            decoder_start_token_id=config.blank_token_id,
            max_new_tokens=64,
        )

    assert (
        token_ids
        == [
            token_id
            for token_id in reference.sequences[0].tolist()
            if token_id != config.blank_token_id
        ][num_tokens_to_replay:]
    )


def test_nemotron_loads_all_transformers_weights() -> None:
    config = _get_tiny_config()
    reference = HFNemotron3_5AsrForRNNT(config).eval()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()

    loaded = model.load_weights(reference.state_dict().items())

    assert loaded == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        source_name = name.removeprefix("audio_encoder.")
        torch.testing.assert_close(tensor, reference.state_dict()[source_name])
