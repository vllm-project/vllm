# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import (
    CompilationConfig,
    ModelConfig,
    SchedulerConfig,
    SpeechToTextConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models.cohere_asr import (
    CohereAsrForConditionalGeneration,
    CohereASRModel,
    RelPositionMultiHeadAttention,
)
from vllm.utils.torch_utils import set_default_torch_dtype


class _ReferenceRelPositionMultiHeadAttention(RelPositionMultiHeadAttention):
    """Relative attention implementation before score accumulation is fused."""

    def forward(self, query, key, value, mask, pos_emb=None):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        assert pos_emb is not None
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        return self.forward_attention(v, scores, mask)


@pytest.mark.parametrize(("query_len", "key_len"), [(2, 2), (3, 5)])
def test_relative_attention_fuses_score_accumulation(monkeypatch, query_len, key_len):
    torch.manual_seed(7)
    batch, heads, model_dim = 2, 4, 32
    optimized = RelPositionMultiHeadAttention(heads, model_dim, None, None)
    reference = _ReferenceRelPositionMultiHeadAttention(heads, model_dim, None, None)
    reference.load_state_dict(optimized.state_dict())

    query = torch.randn(batch, query_len, model_dim)
    key = torch.randn(batch, key_len, model_dim)
    value = torch.randn(batch, key_len, model_dim)
    pos_emb = torch.randn(1, query_len + key_len - 1, model_dim)
    expected = reference(query, key, value, None, pos_emb)

    original_baddbmm = torch.baddbmm
    calls = 0

    def tracked_baddbmm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_baddbmm(*args, **kwargs)

    monkeypatch.setattr(torch, "baddbmm", tracked_baddbmm)
    actual = optimized(query, key, value, None, pos_emb)

    torch.testing.assert_close(actual, expected)
    assert calls == 1


@pytest.mark.parametrize(
    ("audio_duration_s", "expected_tokens"),
    # window_stride=0.01s @ 16kHz -> 160-sample hop; subsampling_factor=8.
    # frames = floor(duration * 16000 / 160); tokens = ceil(frames / 8).
    [(1.0, 13), (10.0, 125), (30.0, 375)],
)
def test_get_num_audio_tokens_streaming_estimate(audio_duration_s, expected_tokens):
    """The duration-based estimate must convert ``window_stride`` (seconds) to a
    sample hop and divide by the encoder subsampling factor. Values are pinned to
    concrete numbers so a rounding regression is actually caught."""
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            preprocessor={"window_stride": 0.01, "sample_rate": 16000},
            encoder={"subsampling_factor": 8},
        )
    )
    stt_config = SpeechToTextConfig(sample_rate=16000)

    got = CohereAsrForConditionalGeneration.get_num_audio_tokens(
        audio_duration_s, stt_config, model_config
    )

    assert got == expected_tokens


class _CohereASRWeightLoadingModel(CohereASRModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.register_buffer("pos_bias_u", torch.zeros(2, 2))
        self.register_buffer("pos_bias_v", torch.zeros(2, 2))
        self.attention = RelPositionMultiHeadAttention(
            n_head=2,
            n_feat=4,
            pos_bias_u=self.pos_bias_u,
            pos_bias_v=self.pos_bias_v,
        )


@pytest.mark.cpu_test
def test_load_weights_preserves_runtime_bias_dtype() -> None:
    with set_default_torch_dtype(torch.float16):
        model = _CohereASRWeightLoadingModel()

    loaded_weight = torch.ones_like(model.pos_bias_u, dtype=torch.float32)
    model.load_weights([("pos_bias_u", loaded_weight), ("pos_bias_v", loaded_weight)])

    assert model.pos_bias_u.dtype == torch.float16
    assert model.pos_bias_v.dtype == torch.float16
    assert model.attention.pos_bias_u.dtype == torch.float16
    assert model.attention.pos_bias_v.dtype == torch.float16
    torch.testing.assert_close(model.pos_bias_u, loaded_weight.half())

    output = model.attention(
        query=torch.randn(1, 3, 4, dtype=torch.float16),
        key=torch.randn(1, 3, 4, dtype=torch.float16),
        value=torch.randn(1, 3, 4, dtype=torch.float16),
        mask=None,
        pos_emb=torch.randn(1, 5, 4, dtype=torch.float16),
    )

    assert output.dtype == torch.float16
    assert output.shape == (1, 3, 4)


@pytest.fixture
def cohere_encoder_graph(tmp_path, dist_init):
    from transformers import CohereAsrConfig

    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    hf_config = CohereAsrConfig(
        encoder_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_mel_bins": 8,
            "subsampling_conv_channels": 4,
            "conv_kernel_size": 3,
        },
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=64,
        vocab_size=64,
        architectures=["CohereAsrForConditionalGeneration"],
    )
    hf_config.save_pretrained(tmp_path)

    def legacy_config(config):
        config.encoder = {
            "feat_in": 8,
            "n_layers": 2,
            "d_model": 32,
            "feat_out": -1,
            "causal_downsampling": False,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 4,
            "ff_expansion_factor": 2,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "xscaling": False,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 3,
            "conv_norm_type": "batch_norm",
            "conv_context_size": None,
        }
        config.transf_decoder = {
            "config_dict": {
                "hidden_size": 256,
                "num_layers": 1,
                "num_attention_heads": 4,
                "inner_size": 512,
                "max_sequence_length": 64,
                "hidden_act": "relu",
            }
        }
        config.head = {"hidden_size": 256, "num_classes": 64}
        config.preprocessor = {"sample_rate": 16000, "window_stride": 0.01}
        return config

    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=64,
        hf_overrides=legacy_config,
    )
    config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            is_encoder_decoder=True,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_max_vision_items_per_batch=4,
        ),
    )
    with (
        set_current_vllm_config(config),
        torch.backends.cudnn.flags(allow_tf32=False),
        torch.random.fork_rng(),
    ):
        torch.manual_seed(7)
        # The decoder requires a half-precision cross-attention backend. Keep
        # the encoder/projection in float32 for strict replay comparisons.
        with set_default_torch_dtype(torch.bfloat16):
            model = CohereAsrForConditionalGeneration(vllm_config=config).cuda().eval()
        model.model.encoder.float()
        model.model.encoder_decoder_proj.float()
        manager = EncoderCudaGraphManager(
            vllm_config=config,
            device=torch.device("cuda"),
            dtype=torch.float32,
            model=model,
        )
        with torch.inference_mode():
            manager.capture(torch.cuda.graph_pool_handle())
            yield model, manager
        manager.clear()


def test_encoder_graph_replay_preserves_frames_and_owned_outputs(cohere_encoder_graph):
    """Replay must update lengths/data without overwriting retained outputs."""
    from vllm.utils.torch_utils import async_tensor_h2d

    model, manager = cohere_encoder_graph
    features = torch.randn(3, 8, 513, device="cuda")
    lengths = torch.tensor([513, 173, 93])
    inputs = {"input_features": features, "length": lengths}
    with torch.inference_mode():
        reference = model.embed_multimodal(
            input_features=features, length=async_tensor_h2d(lengths, device="cuda")
        )
        actual = manager.execute(inputs)
        assert len(actual) == len(reference) == 3
        for expected, output in zip(reference, actual):
            torch.testing.assert_close(output, expected)
        retained = [output.clone() for output in actual]
        lengths.copy_(torch.tensor([511, 170, 90]))
        inputs["input_features"] = features + 0.1
        updated = manager.execute(inputs)
        expected = model.embed_multimodal(
            input_features=inputs["input_features"],
            length=async_tensor_h2d(lengths, device="cuda"),
        )
        assert len(updated) == len(expected) == 3
        for first, saved, output, target in zip(actual, retained, updated, expected):
            assert torch.equal(first, saved)
            torch.testing.assert_close(output, target)
    assert manager.get_cumulative_stats()["graph_hits"] == 6


def test_encoder_graph_oversized_axis_uses_original_eager_path(cohere_encoder_graph):
    """An uncaptured time axis must preserve the original encoder result."""
    model, manager = cohere_encoder_graph
    features = torch.randn(1, 8, 4097, device="cuda")
    lengths = torch.tensor([4097])
    with torch.inference_mode():
        reference = model.embed_multimodal(input_features=features, length=lengths)
        actual = manager.execute({"input_features": features, "length": lengths})
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    stats = manager.get_cumulative_stats()
    assert stats["graph_hits"] == 0 and stats["graph_misses"] == 1
