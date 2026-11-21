# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio encoder CUDA graphs keep the outputs of the eager audio tower."""

import pytest

from vllm.assets.audio import AudioAsset
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODEL = "Qwen/Qwen3-ASR-0.6B"
PROMPT = (
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _check_audio_fallback_outputs(worker):
    import torch

    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    runner = worker.model_runner
    manager = runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    model = runner.model
    long_frames = (manager.token_budgets[-1] // 13 + 1) * 100
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    torch.manual_seed(0)
    with torch.inference_mode():
        for lengths in ([100, long_frames, 250], [100] * (manager.max_batch_size + 1)):
            kwargs = {
                "input_audio_features": torch.randn(
                    model.config.audio_config.num_mel_bins,
                    sum(lengths),
                    device=runner.device,
                    dtype=model.audio_tower.dtype,
                ),
                "audio_feature_lengths": torch.tensor(lengths),
                "feature_attention_mask": (
                    torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
                ),
            }
            misses = manager.graph_misses
            outputs = manager.execute(kwargs)
            expected = model.embed_multimodal(**kwargs)
            assert manager.graph_misses == misses + len(lengths)
            assert len(outputs) == len(expected)
            for out, ref in zip(outputs, expected):
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
            cache = EncoderCache()
            cache.encoder_outputs.update({str(i): out for i, out in enumerate(outputs)})
            del outputs, out
            for i in range(1, len(lengths)):
                cache.free_encoder_cache(str(i))
            out = cache.encoder_outputs["0"]
            assert out.untyped_storage().nbytes() == out.numel() * out.element_size()
            retained.append((out, expected[0]))
        for out, ref in retained:
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
@pytest.mark.parametrize("mm_encoder_attn_backend", ["FLASH_ATTN", "TRITON_ATTN"])
def test_qwen3_asr_audio_encoder_cudagraph(
    vllm_runner, mm_encoder_attn_backend, monkeypatch
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    audios = [
        [AudioAsset(name).audio_and_sample_rate]
        for name in ("mary_had_lamb", "winning_call")
    ]
    prompts = [PROMPT] * len(audios)
    outputs = {}
    for cudagraph_mm_encoder in (False, True):
        with vllm_runner(
            MODEL,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=2,
            kv_cache_memory_bytes=512 * 1024**2,
            limit_mm_per_prompt={"audio": 1},
            attention_config={"backend": mm_encoder_attn_backend},
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            compilation_config={"cudagraph_mm_encoder": cudagraph_mm_encoder},
        ) as vllm_model:
            outputs[cudagraph_mm_encoder] = vllm_model.generate_greedy_logprobs(
                prompts, max_tokens=64, num_logprobs=5, audios=audios
            )
            if cudagraph_mm_encoder:
                vllm_model.collective_rpc(_check_audio_fallback_outputs)

    check_logprobs_close(
        outputs_0_lst=outputs[False],
        outputs_1_lst=outputs[True],
        name_0="eager_encoder",
        name_1="cudagraph_encoder",
    )
