# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.outputs import CompletionOutput
from vllm.platforms import current_platform

from .vlm_utils.builders import sample_frames_with_video_metadata

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# The processor wraps every frame in <|vision_start|>...<|vision_end|> itself. The
# chat template's outer pair is left out: with it, the cache-miss path is itself
# wrong under pruning (#62937, fixed by #62959), and this test would compare against it.
VIDEO_PLACEHOLDER = "<|video_pad|>"

# Shared text after the video, several KV blocks long, so that a repeated request
# reuses the cache past the end of the video and the video is not encoded again.
TEXT_AFTER_VIDEO = (
    "You are reviewing a short home video. Look at the people, the objects they "
    "hold, where they are, what they do from the first frame to the last, and "
    "how the scene changes over time. Use plain words, name only what is "
    "visible, do not guess at things outside the frame, and keep the answer to "
    "a single short sentence that a child could understand. "
) * 3

# The cache hit prefills only the text after the cached blocks and the miss the
# whole prompt, so their logprobs differ at bf16 resolution and a near-tie can flip
# a greedy token. Wrong positions after the video move them by several nats instead.
# Measured with FLASH_ATTN and TRITON_ATTN on an RTX PRO 6000 at 16 and 24 frames:
# up to 0.62 with correct positions, at least 6.9 for a pruned video with the
# positions on main. check_logprobs_close passes on main here: each diverging token
# stays in the other side's top-5.
MAX_LOGPROB_GAP = 2.0


def qwen3_vl_chat_template(*query: str) -> str:
    return f"<|im_start|>user\n{''.join(query)}<|im_end|>\n<|im_start|>assistant\n"


def max_logprob_gap(miss: CompletionOutput, hit: CompletionOutput) -> float:
    """Largest logprob difference of a top-k token between the miss and the hit,
    over the steps whose context is still the same (up to and including the first
    differing token). A token listed on one side only counts with a lower bound:
    its logprob there minus the lowest logprob listed on the other side. A
    differing token outside the other side's top-k makes the gap unbounded.
    """
    assert miss.logprobs is not None and hit.logprobs is not None
    gap = 0.0
    for tok_miss, tok_hit, lp_miss, lp_hit in zip(
        miss.token_ids, hit.token_ids, miss.logprobs, hit.logprobs
    ):
        assert lp_miss is not None and lp_hit is not None
        a = {t: lp.logprob for t, lp in lp_miss.items()}
        b = {t: lp.logprob for t, lp in lp_hit.items()}
        for x, y in ((a, b), (b, a)):
            floor = min(y.values())
            gaps = (abs(v - y[t]) if t in y else v - floor for t, v in x.items())
            gap = max([gap, *gaps])
        if tok_miss != tok_hit:
            if tok_hit not in a or tok_miss not in b:
                return math.inf
            break
    return gap


@pytest.mark.core_model
@pytest.mark.skipif(
    current_platform.is_cpu(), reason="Video pruning is not tested on CPU"
)
@pytest.mark.parametrize(
    "video_pruning_method,video_pruning_rate",
    [("evs", 0.0), ("evs", 0.5), ("vidcom2", 0.5)],
    ids=["unpruned", "evs-0.5", "vidcom2-0.5"],
)
def test_qwen3_vl_pruned_video_prefix_cache_hit(
    vllm_runner, video_assets, video_pruning_method: str, video_pruning_rate: float
) -> None:
    """A prompt whose whole video is a prefix cache hit must score like the cache
    miss: the positions of the text after a pruned video do not depend on whether
    the video was encoded in this request.
    """
    video = sample_frames_with_video_metadata(
        (video_assets[0].np_ndarrays, video_assets[0].metadata), 16
    )
    prompt = qwen3_vl_chat_template(
        VIDEO_PLACEHOLDER,
        TEXT_AFTER_VIDEO,
        "Describe the video in one sentence.",
    )
    params = SamplingParams(temperature=0.0, max_tokens=32, logprobs=5)

    with vllm_runner(
        MODEL,
        runner="generate",
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"video": 1},
        video_pruning_rate=video_pruning_rate,
        video_pruning_method=video_pruning_method,
        enable_prefix_caching=True,
    ) as vllm_model:
        inputs = vllm_model.get_inputs([prompt], videos=[video])
        outputs = [vllm_model.llm.generate(inputs, params)[0] for _ in range(2)]
        vision_end = vllm_model.llm.get_tokenizer().convert_tokens_to_ids(
            "<|vision_end|>"
        )

    # The second request must reuse the cache past the end of the video, which
    # ends at the last <|vision_end|> (one per frame).
    ids = outputs[1].prompt_token_ids
    video_end = len(ids) - 1 - ids[::-1].index(vision_end)
    assert outputs[0].num_cached_tokens == 0
    assert outputs[1].num_cached_tokens > video_end

    miss, hit = (o.outputs[0] for o in outputs)
    gap = max_logprob_gap(miss, hit)
    assert gap < MAX_LOGPROB_GAP, (
        f"logprob gap {gap:.3f} between cache miss and cache hit\n"
        f"cache_miss:\t{miss.text!r}\ncache_hit:\t{hit.text!r}"
    )


@torch.inference_mode()
def _embed_dummy_video(worker) -> torch.Tensor:
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.utils import group_and_batch_mm_kwargs

    runner = worker.model_runner
    budget = MultiModalBudget(
        worker.vllm_config, MULTIMODAL_REGISTRY, enable_cache=False
    )
    items = budget.get_dummy_encoder_profile_inputs("video", 1)
    _, _, mm_kwargs = next(group_and_batch_mm_kwargs(items, device=runner.device))
    (embedding,) = runner.get_model().embed_multimodal(**mm_kwargs)
    return embedding.float().cpu()


@pytest.mark.core_model
def test_encoder_only_matches_full_model_for_pruned_video(vllm_runner, monkeypatch):
    """Pruned video outputs include the embeddings of timestamp and vision
    marker tokens, so an encoder-only instance must still produce them."""
    # collective_rpc sends a function, which needs the pickle fallback.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    embeddings = []
    for mm_encoder_only in (False, True):
        with vllm_runner(
            MODEL,
            mm_encoder_only=mm_encoder_only,
            video_pruning_rate=0.5,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 0, "video": 1},
        ) as model:
            (embedding,) = model.llm.collective_rpc(_embed_dummy_video)
            embeddings.append(embedding)

    full, encoder_only = embeddings
    torch.testing.assert_close(encoder_only, full)


def _observe_encoder_tokens(model):
    model.encoder_token_counts = []

    def record(module, args):
        model.encoder_token_counts.append(
            args[0].shape[0] // module.spatial_merge_size**2
        )

    model.visual.register_forward_pre_hook(record)


@pytest.mark.core_model
@pytest.mark.parametrize("pruning_rate", [0.0, 0.75])
def test_video_pruning_respects_encoder_budget(vllm_runner, pruning_rate, monkeypatch):
    # Enqueue every request before scheduling to make the batch deterministic.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    prompt = (
        "<|im_start|>user\n<|video_pad|>"
        "Describe the video in one sentence.<|im_end|>\n<|im_start|>assistant\n"
    )
    metadata = {
        "fps": 2.0,
        "duration": 8.0,
        "total_num_frames": 16,
        "frames_indices": list(range(16)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "video": [(np.full((16, 256, 256, 3), i * 30, np.uint8), metadata)]
            },
        }
        for i in range(8)
    ]
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 1},
        video_pruning_rate=pruning_rate,
        mm_processor_cache_gb=0,
        mm_processor_kwargs={
            "videos_kwargs": {
                "size": {"shortest_edge": 65536, "longest_edge": 1048576},
                "num_frames": 16,
            }
        },
    ) as runner:
        llm = runner.get_llm()
        budget = MultiModalBudget(
            llm.llm_engine.vllm_config, MULTIMODAL_REGISTRY
        ).encoder_compute_budget
        runner.apply_model(_observe_encoder_tokens)
        outputs = llm.generate(inputs, SamplingParams(temperature=0, max_tokens=16))
        counts = runner.apply_model(lambda model: model.encoder_token_counts)[0]

        assert len(outputs) == len(inputs)
        assert all(output.outputs[0].token_ids for output in outputs)
        assert sum(counts) > budget  # The workload requires multiple encoder steps.
        assert max(counts) <= budget, (counts, budget)
