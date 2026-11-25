# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVS video pruning through the engine on a tiny random-weight Nemotron
Shortcut: the prompt keeps the EVS share of video tokens, and the model returns
one embedding per placeholder position, for frames and for tubelets."""

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import TokenizersBackend
from transformers.models.pixtral import PixtralImageProcessor, PixtralProcessor

from tests.conftest import VllmRunner
from tests.models.nemotron_shortcut.utils import make_config, save_checkpoint_config
from vllm import SamplingParams
from vllm.multimodal.video_prune.evs import compute_retained_tokens_count
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")


def save_processor(path) -> int:
    """A Pixtral processor with a tiny byte-level tokenizer; returns its size."""
    special_tokens = ["<unk>", "<img>", "<image_break>", "</img>"]
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevelDecoder()
    backend.train_from_iterator(
        ["<video>", "This is a video:\nFrame 1 sampled at 0.50 seconds"],
        BpeTrainer(
            vocab_size=512,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        ),
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=special_tokens[1:],
    )
    PixtralProcessor(
        image_processor=PixtralImageProcessor(
            size={"longest_edge": 112}, patch_size={"height": 28, "width": 28}
        ),
        tokenizer=tokenizer,
        patch_size=14,
        spatial_merge_size=2,
        image_token="<img>",
        image_break_token="<image_break>",
        image_end_token="</img>",
    ).save_pretrained(path)
    return len(tokenizer)


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_video_pruning(tmp_path, temporal_patch_size: int) -> None:
    vocab_size = save_processor(tmp_path)
    text = make_config(
        layer_types=["linear_attention", "mlp", "full_attention", "mlp"],
        vocab_size=vocab_size,
        hidden_size=1024,
        num_attention_heads=8,
        head_dim=128,
        # GDP's kernels take 64-dim heads and 128-dim keys.
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    save_checkpoint_config(
        tmp_path,
        text,
        image_token_id=1,
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            patch_size=14,
            image_size=112,
        ),
        video_temporal_patch_size=temporal_patch_size,
    )
    rate, num_frames = 0.5, 8
    # 84 x 112 frames make a 3 x 4 grid of merged 28-pixel patches.
    frames = np.random.default_rng(0).integers(
        0, 256, (num_frames, 84, 112, 3), np.uint8
    )
    metadata = {
        "fps": 2.0,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
    }
    with VllmRunner(
        str(tmp_path),
        trust_remote_code=True,  # the checkpoint's config is remote code
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=2048,
        block_size=None,  # type: ignore[arg-type]  # the hybrid model picks it
        enable_chunked_prefill=True,  # the GDP layers' prefix caching needs it
        limit_mm_per_prompt={"image": 0, "video": 1},
        media_io_kwargs={"video": {"num_frames": num_frames}},
        video_pruning_rate=rate,
    ) as vllm:
        (output,) = vllm.llm.generate(
            {"prompt": "<video>", "multi_modal_data": {"video": (frames, metadata)}},
            SamplingParams(max_tokens=1),
        )

    num_tubelets = num_frames // temporal_patch_size
    kept = compute_retained_tokens_count(3 * 4, num_tubelets, rate)
    assert output.prompt_token_ids.count(1) == kept < num_tubelets * 3 * 4
    assert len(output.outputs[0].token_ids) == 1
