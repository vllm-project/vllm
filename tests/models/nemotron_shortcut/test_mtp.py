# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP head drafts from the target's input stream, where image and video
positions hold the vision embeddings."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.models.nemotron_shortcut.nvidia.mtp import NemotronShortcutMTP


def test_mtp_merges_multimodal_embeddings():
    assert supports_multimodal_embeddings(NemotronShortcutMTP)
    table = torch.arange(8, dtype=torch.float32).repeat(3, 1).T  # row i holds i
    drafter = SimpleNamespace(get_input_embeddings=lambda input_ids: table[input_ids])
    input_ids = torch.tensor([1, 5, 6, 2])

    text_only = NemotronShortcutMTP.embed_input_ids(drafter, input_ids)
    assert torch.equal(text_only, table[input_ids])

    mm_embeds = [torch.full((2, 3), -1.0)]
    is_multimodal = torch.tensor([False, True, True, False])
    merged = NemotronShortcutMTP.embed_input_ids(
        drafter, input_ids, mm_embeds, is_multimodal=is_multimodal
    )
    assert torch.equal(merged[[0, 3]], table[[1, 2]])
    assert torch.equal(merged[[1, 2]], mm_embeds[0])

    with pytest.raises(ValueError, match="is_multimodal"):
        NemotronShortcutMTP.embed_input_ids(drafter, input_ids, mm_embeds)


def test_dense_checkpoint_without_index_topk_generates_with_mtp(tmp_path, vllm_runner):
    from vllm import SamplingParams

    from .utils import make_config, save_checkpoint_config

    text = make_config(
        layer_types=["full_attention", "mlp"],
        hidden_size=512,
        head_dim=128,
        num_nextn_predict_layers=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    del text.index_topk
    save_checkpoint_config(tmp_path, text, image_token_id=text.vocab_size - 1)
    with vllm_runner(
        str(tmp_path),
        trust_remote_code=True,  # the test writes its checkpoint's config module
        dtype="bfloat16",
        load_format="dummy",
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=128,
        max_num_seqs=2,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    ) as runner:
        (request,) = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}],
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        )
        metrics = {m.name: m for m in runner.llm.get_metrics()}
        assert len(request.outputs[0].token_ids) == 8
        assert metrics["vllm:spec_decode_num_drafts"].value > 0
