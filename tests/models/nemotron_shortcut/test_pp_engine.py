# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism through the whole engine on a tiny random-weight
Nemotron Shortcut with `residual_in_fp32`, whose stages exchange fp32 streams."""

import random

from tests.conftest import VllmRunner
from tests.models.nemotron_shortcut.utils import (
    make_config,
    realistic_weights,
    save_checkpoint_config,
)
from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_marks
from vllm import SamplingParams

pytestmark = multi_gpu_marks(num_gpus=2)

VOCAB = 32
rng = random.Random(0)
# The last token id is the image token, so the prompts do not use it.
PROMPTS = [[rng.randrange(VOCAB - 1) for _ in range(n)] for n in (7, 300)]


def run(path, pipeline_parallel_size: int, **kwargs) -> list:
    with VllmRunner(
        str(path),
        trust_remote_code=True,  # the checkpoint's config is remote code
        load_format="dummy",
        skip_tokenizer_init=True,
        pipeline_parallel_size=pipeline_parallel_size,
        **kwargs,
        dtype="bfloat16",
        max_model_len=1024,
        block_size=None,  # type: ignore[arg-type]  # the hybrid model picks it
        enable_chunked_prefill=True,  # the GDP layers' prefix caching needs it
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as vllm:
        vllm.llm.apply_model(realistic_weights)
        params = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
        outputs = vllm.llm.generate([{"prompt_token_ids": p} for p in PROMPTS], params)
        return [(o.outputs[0].token_ids, "", o.outputs[0].logprobs) for o in outputs]


def test_pp_matches_one_stage(tmp_path, monkeypatch) -> None:
    """Two PP stages decode as one stage does: the second stage receives the
    fp32 streams the first sends."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model sends realistic_weights to the workers by pickle.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # No MoE layers: router near-ties would flip experts between the two runs.
    layers = ["linear_attention", "mlp", "full_attention", "mlp"]
    text = make_config(
        layer_types=layers * 2,
        vocab_size=VOCAB,
        hidden_size=1024,
        num_attention_heads=8,
        head_dim=128,
        # GDP's kernels take 64-dim heads and 128-dim keys.
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        residual_in_fp32=True,
    )
    save_checkpoint_config(tmp_path, text, image_token_id=VOCAB - 1)

    check_logprobs_close(
        outputs_0_lst=run(tmp_path, 1),
        outputs_1_lst=run(tmp_path, 2),
        name_0="one_stage",
        name_1="pp2",
    )


def test_pp_with_sdsa_matches_one_stage(tmp_path, monkeypatch) -> None:
    """With SDSA, the first stage has one SDSA layer to the two GDP layers of
    a GDP group, so its SDSA indexer cannot share the GDP layers' regions of
    the KV cache, and two stages still decode as one stage does."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    text = make_config(
        layer_types=["linear_attention", "full_attention"] * 3,
        vocab_size=VOCAB,
        hidden_size=1024,
        num_attention_heads=8,
        head_dim=256,
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    save_checkpoint_config(tmp_path, text, image_token_id=VOCAB - 1)

    sdsa = {"hf_overrides": {"text_config": {"index_topk": 512}}}
    check_logprobs_close(
        outputs_0_lst=run(tmp_path, 1, **sdsa),
        outputs_1_lst=run(tmp_path, 2, **sdsa),
        name_0="one_stage",
        name_1="pp2",
    )
