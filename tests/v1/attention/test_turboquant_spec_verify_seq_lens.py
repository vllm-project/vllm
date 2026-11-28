# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant spec-verify rows must take causal seq_lens from the GPU seq_lens.

Under async scheduling seq_lens_cpu is an upper bound that assumes every draft
of the previous step was accepted. The decode kernel is spied on, so this runs
on CPU.
"""

import json
from pathlib import Path

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends import turboquant_attn
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)

_NUM_HEADS, _NUM_KV_HEADS, _HEAD_SIZE = 4, 2, 64


def _make_impl(model_dir: Path) -> TurboQuantAttentionImpl:
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": _NUM_HEADS * _HEAD_SIZE,
                "intermediate_size": 512,
                "num_attention_heads": _NUM_HEADS,
                "num_key_value_heads": _NUM_KV_HEADS,
                "num_hidden_layers": 1,
                "vocab_size": 1000,
                "max_position_embeddings": 2048,
            }
        )
    )
    model_config = ModelConfig(
        str(model_dir), skip_tokenizer_init=True, dtype="bfloat16"
    )
    with set_current_vllm_config(VllmConfig(model_config=model_config)):
        return TurboQuantAttentionImpl(
            num_heads=_NUM_HEADS,
            head_size=_HEAD_SIZE,
            scale=_HEAD_SIZE**-0.5,
            num_kv_heads=_NUM_KV_HEADS,
            kv_cache_dtype="turboquant_4bit_nc",
        )


def test_spec_verify_uses_gpu_seq_lens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # All 3 drafts of the last step were rejected: seq_lens is 100, the CPU
    # upper bound still says 103.
    q_len, seq_len, seq_len_cpu_upper_bound = 4, 100, 103
    seen: list[list[int]] = []

    def spy_decode(*, query: torch.Tensor, seq_lens: torch.Tensor, **_):
        seen.append(seq_lens.tolist())
        return torch.zeros_like(query)

    monkeypatch.setattr(
        turboquant_attn, "triton_turboquant_decode_attention", spy_decode
    )
    meta = TurboQuantMetadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        slot_mapping=torch.zeros(q_len, dtype=torch.int64),
        block_table=torch.zeros(1, 8, dtype=torch.int32),
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        num_actual_tokens=q_len,
        max_query_len=q_len,
        max_seq_len=seq_len_cpu_upper_bound,
        is_prefill=True,
        query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([seq_len_cpu_upper_bound], dtype=torch.int32),
    )
    query = torch.randn(q_len, _NUM_HEADS, _HEAD_SIZE)
    kv = torch.randn(q_len, _NUM_KV_HEADS, _HEAD_SIZE)
    _make_impl(tmp_path)._prefill_attention(
        query, kv, kv, torch.empty(0), meta, torch.empty(0), torch.empty(0)
    )

    assert seen == [list(range(seq_len - q_len + 1, seq_len + 1))]
