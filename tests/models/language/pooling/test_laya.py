# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LayaForDecision against a PyTorch port of Laya's `DecisionModel`.

Published Laya checkpoints need `convert_checkpoint.py` before vLLM can load
them, so this builds a small random checkpoint in the same layout. A batch
mixing question types and option counts checks the per-sequence question
type, marker gathering and temperature buckets, and that a prompt without
markers gets an empty output rather than failing the engine.
"""

import json

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

CLS, SEP, MASK = 1, 2, 4
QTYPES = ("choice", "score", "noul")
QTYPE_TOKEN_IDS = [10, 11, 12]
TEMPERATURE = [0.7, 1.3, 2.0]
TEMPERATURE_BY_OPTIONS = {"choice:11+": 3.0}


def _config() -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=512,
        # Mixes enough context that each option marker scores differently.
        initializer_range=0.2,
        pad_token_id=0,
        cls_token_id=CLS,
        sep_token_id=SEP,
    )


class DecisionModel(nn.Module):
    """`laya.common.DecisionModel` at inference."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        d = config.hidden_size
        self.encoder = ModernBertModel(config)
        layer = nn.TransformerEncoderLayer(
            d, d // 64, 4 * d, 0.0, batch_first=True, norm_first=True
        )
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids])
        qtype = QTYPE_TOKEN_IDS.index(ids[1])
        h = self.encoder(input_ids=input_ids).last_hidden_state
        h = self.head(h + self.type_emb.weight[qtype])[0]

        logits = self.scorer(h[input_ids[0] == MASK]).squeeze(-1)
        p = torch.softmax(logits, -1)
        k = len(logits)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum() / torch.log(torch.tensor(k))
        top2 = p.topk(2).values
        feats = torch.stack([top2[0], top2[0] - top2[1], ent, torch.tensor(k / 255)])
        act = torch.softmax(self.act_head(torch.cat([h[0], feats])), -1)

        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        t = TEMPERATURE_BY_OPTIONS.get(f"{QTYPES[qtype]}:{size}", TEMPERATURE[qtype])
        scores = torch.softmax(logits / t, -1)
        return torch.cat([scores[:, None], act.expand(k, -1)], -1)


def _prompt(gen: torch.Generator, qtype: int, num_options: int) -> list[int]:
    def words(n: int) -> list[int]:
        return torch.randint(20, 512, (n,), generator=gen).tolist()

    ids = [CLS, QTYPE_TOKEN_IDS[qtype], *words(5), SEP]
    for _ in range(num_options):
        ids += [MASK, *words(2)]
    return ids + [SEP, *words(12), SEP]


@torch.inference_mode()
def test_laya_matches_reference(vllm_runner, tmp_path, monkeypatch):
    # Building the reference here leaves torch threads running; forking the
    # engine after that can hang it.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.manual_seed(0)
    config = _config()
    reference = DecisionModel(config).eval()
    # Spread the scores and type embeddings so a wrong marker, question type
    # or temperature changes the output well beyond the tolerance.
    nn.init.normal_(reference.scorer[3].weight, std=1.0)
    nn.init.normal_(reference.type_emb.weight, std=1.0)

    save_file(
        {k: v.contiguous() for k, v in reference.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    hf_config = config.to_dict()
    hf_config["architectures"] = ["LayaForDecision"]
    hf_config["laya_config"] = {
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "temperature": TEMPERATURE,
        "temperature_by_options": TEMPERATURE_BY_OPTIONS,
        "mask_token_id": MASK,
        "qtype_token_ids": QTYPE_TOKEN_IDS,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config))

    gen = torch.Generator().manual_seed(0)
    prompts = [
        _prompt(gen, qtype, num_options)
        for qtype, num_options in [(0, 4), (1, 3), (2, 2), (0, 12), (2, 2)]
    ]
    expected = [reference(ids) for ids in prompts]
    prompts.insert(2, [CLS, 99, 100, SEP])
    expected.insert(2, torch.empty(0, 3))

    with vllm_runner(
        str(tmp_path),
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        skip_tokenizer_init=True,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.llm.encode(
            [{"prompt_token_ids": ids} for ids in prompts],
            pooling_task="token_classify",
        )

    for want, output in zip(expected, outputs):
        torch.testing.assert_close(
            output.outputs.data.cpu().float(), want, atol=1e-2, rtol=0
        )
