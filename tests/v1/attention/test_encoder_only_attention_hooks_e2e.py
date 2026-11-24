# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test of the encoder attention hooks through `LLM`: a BERT model
registered in-process runs its own Flex backend and metadata (a causal mask
within each request) on half of its layers and the selected backend on the
rest. Its outputs must match a reference that applies the two masks per layer,
and the stock model must be unchanged."""

from collections.abc import Iterable

import pytest
import torch
from torch import nn

from vllm import ModelRegistry
from vllm.compilation.decorators import support_torch_compile
from vllm.config import PoolerConfig, VllmConfig
from vllm.model_executor.models.bert import (
    BertEmbedding,
    BertEmbeddingModel,
    BertLayer,
    BertModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends import flex_attention

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ARCH = "HalfCausalBertForHookTest"
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello",
    (
        "Encoder attention hooks let a model own the attention layout of some "
        "of its layers while the remaining layers keep the backend vLLM selects."
    ),
]


def _causal_within_request(mask_mod):
    def mask(b, h, q_idx, kv_idx):
        return mask_mod(b, h, q_idx, kv_idx) & (kv_idx <= q_idx)

    return mask


class _CausalMetadata(flex_attention.FlexAttentionMetadata):
    def get_mask_mod(self):
        return _causal_within_request(super().get_mask_mod())


class _CausalBuilder(flex_attention.FlexAttentionMetadataBuilder):
    def _make_metadata(self, **kwargs):
        return _CausalMetadata(**kwargs)


# Same class name as the stock backend, as is natural in a plugin's module: the
# model's layers must still get their own attention group.
class FlexAttentionBackend(flex_attention.FlexAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return _CausalBuilder


def _num_own_layers(num_hidden_layers: int) -> int:
    return num_hidden_layers // 2


@support_torch_compile
class _HalfCausalBertModel(nn.Module):
    hf_to_vllm_mapper = BertModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        num_own = _num_own_layers(config.num_hidden_layers)
        self.embeddings = BertEmbedding(config)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                BertLayer(
                    config=config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.encoder.layer.{layer_idx}",
                    attn_backend=FlexAttentionBackend if layer_idx < num_own else None,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        for layer in self.encoder.layer:
            hidden_states = layer(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class _HalfCausalBertEmbeddingModel(BertEmbeddingModel):
    def _build_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return _HalfCausalBertModel(vllm_config=vllm_config, prefix=prefix)


def _token_embeddings(vllm_runner, model_kwargs) -> list[torch.Tensor]:
    with vllm_runner(
        MODEL,
        runner="pooling",
        dtype="float32",
        max_model_len=256,
        pooler_config=PoolerConfig(task="token_embed"),
        **model_kwargs,
    ) as m:
        outputs = m.token_embed(PROMPTS)
    return [torch.as_tensor(out, dtype=torch.float32) for out in outputs]


@torch.inference_mode()
def _reference(num_own: int) -> list[torch.Tensor]:
    """Transformers forward with a causal mask on the first `num_own` layers
    and the full (bidirectional) mask on the rest."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
    min_value = torch.finfo(torch.float32).min
    outputs = []
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        length = input_ids.shape[1]
        full = torch.zeros(1, 1, length, length, device="cuda")
        causal = torch.full_like(full, min_value).triu(1)
        hidden_states = model.embeddings(input_ids=input_ids)
        for layer_idx, layer in enumerate(model.encoder.layer):
            out = layer(
                hidden_states, attention_mask=causal if layer_idx < num_own else full
            )
            hidden_states = out[0] if isinstance(out, tuple) else out
        outputs.append(hidden_states[0].float().cpu())
    del model
    torch.accelerator.empty_cache()
    return outputs


def _assert_close(actual: list[torch.Tensor], expected: list[torch.Tensor]):
    # Compare directions: the pooler may normalize token embeddings. The
    # tolerance covers FP32 kernel differences (up to ~3e-4 between vLLM's
    # stock layers and Transformers); the masks differ by far more.
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape
        got = nn.functional.normalize(got, dim=-1)
        want = nn.functional.normalize(want, dim=-1)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-3)


def _max_diff(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> float:
    return max(
        (nn.functional.normalize(got, dim=-1) - nn.functional.normalize(want, dim=-1))
        .abs()
        .max()
        .item()
        for got, want in zip(actual, expected, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize(
    "stock_backend",
    [
        "TRITON_ATTN",
        pytest.param(
            "FLEX_ATTENTION",
            marks=pytest.mark.skip(
                reason="Two FlexAttention groups need "
                "https://github.com/vllm-project/vllm/pull/63985"
            ),
        ),
    ],
)
def test_model_owned_backend_on_half_the_layers(
    vllm_runner, monkeypatch, enforce_eager, stock_backend
):
    # The model is registered in this process, so the engine core runs here too.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    ModelRegistry.register_model(ARCH, _HalfCausalBertEmbeddingModel)
    common = dict(
        enforce_eager=enforce_eager,
        attention_config={"backend": stock_backend},
    )

    stock = _token_embeddings(vllm_runner, common)
    mixed = _token_embeddings(
        vllm_runner, dict(common, hf_overrides={"architectures": [ARCH]})
    )

    from transformers import AutoConfig

    num_own = _num_own_layers(AutoConfig.from_pretrained(MODEL).num_hidden_layers)
    stock_reference = _reference(num_own=0)
    mixed_reference = _reference(num_own=num_own)

    print(
        f"max diff: stock vs reference {_max_diff(stock, stock_reference):.2e}, "
        f"mixed vs two-mask reference {_max_diff(mixed, mixed_reference):.2e}, "
        f"mixed vs stock reference {_max_diff(mixed, stock_reference):.2e}"
    )
    # The stock model is unchanged; the model's layers run its own mask.
    _assert_close(stock, stock_reference)
    _assert_close(mixed, mixed_reference)
    assert _max_diff(mixed, stock_reference) > 1e-1
