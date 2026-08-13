# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for sequence and token pooler head classes."""

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

_HIDDEN = 16
_BATCH = 3


def _make_params(
    n: int,
    *,
    task: str = "embed",
    dimensions: int | None = None,
    use_activation: bool | None = None,
) -> list[PoolingParams]:
    return [
        PoolingParams(task=task, dimensions=dimensions, use_activation=use_activation)
        for _ in range(n)
    ]


def _make_metadata(pooling_params: list[PoolingParams]) -> PoolingMetadata:
    n = len(pooling_params)
    return PoolingMetadata(
        prompt_lens=torch.ones(n, dtype=torch.long),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=pooling_params,
        pooling_states=[PoolingStates() for _ in range(n)],
    )


def _linear(in_f: int, out_f: int) -> nn.Linear:
    torch.manual_seed(42)
    return nn.Linear(in_f, out_f, bias=False)


# ---------------------------------------------------------------------------
# EmbeddingPoolerHead
# ---------------------------------------------------------------------------
class TestEmbeddingPoolerHead:
    def test_supported_tasks(self):
        head = EmbeddingPoolerHead()
        assert head.get_supported_tasks() == {"embed"}

    def test_passthrough(self):
        head = EmbeddingPoolerHead()
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH))
        out = head(x, meta)
        assert torch.equal(out, x)

    def test_head_dtype(self):
        head = EmbeddingPoolerHead(head_dtype=torch.float16)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH))
        out = head(x, meta)
        assert out.dtype == torch.float16

    def test_projector(self):
        proj = _linear(_HIDDEN, 8)
        head = EmbeddingPoolerHead(projector=proj)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH))
        out = head(x, meta)
        assert out.shape == (_BATCH, 8)
        assert torch.allclose(out, proj(x))

    def test_matryoshka_uniform(self):
        head = EmbeddingPoolerHead()
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, dimensions=4)
        meta = _make_metadata(params)
        out = head(x, meta)
        assert out.shape == (_BATCH, 4)
        assert torch.equal(out, x[..., :4])

    def test_matryoshka_mixed(self):
        head = EmbeddingPoolerHead()
        x = torch.randn(2, _HIDDEN)
        params = [
            PoolingParams(task="embed", dimensions=4),
            PoolingParams(task="embed", dimensions=8),
        ]
        meta = _make_metadata(params)
        out = head(x, meta)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape[-1] == 4
        assert out[1].shape[-1] == 8

    def test_matryoshka_mixed_with_none(self):
        head = EmbeddingPoolerHead()
        x = torch.randn(2, _HIDDEN)
        params = [
            PoolingParams(task="embed", dimensions=4),
            PoolingParams(task="embed", dimensions=None),
        ]
        meta = _make_metadata(params)
        out = head(x, meta)
        assert isinstance(out, list)
        assert out[0].shape[-1] == 4
        assert torch.equal(out[1], x[1])

    def test_activation_uniform_true(self):
        head = EmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, use_activation=True)
        meta = _make_metadata(params)
        out = head(x, meta)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(_BATCH), atol=1e-5)

    def test_activation_uniform_false(self):
        head = EmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, use_activation=False)
        meta = _make_metadata(params)
        out = head(x, meta)
        assert torch.equal(out, x)

    def test_activation_mixed_flags(self):
        head = EmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(2, _HIDDEN)
        params = [
            PoolingParams(task="embed", use_activation=True),
            PoolingParams(task="embed", use_activation=False),
        ]
        meta = _make_metadata(params)
        out = head(x, meta)
        assert isinstance(out, list)
        norm_0 = torch.linalg.norm(out[0], dim=-1)
        assert torch.allclose(norm_0, torch.ones(1), atol=1e-5)
        assert torch.equal(out[1], x[1])

    def test_list_input_gets_stacked(self):
        head = EmbeddingPoolerHead()
        tensors = [torch.randn(_HIDDEN) for _ in range(_BATCH)]
        meta = _make_metadata(_make_params(_BATCH))
        out = head(tensors, meta)
        assert out.shape == (_BATCH, _HIDDEN)
        expected = torch.stack(tensors)
        assert torch.equal(out, expected)

    def test_projector_then_matryoshka(self):
        proj = _linear(_HIDDEN, 8)
        head = EmbeddingPoolerHead(projector=proj)
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, dimensions=4)
        meta = _make_metadata(params)
        out = head(x, meta)
        assert out.shape == (_BATCH, 4)
        assert torch.equal(out, proj(x)[..., :4])

    def test_matryoshka_then_activation(self):
        head = EmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, dimensions=4, use_activation=True)
        meta = _make_metadata(params)
        out = head(x, meta)
        assert out.shape == (_BATCH, 4)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(_BATCH), atol=1e-5)

    def test_empty_batch(self):
        head = EmbeddingPoolerHead()
        x = torch.randn(0, _HIDDEN)
        meta = _make_metadata([])
        out = head(x, meta)
        assert out.shape == (0, _HIDDEN)


# ---------------------------------------------------------------------------
# ClassifierPoolerHead
# ---------------------------------------------------------------------------
class TestClassifierPoolerHead:
    def test_supported_tasks(self):
        head = ClassifierPoolerHead()
        assert head.get_supported_tasks() == {"classify"}

    def test_passthrough(self):
        head = ClassifierPoolerHead()
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert torch.equal(out, x)

    def test_head_dtype(self):
        head = ClassifierPoolerHead(head_dtype=torch.float16)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert out.dtype == torch.float16

    def test_classifier(self):
        clf = _linear(_HIDDEN, 3)
        head = ClassifierPoolerHead(classifier=clf)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert out.shape == (_BATCH, 3)
        assert torch.allclose(out, clf(x))

    def test_logit_mean(self):
        head = ClassifierPoolerHead(logit_mean=2.0)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert torch.allclose(out, x - 2.0)

    def test_logit_sigma(self):
        head = ClassifierPoolerHead(logit_sigma=0.5)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert torch.allclose(out, x / 0.5)

    def test_platt_scaling_combined(self):
        head = ClassifierPoolerHead(logit_mean=1.0, logit_sigma=2.0)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        assert torch.allclose(out, (x - 1.0) / 2.0)

    def test_activation_uniform_true(self):
        head = ClassifierPoolerHead(activation=PoolerNormalize())
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, task="classify", use_activation=True)
        meta = _make_metadata(params)
        out = head(x, meta)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(_BATCH), atol=1e-5)

    def test_activation_uniform_false(self):
        head = ClassifierPoolerHead(activation=PoolerNormalize())
        x = torch.randn(_BATCH, _HIDDEN)
        params = _make_params(_BATCH, task="classify", use_activation=False)
        meta = _make_metadata(params)
        out = head(x, meta)
        assert torch.equal(out, x)

    def test_activation_mixed_flags(self):
        head = ClassifierPoolerHead(activation=PoolerNormalize())
        x = torch.randn(2, _HIDDEN)
        params = [
            PoolingParams(task="classify", use_activation=True),
            PoolingParams(task="classify", use_activation=False),
        ]
        meta = _make_metadata(params)
        out = head(x, meta)
        assert isinstance(out, list)
        norm_0 = torch.linalg.norm(out[0], dim=-1)
        assert torch.allclose(norm_0, torch.ones(1), atol=1e-5)
        assert torch.equal(out[1], x[1])

    def test_list_input_gets_stacked(self):
        head = ClassifierPoolerHead()
        tensors = [torch.randn(_HIDDEN) for _ in range(_BATCH)]
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(tensors, meta)
        assert out.shape == (_BATCH, _HIDDEN)
        expected = torch.stack(tensors)
        assert torch.equal(out, expected)

    def test_classifier_then_platt_scaling(self):
        clf = _linear(_HIDDEN, 3)
        head = ClassifierPoolerHead(classifier=clf, logit_mean=1.0, logit_sigma=2.0)
        x = torch.randn(_BATCH, _HIDDEN)
        meta = _make_metadata(_make_params(_BATCH, task="classify"))
        out = head(x, meta)
        expected = (clf(x) - 1.0) / 2.0
        assert torch.allclose(out, expected)

    def test_empty_batch(self):
        head = ClassifierPoolerHead()
        x = torch.randn(0, _HIDDEN)
        meta = _make_metadata([])
        out = head(x, meta)
        assert out.shape == (0, _HIDDEN)


# ---------------------------------------------------------------------------
# TokenEmbeddingPoolerHead
# ---------------------------------------------------------------------------
class TestTokenEmbeddingPoolerHead:
    def test_supported_tasks(self):
        head = TokenEmbeddingPoolerHead()
        assert head.get_supported_tasks() == {"token_embed"}

    def test_passthrough(self):
        head = TokenEmbeddingPoolerHead()
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed")
        out = head.forward_chunk(x, param)
        assert torch.equal(out, x)

    def test_none_chunked_prefill(self):
        head = TokenEmbeddingPoolerHead()
        param = PoolingParams(task="token_embed")
        out = head.forward_chunk(None, param)
        assert out is None

    def test_head_dtype(self):
        head = TokenEmbeddingPoolerHead(head_dtype=torch.float16)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed")
        out = head.forward_chunk(x, param)
        assert out.dtype == torch.float16

    def test_projector(self):
        proj = _linear(_HIDDEN, 8)
        head = TokenEmbeddingPoolerHead(projector=proj)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed")
        out = head.forward_chunk(x, param)
        assert out.shape == (5, 8)
        assert torch.allclose(out, proj(x))

    def test_matryoshka_truncation(self):
        head = TokenEmbeddingPoolerHead()
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed", dimensions=4)
        out = head.forward_chunk(x, param)
        assert out.shape == (5, 4)
        assert torch.equal(out, x[..., :4])

    def test_activation_true(self):
        head = TokenEmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed", use_activation=True)
        out = head.forward_chunk(x, param)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_activation_false(self):
        head = TokenEmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed", use_activation=False)
        out = head.forward_chunk(x, param)
        assert torch.equal(out, x)

    def test_projector_then_matryoshka(self):
        proj = _linear(_HIDDEN, 8)
        head = TokenEmbeddingPoolerHead(projector=proj)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed", dimensions=4)
        out = head.forward_chunk(x, param)
        assert out.shape == (5, 4)
        assert torch.equal(out, proj(x)[..., :4])

    def test_matryoshka_then_activation(self):
        head = TokenEmbeddingPoolerHead(activation=PoolerNormalize())
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_embed", dimensions=4, use_activation=True)
        out = head.forward_chunk(x, param)
        assert out.shape == (5, 4)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_forward_mixed_batch_chunked_prefill(self):
        head = TokenEmbeddingPoolerHead()
        pooled_data = [torch.randn(5, _HIDDEN), None, torch.randn(3, _HIDDEN)]
        params = _make_params(3, task="token_embed")
        meta = _make_metadata(params)
        out = head(pooled_data, meta)
        assert len(out) == 3
        assert torch.equal(out[0], pooled_data[0])
        assert out[1] is None
        assert torch.equal(out[2], pooled_data[2])

    def test_forward_empty_batch(self):
        head = TokenEmbeddingPoolerHead()
        meta = _make_metadata([])
        out = head([], meta)
        assert out == []


# ---------------------------------------------------------------------------
# TokenClassifierPoolerHead
# ---------------------------------------------------------------------------
class TestTokenClassifierPoolerHead:
    def test_supported_tasks(self):
        head = TokenClassifierPoolerHead()
        assert head.get_supported_tasks() == {"token_classify"}

    def test_passthrough(self):
        head = TokenClassifierPoolerHead()
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert torch.equal(out, x)

    def test_none_chunked_prefill(self):
        head = TokenClassifierPoolerHead()
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(None, param)
        assert out is None

    def test_head_dtype(self):
        head = TokenClassifierPoolerHead(head_dtype=torch.float16)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert out.dtype == torch.float16

    def test_classifier(self):
        clf = _linear(_HIDDEN, 3)
        head = TokenClassifierPoolerHead(classifier=clf)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert out.shape == (5, 3)
        assert torch.allclose(out, clf(x))

    def test_logit_mean(self):
        head = TokenClassifierPoolerHead(logit_mean=2.0)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert torch.allclose(out, x - 2.0)

    def test_logit_sigma(self):
        head = TokenClassifierPoolerHead(logit_sigma=0.5)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert torch.allclose(out, x / 0.5)

    def test_platt_scaling_combined(self):
        head = TokenClassifierPoolerHead(logit_mean=1.0, logit_sigma=2.0)
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify")
        out = head.forward_chunk(x, param)
        assert torch.allclose(out, (x - 1.0) / 2.0)

    def test_activation_true(self):
        head = TokenClassifierPoolerHead(activation=PoolerNormalize())
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify", use_activation=True)
        out = head.forward_chunk(x, param)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_activation_false(self):
        head = TokenClassifierPoolerHead(activation=PoolerNormalize())
        x = torch.randn(5, _HIDDEN)
        param = PoolingParams(task="token_classify", use_activation=False)
        out = head.forward_chunk(x, param)
        assert torch.equal(out, x)

    def test_forward_mixed_batch_chunked_prefill(self):
        head = TokenClassifierPoolerHead()
        pooled_data = [torch.randn(5, _HIDDEN), None, torch.randn(3, _HIDDEN)]
        params = _make_params(3, task="token_classify")
        meta = _make_metadata(params)
        out = head(pooled_data, meta)
        assert len(out) == 3
        assert torch.equal(out[0], pooled_data[0])
        assert out[1] is None
        assert torch.equal(out[2], pooled_data[2])

    def test_forward_empty_batch(self):
        head = TokenClassifierPoolerHead()
        meta = _make_metadata([])
        out = head([], meta)
        assert out == []
