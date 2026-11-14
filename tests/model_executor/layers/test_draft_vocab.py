# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.draft_vocab import (
    DraftVocab,
    load_draft_token_ids,
    load_draft_token_map,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.online.lm_head import (
    refresh_quantized_lm_heads,
    restore_quantized_lm_heads,
    share_source_lm_heads,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _head(vocab: int = VOCAB, device: str = "cpu") -> ParallelLMHead:
    head = ParallelLMHead(vocab, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    head = head.to(device)
    head.quant_method.process_weights_after_loading(head)
    return head


def _hidden(n: int = 64, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(n, HIDDEN, generator=gen).to(device)


@pytest.mark.cpu_test
def test_load_draft_token_ids(tmp_path):
    path = tmp_path / "map.pt"
    torch.save(torch.tensor([9, 2, 9, 4]), path)
    assert load_draft_token_ids(str(path), extra_ids=[7, 2]).tolist() == [2, 4, 7, 9]


@pytest.mark.cpu_test
def test_pruned_head_maps_draft_ids_to_target_ids(default_vllm_config):
    """EAGLE3/DFlash pruned heads: logits land on their d2t target ids."""
    head = _head(len(DRAFT_IDS))
    draft_vocab = DraftVocab(LogitsProcessor(len(DRAFT_IDS)), VOCAB)
    d2t = draft_vocab.draft_id_to_target_id
    assert d2t is not None
    d2t.data.copy_(torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)))

    hidden = _hidden()
    logits = draft_vocab.compute_logits(head, hidden)
    assert torch.equal(logits[:, DRAFT_IDS], draft_vocab.logits_processor(head, hidden))
    assert torch.isinf(logits).sum() == hidden.shape[0] * (VOCAB - len(DRAFT_IDS))
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_rows", [0, 16])
def test_token_map_drafts_over_listed_and_picked_rows(
    default_vllm_config, dist_init, device, num_rows
):
    """Each token drafts over the list plus `num_rows` picked rows, with the
    exact logits of the full head on all of them."""
    head = _head(device=device)
    hidden = _hidden(device=device)
    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    full = draft_vocab.compute_logits(head, hidden).float()
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), num_rows)

    logits = draft_vocab.compute_logits(head, hidden).float()
    kept = logits.isfinite()
    assert torch.all(kept[:, DRAFT_IDS])
    assert torch.all(kept.sum(dim=-1) == len(DRAFT_IDS) + num_rows)
    torch.testing.assert_close(logits[kept], full[kept], atol=1e-4, rtol=1e-4)
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantized rows need CUDA"
)


def _bf16_head(
    monkeypatch, vocab: int, hidden_size: int, cls=ParallelLMHead
) -> VocabParallelEmbedding:
    # Online methods read the model dtype from the current config.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    return cls(vocab, hidden_size, disable_tp=True)


@requires_cuda
@pytest.mark.parametrize(
    ("quantization", "max_rel_err"), [("fp8", 0.05), ("nvfp4", 0.15)]
)
@pytest.mark.parametrize("extra_listed_id", [False, True])
def test_quantized_rows_track_bf16_rows(
    default_vllm_config,
    dist_init,
    monkeypatch,
    quantization,
    max_rel_err,
    extra_listed_id,
):
    """Quantized rows preserve logical token IDs, including unaligned maps."""
    vocab, hidden_size = 4000, 512
    head = _bf16_head(monkeypatch, vocab, hidden_size)
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(head.weight.shape, generator=gen) * 0.02
    token_ids = torch.arange(0, vocab, 4)
    if extra_listed_id:
        token_ids = torch.cat((token_ids, torch.tensor([1]))).sort().values
        weight = weight.abs()
    head.weight.data.copy_(weight)
    head = head.to("cuda", torch.bfloat16)
    hidden = torch.randn(
        512,
        hidden_size,
        generator=torch.Generator(device="cuda").manual_seed(1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if extra_listed_id:
        # Dummy zero rows must not win when every real logit is negative.
        hidden[-1].fill_(-1)

    draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    draft_vocab.load_token_map(head, token_ids, 256, quantization)
    out = draft_vocab.compute_logits(head, hidden).float()
    full = LogitsProcessor(vocab)(head, hidden).float()
    kept = out.isfinite()
    assert out.shape == (hidden.shape[0], vocab)
    assert torch.all(kept[:, token_ids])
    assert torch.all(kept.sum(dim=-1) == len(token_ids) + 256)
    if extra_listed_id:
        assert torch.all(out[-1, kept[-1]] < 0)
    assert (out[kept] - full[kept]).norm() / full[kept].norm() < max_rel_err


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_quantized_listed_rows_follow_target_head(
    default_vllm_config, dist_init, monkeypatch, tmp_path, quantization
):
    """Refreshing the drafter's quantized heads re-derives the listed rows from
    the updated target lm_head, in place. The head is a tied input embedding,
    as in Qwen3.5."""
    vocab, hidden_size = 4000, 512
    head = _bf16_head(monkeypatch, vocab, hidden_size, VocabParallelEmbedding)
    head = head.to("cuda", torch.bfloat16)
    head.weight.data.normal_(std=0.02)
    token_ids = list(range(1, vocab, 3))
    token_map = tmp_path / "map.json"
    token_map.write_text(str(token_ids))
    drafter = torch.nn.Module()
    drafter.lm_head = head
    drafter.draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_token_map=str(token_map),
            draft_token_map_dynamic_rows=0,
            draft_lm_head_quantization=quantization,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(), try_get_generation_config=dict
        ),
    )
    load_draft_token_map(drafter, vllm_config)
    rows = drafter.draft_token_map_head
    assert rows is drafter.draft_vocab.rows
    ptrs = {k: t.data_ptr() for k, t in rows.state_dict().items()}

    # A copy of some rows is never swapped in for the full head.
    share_source_lm_heads(drafter)
    assert drafter.draft_token_map_head is rows
    restore_quantized_lm_heads(drafter, refresh=False)

    head.weight.data.normal_(std=0.02)
    hidden = torch.randn(64, hidden_size, device="cuda", dtype=torch.bfloat16)
    fresh = DraftVocab(LogitsProcessor(vocab), vocab)
    fresh.load_token_map(head, torch.tensor(token_ids), 0, quantization)
    expected = fresh.compute_logits(head, hidden)
    assert not torch.equal(drafter.draft_vocab.compute_logits(head, hidden), expected)

    refresh_quantized_lm_heads(drafter)
    assert torch.equal(drafter.draft_vocab.compute_logits(head, hidden), expected)
    assert {k: t.data_ptr() for k, t in rows.state_dict().items()} == ptrs


def _nvfp4_checkpoint(vocab: int, hidden: int) -> dict[str, torch.Tensor]:
    """A ModelOpt NVFP4 lm_head as stored in the checkpoint."""
    gen = torch.Generator().manual_seed(3)
    return {
        "weight": torch.randint(
            0, 256, (vocab, hidden // 2), dtype=torch.uint8, generator=gen
        ),
        "weight_scale": (torch.rand(vocab, hidden // 16, generator=gen) + 0.5).to(
            torch.float8_e4m3fn
        ),
        "weight_scale_2": torch.tensor(0.01),
    }


def _dequant(packed, scales, global_scale) -> torch.Tensor:
    return dequantize_to_dtype(
        packed, scales, global_scale, torch.float32, swizzle=False
    )


class _DequantKernel:
    """Stands in for the NVFP4 kernel, which needs CUDA."""

    def input_quant_key(self):
        return None

    def process_weights_after_loading(self, layer):
        pass

    def apply_weights(self, layer, x, bias=None):
        w = _dequant(layer.weight, layer.weight_scale, layer.weight_global_scale)
        return x @ w.t().to(x.dtype)


def _nvfp4_head(
    weights: dict[str, torch.Tensor], quant_method: str, device: str = "cpu"
) -> ParallelLMHead:
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method=quant_method, is_checkpoint_nvfp4_serialized=True
    )
    vocab, hidden = weights["weight"].shape[0], weights["weight"].shape[1] * 2
    with torch.device(device):
        head = ParallelLMHead(
            vocab,
            hidden,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="lm_head",
            disable_tp=True,
        )
    params = dict(head.named_parameters())
    for name, weight in weights.items():
        params[name].weight_loader(params[name], weight.to(device))
    head.quant_method.process_weights_after_loading(head)
    return head


def _keep(draft_vocab: DraftVocab, checkpoint, head) -> None:
    weights = [("lm_head." + n, w) for n, w in checkpoint.items()]
    weights.append(("model.fc.weight", torch.zeros(1)))
    assert list(draft_vocab.keep_head_weights(weights, head)) == weights


@pytest.mark.cpu_test
def test_quantized_head_drafts_over_checkpoint_rows(
    default_vllm_config, dist_init, monkeypatch
):
    """A quantized lm_head drafts over its own checkpoint rows, with the logits
    of the full head on them, and releases the kept tensors once loaded."""
    monkeypatch.setattr(
        modelopt, "select_linear_kernel", lambda *a, **kw: _DequantKernel()
    )
    checkpoint = _nvfp4_checkpoint(VOCAB, HIDDEN)
    head = _nvfp4_head(checkpoint, "W4A16_NVFP4")
    hidden = _hidden().bfloat16()
    full = LogitsProcessor(VOCAB)(head, hidden)

    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB, keep_quantized_head=True)
    # Serving sets head_dtype to the model dtype.
    draft_vocab.logits_processor.head_dtype = torch.bfloat16
    _keep(draft_vocab, checkpoint, head)
    kept = [weakref.ref(w) for w in draft_vocab.head_weights.values()]
    assert len(kept) == len(checkpoint)
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS))
    assert not draft_vocab.head_weights
    assert all(ref() is None for ref in kept)

    logits = draft_vocab.compute_logits(head, hidden)
    expected = torch.full_like(full, float("-inf"))
    expected[:, DRAFT_IDS] = full[:, DRAFT_IDS]
    torch.testing.assert_close(logits, expected)
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))

    def load(checkpoint, **kwargs):
        draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB, True)
        _keep(draft_vocab, checkpoint, head)
        draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), **kwargs)

    with pytest.raises(ValueError, match="quantized in the checkpoint"):
        load(checkpoint, quantization="fp8")
    # 128x128 block scales are not indexed by output row.
    with pytest.raises(ValueError, match="not indexed by output row"):
        load({**checkpoint, "weight_scale": torch.ones(2, 1)})
    with pytest.raises(ValueError, match="keep_head_weights"):
        DraftVocab(LogitsProcessor(VOCAB), VOCAB).load_token_map(
            head, torch.tensor(DRAFT_IDS)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 kernels need CUDA")
@pytest.mark.parametrize("quant_method", ["W4A16_NVFP4", "NVFP4"])
def test_nvfp4_checkpoint_head_rows_match_full_head(
    default_vllm_config, dist_init, quant_method
):
    """Listed and dynamic rows of an NVFP4 checkpoint head on the real kernels."""
    vocab, hidden_size, num_rows = 4096, 512, 256
    checkpoint = _nvfp4_checkpoint(vocab, hidden_size)
    if quant_method == "NVFP4":
        checkpoint["input_scale"] = torch.tensor(0.01)
    head = _nvfp4_head(checkpoint, quant_method, "cuda")
    hidden = torch.randn(64, hidden_size, device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(0, vocab, 4)
    full = LogitsProcessor(vocab)(head, hidden).float()
    weight = _dequant(
        checkpoint["weight"], checkpoint["weight_scale"], checkpoint["weight_scale_2"]
    )
    reference = hidden.float() @ weight.cuda().t()

    def draft(dynamic_rows):
        draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab, True)
        _keep(draft_vocab, checkpoint, head)
        draft_vocab.load_token_map(head, token_ids, dynamic_rows)
        return draft_vocab.compute_logits(head, hidden).float()

    logits = draft(0)
    kept = logits.isfinite()
    assert torch.all(kept.sum(dim=-1) == token_ids.numel())
    torch.testing.assert_close(logits[kept], full[kept], atol=1e-2, rtol=1e-2)

    logits = draft(num_rows)
    kept = logits.isfinite()
    assert torch.all(kept.sum(dim=-1) == token_ids.numel() + num_rows)
    # The dynamic rows run with unquantized activations.
    rel_err = (logits[kept] - reference[kept]).norm() / reference[kept].norm()
    assert rel_err < (0.02 if quant_method == "W4A16_NVFP4" else 0.2)
