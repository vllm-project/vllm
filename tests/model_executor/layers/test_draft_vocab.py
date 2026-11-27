# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import ray
import torch
import torch.nn as nn

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.draft_vocab import (
    DraftVocab,
    draft_ids_to_target,
    load_draft_token_map,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    refresh_quantized_lm_heads,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import _quantize_nvfp4
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_gemv_bf16_fp4_gather

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]


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
    top = draft_vocab.logits_processor.get_top_tokens(head, hidden)
    assert torch.equal(draft_ids_to_target(top, d2t), logits.argmax(-1))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 rows need CUDA"
)
requires_dynamic_rows = pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(120)
        and has_flashinfer_gemv_bf16_fp4_gather()
    ),
    reason="Dynamic rows need SM12x and FlashInfer's gemv_bf16_fp4_gather",
)
TM_VOCAB, TM_HIDDEN, TM_STATIC = 4000, 512, 1333


def _token_map_drafter(
    monkeypatch, dynamic_rows: int, weight=None, *, disable_tp: bool = True
):
    """An MTP-like drafter sharing a tied BF16 lm_head, as in Qwen3.5, with
    random rows or this rank's rows of `weight`."""
    # Online methods read the model dtype from the current config.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    drafter = nn.Module()
    drafter.lm_head = VocabParallelEmbedding(TM_VOCAB, TM_HIDDEN, disable_tp=disable_tp)
    drafter.lm_head.to("cuda", torch.bfloat16)
    if weight is None:
        drafter.lm_head.weight.data.normal_(std=0.02)
    else:
        _load_rows(drafter.lm_head, weight)
    drafter.draft_vocab = DraftVocab(LogitsProcessor(TM_VOCAB), TM_VOCAB)
    spec = SimpleNamespace(
        draft_token_map_static_rows=TM_STATIC,
        draft_token_map_dynamic_rows=dynamic_rows or None,
    )
    load_draft_token_map(drafter, spec)
    return drafter


@requires_cuda
@pytest.mark.parametrize(
    "dynamic_rows", [0, pytest.param(16, marks=requires_dynamic_rows)]
)
def test_token_map_logits_match_reference(
    default_vllm_config, dist_init, monkeypatch, dynamic_rows
):
    """Drafted logits are the NVFP4 logits of the static and the picked rows,
    and -inf everywhere else."""
    drafter = _token_map_drafter(monkeypatch, dynamic_rows)
    head, draft_vocab = drafter.lm_head, drafter.draft_vocab
    hidden = torch.randn(64, TM_HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = draft_vocab.compute_logits(head, hidden).float()

    def nvfp4_logits(packed, scales, global_scale):
        rows = dequantize_to_dtype(
            packed, scales, global_scale, torch.float32, swizzle=False
        )
        return hidden.float() @ rows.t()

    expected = torch.full_like(out, float("-inf"))
    static = head.weight[:TM_STATIC]
    expected[:, :TM_STATIC] = nvfp4_logits(
        *_quantize_nvfp4(static, weight_amax(static))
    )
    if dynamic_rows:
        picked = out[:, TM_STATIC:].isfinite()
        assert torch.all(picked.sum(dim=-1) == dynamic_rows)
        reference = nvfp4_logits(*draft_vocab.nvfp4)
        expected[:, TM_STATIC:] = torch.where(picked, reference, float("-inf"))
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), out.argmax(-1))


@requires_cuda
@requires_dynamic_rows
def test_token_map_follows_target_head(default_vllm_config, dist_init, monkeypatch):
    """After a weight update of the lm_head, refreshing re-derives the static
    rows, the dynamic rows and the scorer, in place."""
    drafter = _token_map_drafter(monkeypatch, 16)
    head, draft_vocab = drafter.lm_head, drafter.draft_vocab
    ptrs = [t.data_ptr() for t in (*draft_vocab.nvfp4, draft_vocab.basis)]
    hidden = torch.randn(64, TM_HIDDEN, device="cuda", dtype=torch.bfloat16)

    def state(dv: DraftVocab) -> list[torch.Tensor]:
        static = dv.compute_logits(head, hidden)[:, :TM_STATIC]
        return [static, *dv.nvfp4, dv.basis, dv.scorer.weight]

    head.weight.data.normal_(std=0.02)
    fresh = DraftVocab(LogitsProcessor(TM_VOCAB), TM_VOCAB)
    fresh.load_token_map(head, TM_STATIC, 16)
    # The picks themselves may break score ties either way, so compare what
    # refresh derives rather than the drafted logits.
    assert not any(map(torch.equal, state(draft_vocab), state(fresh)))

    refresh_quantized_lm_heads(drafter)
    assert all(map(torch.equal, state(draft_vocab), state(fresh)))
    assert [t.data_ptr() for t in (*draft_vocab.nvfp4, draft_vocab.basis)] == ptrs


def _load_rows(head: VocabParallelEmbedding, weight: torch.Tensor) -> None:
    shard = head.shard_indices
    rows = weight[shard.org_vocab_start_index : shard.org_vocab_end_index]
    head.weight.data.zero_()
    head.weight.data[: rows.shape[0]] = rows


def _compare_tp_with_single_rank(dynamic_rows: int):
    @ray.remote(num_gpus=1, max_calls=1)
    def run(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
        torch.accelerator.set_device_index(rank)
        init_test_distributed_environment(
            tp_size, pp_size, rank, distributed_init_port, local_rank=rank
        )
        torch.manual_seed(0)
        with set_current_vllm_config(VllmConfig()):
            weight = 0.02 * torch.randn(
                TM_VOCAB, TM_HIDDEN, device="cuda", dtype=torch.bfloat16
            )
            single, sharded = (
                _token_map_drafter(
                    monkeypatch, dynamic_rows, weight, disable_tp=disable_tp
                )
                for disable_tp in (True, False)
            )
            assert sharded.lm_head.tp_size == tp_size
            hidden = torch.randn(64, TM_HIDDEN, device="cuda", dtype=torch.bfloat16)

            def check() -> None:
                expected, out = (
                    d.draft_vocab.compute_logits(d.lm_head, hidden)
                    for d in (single, sharded)
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                top = sharded.draft_vocab.get_top_tokens(sharded.lm_head, hidden)
                assert torch.equal(top, expected.argmax(-1))

            check()
            # A weight update of the lm_head reaches the rows each rank holds.
            weight = weight.flip(0)
            for drafter in (single, sharded):
                _load_rows(drafter.lm_head, weight)
                refresh_quantized_lm_heads(drafter)
            check()

    return run


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "dynamic_rows", [0, pytest.param(16, marks=requires_dynamic_rows)]
)
def test_token_map_tp_matches_single_rank(monkeypatch, dynamic_rows):
    """Split over TP ranks, the token map drafts the logits of one rank."""
    multi_process_parallel(
        monkeypatch, 2, 1, _compare_tp_with_single_rank(dynamic_rows)
    )


@pytest.mark.cpu_test
def test_draft_vocab_larger_than_target_raises(default_vllm_config):
    with pytest.raises(ValueError, match="larger than the target"):
        DraftVocab(LogitsProcessor(VOCAB + 1), VOCAB)


@pytest.mark.cpu_test
def test_dspark_reads_d2t_of_subclasses_that_skip_init(default_vllm_config):
    """Gemma4, MiniMax-M3 and Muse Spark DSpark drafters skip the parent
    __init__ and set only `draft_id_to_target_id`; inherited methods use it."""
    model = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = _head(len(DRAFT_IDS))
    model.logits_processor = LogitsProcessor(len(DRAFT_IDS))
    model.config = SimpleNamespace(vocab_size=VOCAB)
    model.draft_id_to_target_id = nn.Parameter(
        torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)), requires_grad=False
    )

    ids = torch.arange(len(DRAFT_IDS))
    assert model.map_draft_to_target(ids).tolist() == DRAFT_IDS
    hidden = _hidden()
    draft_logits = model.logits_processor(model.lm_head, hidden)
    logits = model.compute_logits(hidden)
    assert logits is not None
    assert torch.equal(logits[:, DRAFT_IDS], draft_logits)
