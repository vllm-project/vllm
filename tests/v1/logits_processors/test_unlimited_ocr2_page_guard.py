# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Unlimited-OCR-2 PageCountGuard logits processor."""

import numpy as np
import pytest
import torch

from vllm.model_executor.models.unlimited_ocr2 import (
    PageCountGuard,
    PageCountGuardLogitsProcessor,
)
from vllm.sampling_params import SamplingParams

PAGE_ID = 128827
EOS_ID = 1
VOCAB_SIZE = 129280

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Model Runner V2 processors need CUDA buffers",
)


def _logits() -> torch.Tensor:
    return torch.zeros(VOCAB_SIZE)


def _step(
    guard: PageCountGuard,
    prompt_ids: list[int],
    output_ids: list[int],
) -> torch.Tensor:
    if guard.expected_pages < 0:
        guard.seed_expected_pages(prompt_ids)
    logits = _logits()
    guard.step(output_ids, logits)
    return logits


class TestPageCountGuardCore:
    def test_eos_masked_until_all_pages_done(self):
        prompt = [PAGE_ID] * 2 + [100]
        guard = PageCountGuard(PAGE_ID, EOS_ID)
        logits = _step(guard, prompt, [10, 11])
        assert logits[EOS_ID].item() == -float("inf")
        # Only EOS is masked; everything else is untouched.
        assert torch.isfinite(logits).sum().item() == VOCAB_SIZE - 1

        logits = _step(guard, prompt, [10, 11, PAGE_ID])
        assert logits[EOS_ID].item() == -float("inf")

        logits = _step(guard, prompt, [10, 11, PAGE_ID, 12, PAGE_ID])
        assert torch.isfinite(logits).all()

    def test_page_cap_forces_page_token_or_eos(self):
        prompt = [PAGE_ID] * 2 + [100]
        guard = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=3)
        # Page cap hit while pages remain: only <PAGE> stays finite.
        logits = _step(guard, prompt, [10, 11, 12])
        assert torch.isfinite(logits).nonzero().flatten().tolist() == [PAGE_ID]

        # Page cap hit on the last page: only EOS stays finite.
        guard = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=3)
        logits = _step(guard, [PAGE_ID, 100], [PAGE_ID, 11, 12, 13])
        assert torch.isfinite(logits).nonzero().flatten().tolist() == [EOS_ID]

    def test_extra_page_markers_force_eos(self):
        guard = PageCountGuard(PAGE_ID, EOS_ID)
        # One marker completes the single expected page; the second is extra.
        logits = _step(guard, [PAGE_ID, 100], [10, PAGE_ID, 11, PAGE_ID])
        assert torch.isfinite(logits).nonzero().flatten().tolist() == [EOS_ID]

    def test_no_prompt_markers_is_passthrough(self):
        guard = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=1)
        logits = _step(guard, [100, 101], [10, 11, 12])
        assert torch.isfinite(logits).all()

    def test_nonpositive_page_cap_disables_only_the_cap(self):
        prompt = [PAGE_ID] * 2 + [100]
        guard = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=0)
        logits = _step(guard, prompt, [10] * 100)
        assert logits[EOS_ID].item() == -float("inf")
        assert torch.isfinite(logits).sum().item() == VOCAB_SIZE - 1

    def test_incremental_scanning_matches_one_shot(self):
        prompt = [PAGE_ID] * 3 + [100]
        output_ids = [10, PAGE_ID, 11, 12, PAGE_ID, 13]

        one_shot = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=5)
        expected = _step(one_shot, prompt, output_ids)

        incremental = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=5)
        running: list[int] = []
        for token_id in output_ids:
            running.append(token_id)
            actual = _step(incremental, prompt, running)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "extra_args",
    [
        {"page_token_id": -1},
        {"page_token_id": "<PAGE>"},
        {"eos_token_id": 1.5},
        {"max_page_tokens": "20000"},
        {"page_token_id": 2**63},
        {"eos_token_id": 2**63},
        {"max_page_tokens": 2**63},
        {"eos_token_id": None},
        {"max_page_tokens": None},
    ],
)
def test_validate_params_rejects_bad_types(extra_args):
    params = SamplingParams(extra_args=extra_args)
    with pytest.raises(ValueError):
        PageCountGuardLogitsProcessor.validate_params(params)


def _make_v2_req_state(max_num_reqs: int = 1, max_len: int = 64):
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsProcRequestState,
    )

    device = torch.device("cuda:0")
    return LogitsProcRequestState(
        device=device,
        max_num_reqs=max_num_reqs,
        vocab_size=VOCAB_SIZE,
        all_token_ids=StagedWriteTensor(
            (max_num_reqs, max_len), dtype=torch.int32, device=device
        ),
        prompt_len=UvaBackedTensor(max_num_reqs, dtype=torch.int32),
        prefill_len=UvaBackedTensor(max_num_reqs, dtype=torch.int32),
        total_len=StagedWriteTensor(max_num_reqs, dtype=torch.int32, device=device),
    )


def _make_v2_ctx(num_reqs: int, seq_lens: list[int]):
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext

    device = torch.device("cuda:0")
    idx = torch.arange(num_reqs, device=device)
    return LogitsContext(
        expanded_idx_mapping=idx,
        idx_mapping=idx,
        idx_mapping_np=np.arange(num_reqs),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.long, device=device),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        pos=torch.tensor(seq_lens, dtype=torch.long, device=device) - 1,
        seq_lens_upper_bound_np=np.array(seq_lens, dtype=np.int32),
    )


def _make_v2_guard(model: str = "baidu/Unlimited-OCR-2"):
    from vllm.engine.arg_utils import EngineArgs

    vllm_config = EngineArgs(
        model=model, enable_chunked_prefill=False
    ).create_engine_config()
    return PageCountGuardLogitsProcessor(vllm_config, _make_v2_req_state())


def _commit(processor, prompt_ids: list[int], output_ids: list[int]) -> None:
    ids = prompt_ids + output_ids
    processor.prompt_len_np[0] = len(prompt_ids)
    processor._prompt_len.copy_to_uva()
    processor.all_token_ids.gpu[0, : len(ids)] = torch.tensor(
        ids, dtype=torch.int32, device=processor.all_token_ids.gpu.device
    )


def _apply_v2(processor, prompt_ids, output_ids) -> torch.Tensor:
    _commit(processor, prompt_ids, output_ids)
    processor.apply_staged_writes()
    ctx = _make_v2_ctx(1, [len(prompt_ids) + len(output_ids)])
    return processor.apply(_logits().unsqueeze(0).to("cuda"), ctx)[0].cpu()


@_requires_cuda
class TestPageCountGuardV2:
    @pytest.mark.parametrize("page_cap", [0, 3])
    def test_matches_core_step_by_step(self, page_cap):
        prompt_ids = [PAGE_ID, 100, PAGE_ID, 100]
        output_ids = [10, 11, PAGE_ID, 12, 13, 14, PAGE_ID, 15, PAGE_ID]
        core = PageCountGuard(PAGE_ID, EOS_ID, page_cap)

        v2 = _make_v2_guard()
        params = SamplingParams(
            extra_args={
                "page_token_id": PAGE_ID,
                "eos_token_id": EOS_ID,
                "max_page_tokens": page_cap,
            }
        )
        assert v2.add_request(0, params)

        # Prefill chunks must never count prompt markers as generated output.
        _commit(v2, prompt_ids, [])
        v2.apply_staged_writes()
        for computed in [1, 3, len(prompt_ids)]:
            ctx = _make_v2_ctx(1, [computed])
            logits = v2.apply(_logits().unsqueeze(0).to("cuda"), ctx)[0].cpu()
            assert torch.equal(logits, _step(core, prompt_ids, []))

        running: list[int] = []
        for token_id in output_ids:
            running.append(token_id)
            expected = _step(core, prompt_ids, running)
            actual = _apply_v2(v2, prompt_ids, running)
            assert torch.equal(actual, expected)

    def test_inactive_without_page_token(self):
        v2 = _make_v2_guard("facebook/opt-125m")
        assert not v2.add_request(0, SamplingParams())
        logits = _logits().unsqueeze(0).to("cuda")
        assert v2.apply(logits, _make_v2_ctx(1, [4])) is logits

    def test_no_prompt_markers_is_passthrough(self):
        v2 = _make_v2_guard()
        assert v2.add_request(0, SamplingParams())
        logits = _apply_v2(v2, [100, 101], [10, 11, 12])
        assert torch.isfinite(logits).all()

    def test_defaults_resolved_from_hf_config(self):
        v2 = _make_v2_guard()
        assert v2.add_request(0, SamplingParams())
        guard = v2._guards[0]
        assert guard.page_token_id == PAGE_ID
        assert guard.eos_token_id == EOS_ID
        assert guard.max_page_tokens == 20000

    def test_extra_args_override_defaults(self):
        v2 = _make_v2_guard()
        params = SamplingParams(
            extra_args={
                "page_token_id": 5,
                "eos_token_id": 6,
                "max_page_tokens": 42,
            }
        )
        assert v2.add_request(0, params)
        guard = v2._guards[0]
        assert (guard.page_token_id, guard.eos_token_id, guard.max_page_tokens) == (
            5,
            6,
            42,
        )

    def test_loader_accepts_fqcn(self):
        from vllm.v1.worker.gpu.sample.logits_processor.loader import (
            _load_v2_logitsprocs_by_fqcns,
        )

        classes = _load_v2_logitsprocs_by_fqcns(
            [
                (
                    "vllm.model_executor.models.unlimited_ocr2:"
                    "PageCountGuardLogitsProcessor"
                )
            ]
        )
        assert classes == [PageCountGuardLogitsProcessor]

    def test_apply_has_no_host_reads(self):
        """Both initialization and incremental decoding must stay asynchronous."""
        v2 = _make_v2_guard()
        assert v2.add_request(0, SamplingParams())
        for output in ([], [PAGE_ID, 10], [PAGE_ID, 10, PAGE_ID]):
            _commit(v2, [PAGE_ID, 100], output)
            v2.apply_staged_writes()
            ctx = _make_v2_ctx(1, [2 + len(output)])
            logits = _logits().unsqueeze(0).cuda()
            previous = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                v2.apply(logits, ctx)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            expected = _step(PageCountGuard(PAGE_ID, EOS_ID), [PAGE_ID, 100], output)
            assert torch.equal(logits[0].cpu(), expected)

    def test_batched_reordering_and_slot_reuse(self):
        """State follows slots across row changes, resumption and new occupants."""
        from dataclasses import replace

        from vllm.engine.arg_utils import EngineArgs

        config = EngineArgs(
            model="baidu/Unlimited-OCR-2", enable_chunked_prefill=False
        ).create_engine_config()
        req_state = _make_v2_req_state(max_num_reqs=3)
        v2 = PageCountGuardLogitsProcessor(config, req_state)
        params = SamplingParams(extra_args={"max_page_tokens": 3})
        prompts = [[PAGE_ID, 100], [PAGE_ID, PAGE_ID, 100], [100]]
        outputs = [[PAGE_ID, 10], [10, 11, 12], [10, 11, 12]]
        inactive = set()
        for slot in range(3):
            assert v2.add_request(slot, params)

        def check(order):
            for slot in range(3):
                ids = prompts[slot] + outputs[slot]
                req_state.prompt_len.np[slot] = len(prompts[slot])
                req_state.all_token_ids.gpu[slot, : len(ids)] = torch.tensor(
                    ids, device="cuda"
                )
            req_state.prompt_len.copy_to_uva()
            v2.apply_staged_writes()
            ctx = _make_v2_ctx(3, [len(prompts[s]) + len(outputs[s]) for s in order])
            mapping = torch.tensor(order, device="cuda")
            ctx = replace(
                ctx,
                idx_mapping=mapping,
                expanded_idx_mapping=mapping,
                idx_mapping_np=np.array(order),
            )
            # Preserve existing masks and exercise a noncontiguous logits view.
            logits = torch.arange(
                3 * VOCAB_SIZE * 2, dtype=torch.float32, device="cuda"
            ).reshape(3, -1)[:, ::2]
            logits[:, PAGE_ID] = -float("inf")
            expected = logits.cpu()
            for row, slot in enumerate(order):
                if slot in inactive:
                    continue
                guard = PageCountGuard(PAGE_ID, EOS_ID, 3)
                guard.seed_expected_pages(prompts[slot])
                guard.step(outputs[slot], expected[row])
            actual = v2.apply(logits, ctx).cpu()
            assert torch.equal(actual, expected)

        check([0, 1, 2])
        outputs[0].extend([11, 12])
        outputs[1].extend([PAGE_ID, 10, PAGE_ID, 11, 12, 13])
        check([2, 0, 1])
        # A resumed request arrives with already generated output.
        assert v2.add_request(1, params)
        # Slot 0 is reused by a new request with a different prompt page count.
        assert v2.add_request(0, params)
        prompts[0], outputs[0] = [PAGE_ID] * 3, [PAGE_ID, 10]
        check([1, 2, 0])
        assert not v2.add_request(1, SamplingParams(extra_args={"page_token_id": None}))
        inactive.add(1)
        check([2, 1, 0])


@_requires_cuda
@pytest.mark.parametrize("name", ["eos_token_id", "page_token_id"])
def test_out_of_vocab_guard_does_not_modify_logits_or_adjacent_storage(name):
    v2 = _make_v2_guard()
    assert v2.add_request(0, SamplingParams(extra_args={name: VOCAB_SIZE}))
    _commit(v2, [PAGE_ID, 100], [])
    v2.apply_staged_writes()
    # A sentinel row makes an out-of-range store observable without an illegal
    # device address that would poison the CUDA context for the test process.
    storage = torch.zeros(2, VOCAB_SIZE, device="cuda")
    v2.apply(storage[:1], _make_v2_ctx(1, [2]))
    assert torch.equal(storage, torch.zeros_like(storage))


@_requires_cuda
@pytest.mark.parametrize(
    "output, forced", [([10, 11, 12], PAGE_ID), ([PAGE_ID, PAGE_ID], EOS_ID)]
)
def test_masked_forced_token_preserves_other_sampling_constraints(output, forced):
    """A prior min_tokens/bad_words/grammar mask must not make the row all -inf."""
    guard = PageCountGuard(PAGE_ID, EOS_ID, max_page_tokens=3)
    prompt = [PAGE_ID, 100]
    guard.seed_expected_pages(prompt)
    logits = _logits()
    logits[forced] = -float("inf")
    original = logits.clone()
    guard.step(output, logits)
    assert torch.equal(logits, original)

    v2 = _make_v2_guard()
    assert v2.add_request(0, SamplingParams(extra_args={"max_page_tokens": 3}))
    _commit(v2, prompt, output)
    v2.apply_staged_writes()
    actual = original.unsqueeze(0).cuda()
    v2.apply(actual, _make_v2_ctx(1, [len(prompt) + len(output)]))
    assert torch.equal(actual[0].cpu(), original)
