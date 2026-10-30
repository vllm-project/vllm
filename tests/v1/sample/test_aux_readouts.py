# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests of `SamplingParams.aux_readouts` (Model Runner V2).

The reference readout is a 3-way "decision head" made of three rows of OPT's
output embedding. Its value at a generated token is therefore the raw logits of
those three tokens, which the engine also returns through `logprob_token_ids`:
a reference computation that needs no second model.
"""

import pytest
import torch

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.opt import OPTForCausalLM

# The module engine runs in-process (to see the test-local architecture), so the
# per-test distributed cleanup must not run between its tests; `llm` cleans up.
pytestmark = pytest.mark.skip_global_cleanup

MODEL = "facebook/opt-125m"
LABEL_TOKENS = [4, 6, 8]
ARCH = "AuxReadoutOPTForCausalLM"
FP16_ULPS_4 = dict(rtol=4 * 2**-10, atol=4 * 2**-10)
PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "In a hole in the ground there lived",
]


class AuxReadoutOPTForCausalLM(OPTForCausalLM):
    supports_aux_readout = True
    aux_readout_dims = {"decision": 3, "first_label": 1}

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.register_buffer("label_ids", torch.tensor(LABEL_TOKENS), persistent=False)
        self.readout_rows = 0

    def compute_aux_readouts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.readout_rows += hidden_states.shape[0]
        head = self.lm_head.weight.index_select(0, self.label_ids).float()
        decision = hidden_states.float() @ head.T
        return torch.cat([decision, decision[:, :1]], dim=1)


def _params(**kw) -> SamplingParams:
    return SamplingParams(temperature=0.0, ignore_eos=True, **kw)


def _same_generation(a, b):
    for x, y in zip(a, b):
        x, y = x.outputs[0], y.outputs[0]
        assert x.token_ids == y.token_ids
        for step, (lx, ly) in enumerate(zip(x.logprobs, y.logprobs)):
            t = x.token_ids[step]
            torch.testing.assert_close(
                torch.tensor(lx[t].logprob), torch.tensor(ly[t].logprob), **FP16_ULPS_4
            )


@pytest.fixture(scope="module")
def llm():
    with pytest.MonkeyPatch.context() as mp:
        # In-process engine so the test-local architecture is visible to it.
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        ModelRegistry.register_model(ARCH, AuxReadoutOPTForCausalLM)
        llm = LLM(
            MODEL,
            hf_overrides={"architectures": [ARCH]},
            enable_aux_readouts=True,
            logprobs_mode="raw_logits",
            gpu_memory_utilization=0.3,
            max_model_len=128,
            # Small step budget so that longer prompts are prefilled in chunks.
            max_num_batched_tokens=32,
            max_num_seqs=8,
            seed=0,
        )
        try:
            yield llm
        finally:
            del llm
            torch.accelerator.empty_cache()
            cleanup_dist_env_and_memory()


def _readout_rows(llm) -> int:
    return llm.apply_model(lambda m: m.readout_rows)[0]


def test_readout_matches_reference(llm):
    params = _params(
        max_tokens=3,
        aux_readouts=["decision", "first_label"],
        logprob_token_ids=LABEL_TOKENS,
    )
    for output in llm.generate(PROMPTS, params):
        completion = output.outputs[0]
        readouts = completion.aux_readouts
        assert set(readouts) == {"decision", "first_label"}
        decision = readouts["decision"]
        assert decision.shape == (len(completion.token_ids), 3)
        assert decision.dtype == torch.float32 and decision.device.type == "cpu"
        torch.testing.assert_close(readouts["first_label"][:, 0], decision[:, 0])
        for step, scores in enumerate(completion.logprobs):
            engine = torch.tensor([scores[t].logprob for t in LABEL_TOKENS])
            torch.testing.assert_close(decision[step], engine, rtol=1e-2, atol=5e-2)
    (one,) = llm.generate(PROMPTS[:1], _params(max_tokens=1, aux_readouts=["decision"]))
    assert one.outputs[0].aux_readouts["decision"].shape == (1, 3)


def test_no_readout_work_unless_asked(llm):
    before = _readout_rows(llm)
    llm.generate(PROMPTS, _params(max_tokens=4))
    assert _readout_rows(llm) == before
    llm.generate(PROMPTS[:2], _params(max_tokens=2, aux_readouts=["decision"]))
    assert _readout_rows(llm) == before + 4


def test_asking_changes_no_generation(llm):
    greedy = dict(max_tokens=6, logprobs=3)
    asked, plain = _params(**greedy, aux_readouts=["decision"]), _params(**greedy)
    mixed = llm.generate(PROMPTS, [asked, plain, asked])
    none = llm.generate(PROMPTS, [plain, plain, plain])
    _same_generation(mixed, none)
    assert mixed[1].outputs[0].aux_readouts is None
    assert all(o.outputs[0].aux_readouts is None for o in none)


def test_readout_on_prefix_cache_hit(llm):
    params = _params(
        max_tokens=2, aux_readouts=["decision"], logprob_token_ids=LABEL_TOKENS
    )
    prompt = PROMPTS[2] * 4
    llm.generate([prompt], params)
    (hit,) = llm.generate([prompt], params)
    assert hit.num_cached_tokens > 0
    completion = hit.outputs[0]
    for step, scores in enumerate(completion.logprobs):
        engine = torch.tensor([scores[t].logprob for t in LABEL_TOKENS])
        torch.testing.assert_close(
            completion.aux_readouts["decision"][step], engine, rtol=1e-2, atol=5e-2
        )


def test_readout_with_chunked_prefill(llm):
    # Prefilled over several steps; the rows of the steps that do not finish the
    # prompt are discarded with their logits.
    prompt = "Count slowly: " + " ".join(str(i) for i in range(80))
    params = _params(
        max_tokens=3, aux_readouts=["decision"], logprob_token_ids=LABEL_TOKENS
    )
    (out,) = llm.generate([prompt], params)
    assert len(out.prompt_token_ids) > 2 * 32
    completion = out.outputs[0]
    assert completion.aux_readouts["decision"].shape == (3, 3)
    for step, scores in enumerate(completion.logprobs):
        engine = torch.tensor([scores[t].logprob for t in LABEL_TOKENS])
        torch.testing.assert_close(
            completion.aux_readouts["decision"][step], engine, rtol=1e-2, atol=5e-2
        )


def test_abort_releases_readout_state(llm):
    engine = llm.llm_engine
    for i, prompt in enumerate(PROMPTS):
        engine.add_request(
            f"abort-{i}", prompt, _params(max_tokens=8, aux_readouts=["decision"])
        )
    engine.step()
    engine.abort_request(["abort-1"])
    finished = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = out
    for rid in ("abort-0", "abort-2"):
        assert finished[rid].outputs[0].aux_readouts["decision"].shape == (8, 3)
    if "abort-1" in finished:  # an abort may surface as a final output
        assert len(finished["abort-1"].outputs[0].token_ids) < 8
    # Finished ids reach the runner with the next step's scheduler output.
    llm.generate(PROMPTS[:1], _params(max_tokens=1))
    (pending,) = llm.collective_rpc(
        lambda self: len(self.model_runner.aux_readout_req_ids)
    )
    assert pending == 0


def test_feature_unused_matches_engine_without_flag(llm):
    # Prompts not seen by the module engine, so neither side hits a prefix cache.
    prompts = ["A quick brown fox", "def fibonacci(n):", "Once upon a time"]
    params = _params(max_tokens=6, logprobs=3)
    with_flag = llm.generate(prompts, params)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        stock = LLM(
            MODEL,
            logprobs_mode="raw_logits",
            gpu_memory_utilization=0.3,
            max_model_len=128,
            max_num_batched_tokens=32,
            max_num_seqs=8,
            seed=0,
        )
        try:
            _same_generation(with_flag, stock.generate(prompts, params))
            with pytest.raises(VLLMValidationError, match="enable_aux_readouts"):
                stock.generate(
                    PROMPTS[:1], _params(max_tokens=1, aux_readouts=["decision"])
                )
        finally:
            del stock
            torch.accelerator.empty_cache()


@pytest.mark.parametrize(
    "names", [[], ["nope"], ["decision", "decision"]], ids=["empty", "unknown", "dup"]
)
def test_rejects_bad_names(llm, names):
    with pytest.raises(VLLMValidationError, match="aux_readouts"):
        llm.generate(PROMPTS[:1], _params(max_tokens=1, aux_readouts=names))
