# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced logprob parity of Kimi-K3's specialized decode on and off
(``mono_logprob_parity`` describes the method), on the 8-layer dummy Kimi-K3
with random weights at TP8 / DCP8 and its DSpark draft (3 speculative tokens,
every decode step a verify step): every decode step of 8 requests runs the
kernels of layers 1-7 on every rank in the on arm. Both arms run eager, the
same requests through the same tokens, 64 logprobs a position (the 163840
vocabulary is too large for full distributions): KL over the tokens both arms
return, 73% of the mass, at least ``MIN_OVERLAP`` of the two sets shared.
Neither arm is bit-exact run to run (vLLM's own ops around the kernels are
not), so there is no exact eager-vs-graphs check. Under heavy GPU co-tenancy
the kernels' in-kernel waits can time out and fail-stop the engine: an
environment failure, rerun.

The weights (``RandomWeightLoader``): name-seeded (a TP shard's also by rank)
N(0, 1 / fan_in) matrices, MXFP4 experts with E8M0 codes 115-119, lm_head
scaled for an entropy of 3.0 nats over the top 64 (mean top-1 margin 0.85).
The routed experts quantize their activations to MXFP4, which turns a
rounding-level difference of their input into a ~16% one of their output: at
full scale this makes the random stack chaotic (off vs off decorrelates, KL up
to 5). Their up projection is scaled to ``ROUTED_UP`` of that, at which the
stack is not chaotic, so the routed experts' arithmetic is below this test's
sensitivity: test_e2e_parity checks it stage by stage.

Mutants of layer ``MUTANT_LAYER``'s weights while the kernels run (eager:
weights are read at each launch), worst KL over ``KL`` against each of the 11
off runs below: the MLP AttnRes query negated (``attn_res``, the test's)
40-45x, support overlap min 0.14-0.17; the KDA conv taps reversed 26-31x; the
MLA query norm negated (layer 3) 11-14x. Below 10x: rank 0's shared-expert
down projection zeroed 4.8-5.9x (overlap min 0.68-0.71, under
``MIN_OVERLAP``), A_log + 0.5 0.4-1.0x, and rank 0's routed experts (E8M0
codes of w2 + 1, or w13 of expert pairs swapped) 0.1-0.5x.
"""

import zlib
from contextlib import contextmanager

import pytest
import torch

from tests.models import mono_logprob_parity as P
from tests.models.kimi_k3.amd_specialized_decode.test_e2e_parity import (
    DRAFT,
    MODEL,
    _decode,
    _env,  # noqa: F401
)
from tests.models.kimi_k3.amd_specialized_decode.test_numerics import needs_gfx950
from tests.utils import multi_gpu_marks
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

# module-scoped engines: no process a test (multi_gpu_test)
pytestmark = [needs_gfx950, *multi_gpu_marks(num_gpus=8)]

LOAD_FORMAT = "kimi_k3_mono_random"
VOCAB = 163584  # the ids below the special tokens
REQUESTS, STEPS, NUM_SPEC, MAX_SEQS = 8, 16, 3, 2
LOGPROBS = 64
LM_HEAD_STD = 0.05
ROUTED_UP = 0.03
# Null over 11 off runs and 9 on runs (repeated in one engine, reversed and
# permuted submission order, relaunched, eager and FULL graphs): over their 91
# off-off and on-on pairs KL max 0.026, |dlogprob| max 0.41, spread over the
# positions (p99 within 0.8x of the max: no isolated routing near-ties); over
# the 99 on-off pairs and 6 test runs KL max 0.030, |dlogprob| max 0.43.
# Bounds about 3x the null. Top-1 flips counted at off margins over DLP: none
# in any of these pairs (at 0.1, 1-11 a pair). Support overlap min 0.88 over
# all of them.
KL, DLP = 0.08, 1.2
MIN_OVERLAP = 0.75
ENTROPY = (2.5, 4.0)
MIN_MARGIN = 0.5
MUTANT_LAYER = 4


@torch.no_grad()
def _fill(name: str, t: torch.Tensor, fan_in: int) -> None:
    """A random value for every weight, seeded by ``name``, in the ranges the
    checkpoint's formats hold."""
    if t.device.type == "meta":
        return
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype in (torch.uint8, torch.float8_e8m0fnu) and "scale" in name:
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(115, 120, u8.shape, generator=g, device=t.device))
    elif t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif not torch.is_floating_point(t):
        return
    elif "norm" in name:
        t.copy_(randn(0.1, 1.0))
    elif "lm_head.weight" in name:
        t.copy_(randn(LM_HEAD_STD))
    elif "embed_tokens.weight" in name:
        t.copy_(randn())
    elif "routed_expert_up_proj" in name:
        t.copy_(randn(ROUTED_UP * fan_in**-0.5))
    elif t.dim() > 1:
        t.copy_(randn(fan_in**-0.5))
    else:
        t.copy_(randn(0.1))


def _tp(module) -> tuple[bool, int]:
    """Whether the module's weights are a TP rank's shard, and the factor its
    local fan-in is short of the full one (row-parallel)."""
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    tp = (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
    if not isinstance(module, tp) or module.tp_size == 1:
        return False, 1
    return True, module.tp_size if isinstance(module, RowParallelLinear) else 1


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading, then ``_fill`` before post-load processing. A shard's
    seed includes the rank (alike shards would tie the vocabulary's logits
    eight ways); replicated weights, the router's included, are alike on
    every rank."""

    def load_weights(self, model, model_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        super().load_weights(model, model_config)
        rank = get_tensor_model_parallel_rank()
        for prefix, module in model.named_modules():
            sharded, split = _tp(module)
            tensors = [
                *module.named_parameters(recurse=False),
                *module.named_buffers(recurse=False),
            ]
            for name, t in tensors:
                if (
                    name in module._non_persistent_buffers_set
                    or t.dim() == 0  # KV cache scales
                    or getattr(t, "dummy_weight_value", None) is not None
                ):
                    continue
                if sharded or name.startswith(("w13", "w2")):
                    name = f"{name}@{rank}"
                fan_in = t[0].numel() * split if t.dim() > 1 else 1
                _fill(f"{prefix}.{name}", t, fan_in)


def _swap_pairs(t: torch.Tensor) -> None:
    t.copy_(t.view(t.size(0) // 2, 2, *t.shape[1:]).flip(1).view(t.shape))


# name: (on every rank, else rank 0's only; the layer's tensor; the bug)
MUTANTS = {
    "attn_res": (True, lambda m: m.mlp_res_proj.weight, torch.Tensor.neg_),
    "conv_taps": (
        True,
        lambda m: m.self_attn.conv1d.weight,
        lambda t: t.copy_(t.flip(-1)),
    ),
    "mla_qnorm": (
        True,
        lambda m: m.self_attn.mla_attn.q_a_layernorm.weight,
        torch.Tensor.neg_,
    ),
    "shared_drop": (
        False,
        lambda m: m.mlp.shared_experts.down_proj.weight,
        torch.Tensor.zero_,
    ),
    "a_log": (True, lambda m: m.self_attn.A_log, lambda t: t.add_(0.5)),
    "w2_scale": (
        False,
        lambda m: m.mlp.experts.routed_experts.w2_weight_scale,
        lambda t: t.view(torch.uint8).add_(1),
    ),
    "expert_swap": (
        False,
        lambda m: m.mlp.experts.routed_experts.w13_weight,
        lambda t: _swap_pairs(t.view(torch.uint8)),
    ),
}


@contextmanager
def _corrupt(mutant: str, layer, rank: int):
    every_rank, tensor, bug = MUTANTS[mutant]
    if not every_rank and rank != 0:
        yield
        return
    t = tensor(layer).data
    saved = t.clone()
    bug(t)
    try:
        yield
    finally:
        t.copy_(saved)


class Probe:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers."""

    def epoch(self) -> int:
        return int(_decode(self).runner.epoch[0])

    def mutate(self, mutant: str | None, layer_idx: int = MUTANT_LAYER) -> None:
        """Runs the kernels of layer ``layer_idx`` on its weights with
        ``mutant`` (None: restores); vLLM's ops read them unchanged."""
        for kernels in _decode(self).kernels.values():
            for kernel in kernels.values():
                kernel.__dict__.pop("launch", None)
                if mutant is None or kernel.layer.layer_idx != layer_idx:
                    continue

                def launch(*args, kernel=kernel, launch=kernel.launch):
                    with _corrupt(mutant, kernel.layer, kernel.runner.rank):
                        return launch(*args)

                kernel.launch = launch


def _engine(vllm_runner, mono: bool):
    return vllm_runner(
        MODEL,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        dcp_comm_backend="a2a",
        language_model_only=True,
        load_format=LOAD_FORMAT,
        worker_extension_cls=f"{__name__}.Probe",
        skip_tokenizer_init=True,
        enable_trace_replay=True,
        max_logprobs=LOGPROBS,
        kv_cache_dtype="fp8",
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_seqs=MAX_SEQS,
        gpu_memory_utilization=0.15,
        # plan() rejects the views extend_kv_cache re-binds (#66370)
        enable_extensible_kv_cache=False,
        enforce_eager=True,
        attention_config={"mla_prefill_backend": "ROCM_AITER_FA"},
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": NUM_SPEC,
            "attention_backend": "ROCM_AITER_MLA",
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        },
        additional_config={"decode_backend": "latency_specialized"} if mono else {},
    )


def _forced(llm, mono: bool) -> P.Forced:
    prompts, traces = P.forced_requests(VOCAB, REQUESTS, STEPS)
    if not mono:
        return P.run_forced(llm, prompts, traces, k=LOGPROBS)
    before = llm.collective_rpc("epoch")
    run = P.run_forced(llm, prompts, traces, k=LOGPROBS)
    steps = [b - a for a, b in zip(before, llm.collective_rpc("epoch"))]
    # every decode step (STEPS - 1 a request, MAX_SEQS requests a step) ran
    # the kernels on every rank
    assert steps == [REQUESTS * (STEPS - 1) // MAX_SEQS] * 8, steps
    return run


@pytest.fixture(scope="module")
def stock(vllm_runner):
    with _engine(vllm_runner, mono=False) as runner:
        return _forced(runner.llm, mono=False)


@pytest.fixture(scope="module")
def mono(vllm_runner):
    with _engine(vllm_runner, mono=True) as runner:
        yield runner.llm


def test_logprobs_match_stock(stock, mono):
    """Every forced decode position's top-64 logprobs with the specialized
    decode within the off-vs-off noise of vLLM's own."""
    m = P.compare(stock, _forced(mono, mono=True), clear_margin=DLP)
    print(P.summary(m))
    P.assert_informative(m, ENTROPY, MIN_MARGIN)
    P.assert_parity(m, KL, DLP, min_overlap=MIN_OVERLAP)


@pytest.mark.parametrize("mutant", ["attn_res"])
def test_mutant_fails_parity(stock, mono, mutant):
    """The bounds are not vacuous: one layer's kernels run on corrupted
    weights fail them by over 10x."""
    mono.collective_rpc("mutate", args=(mutant,))
    try:
        m = P.compare(stock, _forced(mono, mono=True))
    finally:
        mono.collective_rpc("mutate", args=(None,))
    print(P.summary(m))
    assert P.worst(m, "kl") >= 10 * KL, P.summary(m)
