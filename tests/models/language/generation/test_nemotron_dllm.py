# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Nemotron Ministral masked block-diffusion LM
(``MinistralDiffEncoderModel`` / ``nemotron_dllm``).

Tiers:

* ``test_config_registration`` / ``test_arch_in_model_registry`` — CPU-only, no
  weights. Verify the config exposes ``canvas_length`` (vLLM's diffusion
  detection key) and the architecture resolves in the model registry.
* ``test_masked_step_sampling_semantics`` — GPU, no weights. Drives
  ``_compiled_masked_step`` on synthetic logits: greedy at temperature 0,
  Gumbel-max sampling at temperature 1, and at-unmask logprob capture (each
  emitted logprob reflects the denoise step its token was unmasked at, which
  RL importance correction depends on).
* ``test_logits_parity`` — GPU + weights. vLLM's per-position logits over a
  masked canvas must match the HF reference (argmax-exact; bf16-ULP diffs).
* ``test_greedy_generation`` — GPU + weights. End-to-end greedy block-diffusion
  generation must produce the reference answer.

The weights tests need the model locally (not on the public Hub): set
``NEMOTRON_DLM_MODEL_PATH`` or use the default. They skip when weights/GPU absent.
"""
import os

import pytest
import torch

MODEL_PATH = os.environ.get(
    "NEMOTRON_DLM_MODEL_PATH",
    "/linnanw/justGRPO/asset/Nemotron-Labs-Diffusion-3B",
)
MASK_TOKEN_ID = 100
CANVAS_LENGTH = 32
MAX_MODEL_LEN = 2048


def _weights_available() -> bool:
    return os.path.isfile(os.path.join(MODEL_PATH, "config.json"))


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason=f"Nemotron-DLM weights not found at {MODEL_PATH}",
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


# ---------------------------------------------------------------------------
# CPU-only: registration & diffusion detection
# ---------------------------------------------------------------------------
def test_config_registration():
    from vllm.transformers_utils.configs import MinistralDLMConfig

    cfg = MinistralDLMConfig(block_size=32, mask_token_id=100)
    # canvas_length is the field ModelConfig.is_diffusion keys off of.
    assert cfg.canvas_length == 32
    assert cfg.mask_token_id == 100


def test_arch_in_model_registry():
    from vllm.model_executor.models.registry import ModelRegistry

    assert "MinistralDiffEncoderModel" in ModelRegistry.get_supported_archs()


def test_diffusion_config_temperature():
    from vllm.config.diffusion import DiffusionConfig

    # Engine-wide denoising temperature (per-request temperature is rejected
    # for diffusion models); unset -> model default (greedy for masked
    # diffusion). Negative values are invalid.
    assert DiffusionConfig(canvas_length=32).temperature is None
    assert DiffusionConfig(canvas_length=32, temperature=1.0).temperature == 1.0
    with pytest.raises(Exception):
        DiffusionConfig(canvas_length=32, temperature=-0.5)


# ---------------------------------------------------------------------------
# GPU, no weights: compiled masked-diffusion step semantics
# ---------------------------------------------------------------------------
@requires_gpu
def test_masked_step_sampling_semantics():
    from vllm.model_executor.models.nemotron_dllm import _compiled_masked_step

    device = "cuda"
    CL, V, MAX_REQS = 4, 12, 2
    mask_id = 10
    max_steps = 2  # even schedule: k = ceil(4/2) = 2 unmasked per step

    def fresh_state():
        return {
            "canvas": torch.full(
                (MAX_REQS, CL), mask_id, dtype=torch.int64, device=device
            ),
            "step": torch.zeros(MAX_REQS, dtype=torch.int32, device=device),
            "phase": torch.zeros(MAX_REQS, dtype=torch.bool, device=device),
            "lp": torch.zeros(MAX_REQS, CL, dtype=torch.float32, device=device),
            "rk": torch.zeros(MAX_REQS, CL, dtype=torch.int32, device=device),
        }

    def run_step(st, logits, temperature, leftmost=False, steps=max_steps,
                 req_temp=1.0):
        sampled = torch.zeros(1, CL, dtype=torch.int32, device=device)
        num_sampled = torch.zeros(1, dtype=torch.int32, device=device)
        draft = torch.zeros(MAX_REQS, CL, dtype=torch.int64, device=device)
        zero = torch.zeros(1, dtype=torch.int64, device=device)
        _compiled_masked_step(
            logits.reshape(-1, V),
            zero,  # decode_slots
            zero,  # decode_idx
            zero,  # all_slots
            torch.full((1,), CL, dtype=torch.int64, device=device),
            torch.full((1,), req_temp, dtype=torch.float32, device=device),
            st["canvas"], st["step"], st["phase"], st["lp"], st["rk"],
            sampled, num_sampled, draft,
            max_denoising_steps=steps, mask_token_id=mask_id, CL=CL,
            temperature=temperature, capture_logprobs=True,
            leftmost=leftmost,
        )

    torch.manual_seed(0)
    logits1 = torch.randn(1, CL, V, device=device) * 4.0
    logits1[..., mask_id] = -100.0  # never pick the mask token

    # Greedy (T=0): unmasked tokens == argmax, captured logprob/rank match
    # the step's log_softmax and top-1 rank.
    st = fresh_state()
    run_step(st, logits1, 0.0)
    lp1 = logits1[0].float().log_softmax(-1)
    x0 = logits1[0].argmax(-1)
    step1 = st["canvas"][0] != mask_id
    assert int(step1.sum()) == 2
    assert torch.equal(st["canvas"][0][step1], x0[step1])
    want = lp1.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(st["lp"][0][step1], want[step1], atol=1e-5)
    assert (st["rk"][0][step1] == 1).all()

    # At-unmask persistence: a second step with DIFFERENT logits must not
    # rewrite the step-1 captures, and step-2 captures reflect step-2 logits.
    lp_after1 = st["lp"][0].clone()
    torch.manual_seed(1)
    logits2 = torch.randn(1, CL, V, device=device) * 4.0
    logits2[..., mask_id] = -100.0
    run_step(st, logits2, 0.0)
    assert not (st["canvas"][0] == mask_id).any()
    assert torch.allclose(st["lp"][0][step1], lp_after1[step1])
    step2 = ~step1
    lp2 = logits2[0].float().log_softmax(-1)
    want2 = lp2.gather(-1, st["canvas"][0].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(st["lp"][0][step2], want2[step2], atol=1e-5)

    # Sampling (T=1): draws differ across seeds; the captured logprob/rank
    # match the SAMPLED token under this step's distribution.
    draws = []
    for seed in range(3):
        torch.manual_seed(100 + seed)
        st = fresh_state()
        run_step(st, logits1 * 0.25, 1.0)  # flatter -> diverse draws
        unm = st["canvas"][0] != mask_id
        toks = st["canvas"][0]
        lp_ref = (logits1[0] * 0.25).float().log_softmax(-1)
        got = lp_ref.gather(-1, toks.clamp(0, V - 1).unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(st["lp"][0][unm], got[unm], atol=1e-5)
        ranks = (lp_ref >= got.unsqueeze(-1)).sum(-1).int()
        assert torch.equal(st["rk"][0][unm], ranks[unm])
        draws.append(tuple(toks[unm].tolist()))
    assert len(set(draws)) >= 2, f"3 seeds gave identical draws: {draws}"

    # Per-request greedy on a sampling engine (temperature=0 requests, e.g.
    # greedy validation during RL): zero Gumbel noise for that row -> argmax.
    st = fresh_state()
    run_step(st, logits1, 1.0, req_temp=0.0)
    unm = st["canvas"][0] != mask_id
    assert torch.equal(st["canvas"][0][unm], x0[unm])

    # Leftmost selection: reveal order is strictly position order even when
    # a rightmost position is by far the most confident (leftmost-reveal RL
    # rollouts need this to match the trainer's recompute factorization).
    spiked = logits1.clone()
    spiked[0, CL - 1, 5] = 50.0
    st = fresh_state()
    order = []
    for _ in range(CL):  # steps=CL -> k=1 per step
        before = (st["canvas"][0] != mask_id).clone()
        run_step(st, spiked, 0.0, leftmost=True, steps=CL)
        new = ((st["canvas"][0] != mask_id) & ~before).nonzero()
        order.extend(new.flatten().tolist())
    assert order == list(range(CL)), f"leftmost reveal order: {order}"


# ---------------------------------------------------------------------------
# HF reference (ground truth). transformers>=5 dropped two symbols the model's
# custom code imports; shim them to the shipped behavior.
# ---------------------------------------------------------------------------
def _install_tf_compat():
    import transformers.masking_utils as mu
    import transformers.utils.generic as g

    if not hasattr(g, "check_model_inputs"):
        def check_model_inputs(func=None, **_):
            return func if func is not None else (lambda f: f)

        g.check_model_inputs = check_model_inputs
    if not hasattr(mu, "sdpa_mask_older_torch"):
        mu.sdpa_mask_older_torch = mu.sdpa_mask


def _load_hf(dtype=torch.bfloat16, device="cuda"):
    _install_tf_compat()
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    return tok, model


def _prompt_ids(tok, device="cuda"):
    msgs = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
    return ids.to(device)


def _vllm_manual_forward(obj, token_ids, causal_flag):
    """Single forward of the loaded vLLM model over ``token_ids`` as one request
    with per-request ``causal=causal_flag``, driving the GPU model runner
    (KV-cache slots, attn metadata, forward context). Returns fp32 logits
    ``[len(token_ids), vocab]`` on CPU. Runs under ``collective_rpc``.
    """
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.gpu.attn_utils import (
        build_attn_metadata,
        build_slot_mappings_by_layer,
    )

    worker = getattr(obj, "worker", obj)
    runner = worker.model_runner
    device = runner.device
    kv_cache_config = runner.kv_cache_config
    assert len(kv_cache_config.kv_cache_groups) == 1
    block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

    length = len(token_ids)
    n_blocks = (length + block_size - 1) // block_size
    # Block 0 is the null block; use fresh blocks from 1.
    block_ids = torch.arange(1, n_blocks + 1, dtype=torch.int32, device=device)
    block_table = block_ids.unsqueeze(0)
    pos = torch.arange(length, dtype=torch.int64, device=device)
    slot_mapping = (
        block_ids[pos // block_size].to(torch.int64) * block_size
        + pos % block_size
    )
    slot_mappings = slot_mapping.unsqueeze(0)

    query_start_loc = torch.tensor([0, length], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, length], dtype=torch.int32)
    seq_lens = torch.tensor([length], dtype=torch.int32, device=device)

    causal = (
        True
        if (isinstance(causal_flag, bool) and causal_flag)
        else torch.tensor([bool(causal_flag)], dtype=torch.bool, device=device)
    )

    attn_metadata = build_attn_metadata(
        attn_groups=runner.attn_groups,
        num_reqs=1,
        num_tokens=length,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=length,
        seq_lens=seq_lens,
        max_seq_len=MAX_MODEL_LEN,
        block_tables=[block_table],
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        causal=causal,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    input_ids = torch.tensor(token_ids, dtype=torch.int32, device=device)
    with torch.inference_mode():
        with set_forward_context(
            attn_metadata,
            runner.vllm_config,
            num_tokens=length,
            slot_mapping=slot_mappings_by_layer,
        ):
            hidden = runner.model(input_ids=input_ids, positions=pos)
        logits = runner.model.compute_logits(hidden)
    return logits.float().cpu()


# ---------------------------------------------------------------------------
# GPU + weights: logits parity vs HF over a masked canvas (bidirectional)
# ---------------------------------------------------------------------------
@requires_gpu
@requires_weights
def test_logits_parity(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM

    tok, hf = _load_hf()
    prompt = _prompt_ids(tok)
    canvas = torch.full(
        (1, CANVAS_LENGTH), MASK_TOKEN_ID, dtype=prompt.dtype, device=prompt.device
    )
    seq = torch.cat([prompt, canvas], dim=1)
    token_ids = seq[0].tolist()

    with torch.no_grad():
        hf_logits = hf(seq).logits[0].float().cpu()  # fully bidirectional

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    # causal_flag=False -> bidirectional canvas (matches HF single forward).
    vllm_logits = llm.llm_engine.collective_rpc(
        _vllm_manual_forward, args=(token_ids, False)
    )[0]

    canvas_slice = slice(prompt.shape[1], seq.shape[1])
    hf_c = hf_logits[canvas_slice]
    vl_c = vllm_logits[canvas_slice]

    argmax_match = (hf_c.argmax(-1) == vl_c.argmax(-1)).float().mean().item()
    max_diff = (hf_c - vl_c).abs().max().item()

    # argmax-exact is the correctness gate; magnitude diff is bf16-ULP scale
    # (logit magnitude ~30, ULP ~0.25) across different attention kernels.
    assert argmax_match == 1.0, f"argmax mismatch on canvas: {argmax_match:.4f}"
    assert max_diff < 0.6, f"logit max|diff| too large: {max_diff:.4f}"


# ---------------------------------------------------------------------------
# GPU + weights: greedy end-to-end generation
# ---------------------------------------------------------------------------
@requires_gpu
@requires_weights
def test_greedy_generation(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    tok, _ = _load_hf(device="cpu")
    prompt_ids = _prompt_ids(tok, device="cpu")[0].tolist()

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    out = llm.generate(
        {"prompt_token_ids": prompt_ids},
        # diffusion path requires temperature==1.0; sampler is greedy internally.
        SamplingParams(temperature=1.0, max_tokens=CANVAS_LENGTH),
    )
    text = out[0].outputs[0].text
    assert "4" in text, f"unexpected generation: {text!r}"


# ---------------------------------------------------------------------------
# GPU + weights: engine-wide stochastic sampling (RL rollouts)
# ---------------------------------------------------------------------------
@requires_gpu
@requires_weights
def test_stochastic_rollouts(monkeypatch):
    """diffusion_config temperature=1.0 -> distinct rollouts with sane
    per-token logprobs captured at unmask time (the RL rollout contract)."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    tok, _ = _load_hf(device="cpu")
    msgs = [{"role": "user", "content": "Tell me a short story about a robot."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True)
    prompt_ids = enc["input_ids"] if not isinstance(enc, list) else enc

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
        diffusion_config={"temperature": 1.0},
    )
    outs = llm.generate(
        [{"prompt_token_ids": list(prompt_ids)}] * 4,
        SamplingParams(temperature=1.0, max_tokens=2 * CANVAS_LENGTH, logprobs=0),
    )
    texts = {o.outputs[0].text for o in outs}
    assert len(texts) >= 2, f"4 rollouts at T=1 were identical: {texts}"
    for o in outs:
        c = o.outputs[0]
        assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
        for tid, entry in zip(c.token_ids, c.logprobs):
            lp = entry[tid]
            assert lp.logprob <= 1e-6 and lp.rank >= 1
