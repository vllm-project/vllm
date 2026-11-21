# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reveal-rule unit tests and GPU model parity/generation checks.

Weight-dependent tests load nvidia/Nemotron-Labs-Diffusion-3B from the Hub by
default; set NEMOTRON_DLM_MODEL_PATH to another Hub ID or a local 3B or 8B
checkpoint directory. Synthetic reveal rules run eagerly on CPU; GPU
model tests exercise the compiled sampler with the checkpoint's vocabulary.
"""

import json
import os
from functools import partial

import pytest
import torch

MODEL_PATH = os.environ.get(
    "NEMOTRON_DLM_MODEL_PATH",
    "nvidia/Nemotron-Labs-Diffusion-3B",
)
MASK_TOKEN_ID = 100
CANVAS_LENGTH = 32
MAX_MODEL_LEN = 2048


def _weights_available() -> bool:
    # A Hub ID is downloaded on first use; a local path must be a checkpoint.
    if os.path.isdir(MODEL_PATH) or MODEL_PATH.startswith(("/", ".", "~")):
        return os.path.isfile(os.path.join(MODEL_PATH, "config.json"))
    return True


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
    from vllm.transformers_utils.configs import NemotronLabsDiffusionConfig

    cfg = NemotronLabsDiffusionConfig(block_size=32, mask_token_id=100)
    # canvas_length is the field ModelConfig.is_diffusion keys off of.
    assert cfg.canvas_length == 32
    assert cfg.mask_token_id == 100


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_layers,rope_factor",
    [(3072, 9216, 26, 16.0), (4096, 14336, 34, 8.0)],
    ids=["3b", "8b"],
)
def test_checkpoint_dimensions_and_rope(
    tmp_path, hidden_size, intermediate_size, num_layers, rope_factor
):
    """Loading 8B must preserve its dimensions and RoPE, not 3B defaults."""
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs import NemotronLabsDiffusionConfig

    checkpoint = {
        "model_type": "nemotron_labs_diffusion",
        "architectures": ["NemotronLabsDiffusionModel"],
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": int(16384 * rope_factor),
        "block_size": 32,
        "mask_token_id": 100,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": rope_factor,
            "original_max_position_embeddings": 16384,
            "llama_4_scaling_beta": 0.1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, NemotronLabsDiffusionConfig)
    for key, value in checkpoint.items():
        assert getattr(config, key) == value, key
    assert config.canvas_length == checkpoint["block_size"]


def test_arch_in_model_registry():
    from vllm.model_executor.models.registry import ModelRegistry

    assert "NemotronLabsDiffusionModel" in ModelRegistry.get_supported_archs()


@pytest.mark.parametrize("flash_version", [None, 3])
def test_attention_backend_supports_mixed_causality(flash_version):
    from types import SimpleNamespace
    from typing import cast

    from vllm.config import VllmConfig
    from vllm.config.attention import AttentionConfig
    from vllm.model_executor.models.config import (
        NemotronLabsDiffusionForBlockDiffusionConfig,
    )
    from vllm.transformers_utils.configs import NemotronLabsDiffusionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention = AttentionConfig()
    if flash_version is not None:
        attention.backend = AttentionBackendEnum.FLASH_ATTN
        attention.flash_attn_version = flash_version
    config = cast(
        VllmConfig,
        SimpleNamespace(
            attention_config=attention,
            model_config=SimpleNamespace(
                hf_config=NemotronLabsDiffusionConfig(),
                override_generation_config={},
                is_diffusion=True,
            ),
            diffusion_config=None,
            scheduler_config=None,
        ),
    )
    if flash_version is not None:
        with pytest.raises(ValueError, match="requires FA4"):
            NemotronLabsDiffusionForBlockDiffusionConfig.verify_and_update_config(
                config
            )
    else:
        NemotronLabsDiffusionForBlockDiffusionConfig.verify_and_update_config(config)
        assert attention.backend == AttentionBackendEnum.TRITON_ATTN


@pytest.mark.parametrize(
    "architecture,ar_mode,expected",
    [
        ("NemotronLabsDiffusionModel", False, True),
        ("NemotronLabsDiffusionModel", True, False),
        ("NemotronLabsDiffusionForCausalLM", False, False),
        ("LLaDAModelLM", True, True),
    ],
)
def test_ar_mode_diffusion_detection(architecture, ar_mode, expected):
    from types import SimpleNamespace
    from typing import cast

    from vllm.config import ModelConfig
    from vllm.config.model_arch import ModelArchitectureConfig
    from vllm.transformers_utils.configs import NemotronLabsDiffusionConfig

    # The Nemotron config drops its canvas for the AR selectors, so the
    # generic canvas_length check in ModelConfig.is_diffusion applies.
    # Other models keep a canvas regardless of an unrelated ar_mode flag.
    if architecture.startswith("NemotronLabsDiffusion"):
        hf_config = NemotronLabsDiffusionConfig(
            block_size=32, architectures=[architecture]
        )
        hf_config.ar_mode = ar_mode
        assert (hf_config.canvas_length is not None) is expected
    else:
        hf_config = SimpleNamespace(canvas_length=32, ar_mode=ar_mode)
    config = object.__new__(ModelConfig)
    config.hf_config = hf_config
    config.model_arch_config = cast(
        ModelArchitectureConfig, SimpleNamespace(architectures=[architecture])
    )
    assert config.is_diffusion is expected


@pytest.mark.parametrize("explicit_diffusion_config", [False, True])
def test_ar_mode_config(explicit_diffusion_config):
    from types import SimpleNamespace
    from typing import cast

    from vllm.config import VllmConfig
    from vllm.config.attention import AttentionConfig
    from vllm.config.diffusion import DiffusionConfig
    from vllm.config.scheduler import SchedulerConfig
    from vllm.model_executor.models.config import (
        NemotronLabsDiffusionForBlockDiffusionConfig,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # AR permits FA3 and keeps ordinary scheduler concurrency defaults.
    attention = AttentionConfig(
        backend=AttentionBackendEnum.FLASH_ATTN, flash_attn_version=3
    )
    scheduler = SimpleNamespace(max_num_seqs=SchedulerConfig.DEFAULT_MAX_NUM_SEQS)
    config = cast(
        VllmConfig,
        SimpleNamespace(
            attention_config=attention,
            model_config=SimpleNamespace(
                is_diffusion=False,
                override_generation_config={},
                hf_config=SimpleNamespace(),
            ),
            diffusion_config=(
                DiffusionConfig(canvas_length=32) if explicit_diffusion_config else None
            ),
            scheduler_config=scheduler,
        ),
    )
    if explicit_diffusion_config:
        with pytest.raises(ValueError, match="AR mode cannot be combined"):
            NemotronLabsDiffusionForBlockDiffusionConfig.verify_and_update_config(
                config
            )
    else:
        NemotronLabsDiffusionForBlockDiffusionConfig.verify_and_update_config(config)
        assert config.diffusion_config is None
        assert attention.flash_attn_version == 3
        assert scheduler.max_num_seqs == SchedulerConfig.DEFAULT_MAX_NUM_SEQS
        assert config.model_config.override_generation_config["max_new_tokens"] is None


BIDIRECTIONAL_READS = {"ar_mode": True, "is_causal": False}


def _engine_config(
    tmp_path, hf_overrides, architecture="NemotronLabsDiffusionModel", **engine_args
):
    """The engine config of a checkpoint with only a config.json."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.transformers_utils.configs import NemotronLabsDiffusionConfig

    NemotronLabsDiffusionConfig(architectures=[architecture]).save_pretrained(tmp_path)
    return EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        hf_overrides=hf_overrides,
        **engine_args,
    ).create_engine_config()


def test_bidirectional_reads_config(tmp_path):
    user = (
        "vllm.model_executor.models.ming_v3_asr_logits_processor:"
        "NoRepeatNGramLogitsProcessor"
    )
    config = _engine_config(tmp_path, BIDIRECTIONAL_READS, logits_processors=[user])
    # A bidirectional prompt's keys depend on later tokens, and a prompt
    # chunk would not attend to the later chunks.
    assert not config.cache_config.enable_prefix_caching
    assert not config.scheduler_config.enable_chunked_prefill
    # Requests without max_tokens read one token.
    assert config.model_config.override_generation_config["max_new_tokens"] == 1
    assert config.model_config.logits_processors == [
        user,
        "vllm.model_executor.models.nemotron_dllm:NemotronBidirectionalReadValidator",
    ]


@pytest.mark.parametrize("budget,expected", [(2048, 4096), (8192, 8192)])
def test_bidirectional_reads_fit_each_prompt_in_one_step(tmp_path, budget, expected):
    """Without chunked prefill, a prompt above the step budget would wait
    forever, so the budget covers --max-model-len."""
    config = _engine_config(
        tmp_path,
        BIDIRECTIONAL_READS,
        max_model_len=4096,
        max_num_batched_tokens=budget,
    )
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.max_num_batched_tokens == expected


def test_bidirectional_reads_cap_the_step(tmp_path):
    """Without a KV cache, compiled kernels index the gate_up activation
    (tokens x 2 * intermediate_size) with 32-bit offsets, so a step stays
    below 2**31 of its elements."""
    limit = (2**31 - 1) // (2 * 9216)
    config = _engine_config(tmp_path, BIDIRECTIONAL_READS)
    assert config.model_config.max_model_len == limit
    assert config.scheduler_config.max_num_batched_tokens == limit
    with pytest.raises(ValueError, match="tokens per step"):
        _engine_config(tmp_path, BIDIRECTIONAL_READS, max_model_len=limit + 1)
    with pytest.raises(ValueError, match="tokens per step"):
        _engine_config(tmp_path, BIDIRECTIONAL_READS, max_num_batched_tokens=limit + 1)


@pytest.mark.parametrize(
    "hf_overrides,engine_args,match",
    [
        # Masked diffusion needs a causal prompt KV cache.
        ({"is_causal": False}, {}, "ar_mode=True"),
        (
            BIDIRECTIONAL_READS,
            {"speculative_config": {"method": "ngram", "num_speculative_tokens": 2}},
            "speculative decoding",
        ),
        (
            BIDIRECTIONAL_READS,
            {"max_model_len": 4096, "max_num_scheduled_tokens": 2048},
            "max-num-scheduled-tokens",
        ),
        (
            BIDIRECTIONAL_READS,
            {"override_generation_config": {"max_new_tokens": 16}},
            "max_new_tokens must be 1",
        ),
    ],
)
def test_bidirectional_reads_refuse_at_startup(
    tmp_path, hf_overrides, engine_args, match
):
    with pytest.raises(ValueError, match=match):
        _engine_config(tmp_path, hf_overrides, **engine_args)


@pytest.mark.parametrize(
    "architecture,hf_overrides,max_tokens,valid",
    [
        ("NemotronLabsDiffusionModel", BIDIRECTIONAL_READS, 1, True),
        ("NemotronLabsDiffusionModel", BIDIRECTIONAL_READS, 2, False),
        ("NemotronLabsDiffusionModel", BIDIRECTIONAL_READS, None, False),
        ("NemotronLabsDiffusionForCausalLM", {"is_causal": False}, 4, False),
        ("NemotronLabsDiffusionModel", {"ar_mode": True}, 4, True),
    ],
)
def test_bidirectional_reads_require_one_token(
    tmp_path, architecture, hf_overrides, max_tokens, valid
):
    """The engine's admission check of logits processor arguments, which
    precedes any worker."""
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.worker.gpu.sample.logits_processor import (
        build_custom_logits_processors_params_validator,
    )

    config = _engine_config(tmp_path, hf_overrides, architecture)
    validate = build_custom_logits_processors_params_validator(
        config.model_config.logits_processors
    )
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    if valid:
        validate(params)
    else:
        with pytest.raises(VLLMValidationError, match="max_tokens=1"):
            validate(params)
    if hf_overrides == BIDIRECTIONAL_READS:
        # Prompt positions attend to later tokens, so their logprobs are not
        # next-token scores.
        with pytest.raises(VLLMValidationError, match="prompt logprobs"):
            validate(SamplingParams(max_tokens=1, prompt_logprobs=1))


def test_diffusion_model_options_are_model_specific():
    """DiffusionConfig parses without a model; each model checks its own options."""
    from types import SimpleNamespace
    from typing import Any, cast

    from vllm.config.diffusion import DiffusionConfig
    from vllm.model_executor.models.config import (
        DiffusionGemmaDiffusionOptions,
        NemotronDiffusionOptions,
        _parse_diffusion_model_options,
    )

    def parse(options: dict[str, Any], cls: type) -> Any:
        dc = DiffusionConfig(canvas_length=32, model_options=options)
        config = cast(Any, SimpleNamespace(diffusion_config=dc))
        _parse_diffusion_model_options(config, cls)
        return dc.model_options

    opts = parse({"temperature": 0.0}, NemotronDiffusionOptions)
    assert opts.temperature == 0.0
    with pytest.raises(ValueError):
        parse({"temperature": 0.0}, DiffusionGemmaDiffusionOptions)
    with pytest.raises(ValueError):
        parse({"temperature": -0.5}, NemotronDiffusionOptions)


# ---------------------------------------------------------------------------
# CPU, no weights: masked-diffusion step semantics
# ---------------------------------------------------------------------------
@pytest.fixture
def eager_sampler():
    """Test reveal rules independently of Inductor; model tests cover compilation."""
    with torch.compiler.set_stance("force_eager"):
        yield


def test_masked_step_sampling_semantics(eager_sampler):
    from vllm.model_executor.models.nemotron_dllm import _compiled_masked_step

    device = "cpu"
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

    def run_step(st, logits, temperature, leftmost=False, steps=max_steps):
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
            torch.full((1,), temperature, dtype=torch.float32, device=device),
            st["canvas"],
            st["step"],
            st["phase"],
            st["lp"],
            st["rk"],
            sampled,
            num_sampled,
            draft,
            max_denoising_steps=steps,
            mask_token_id=mask_id,
            CL=CL,
            all_greedy=temperature == 0,
            capture_logprobs=True,
            leftmost=leftmost,
            threshold_mode=False,
            log_threshold=0.0,
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


@pytest.mark.parametrize("max_steps,valid_length", [(4, 4), (1, 4), (4, 2)])
def test_threshold_unmasking_and_commit(max_steps, valid_length, eager_sampler):
    """Threshold decoding progresses, preserves revealed tokens, and clips padding."""
    import math

    from vllm.model_executor.models.nemotron_dllm import _compiled_masked_step

    device = "cpu"
    width, vocab, mask = 4, 8, 7
    slots = torch.tensor([1], device=device)
    indices = torch.tensor([0], device=device)
    lengths = torch.tensor([valid_length], device=device)
    temps = torch.zeros(1, device=device)
    canvas = torch.full((2, width), mask, device=device, dtype=torch.int64)
    step = torch.zeros(2, device=device, dtype=torch.int32)
    phase = torch.zeros(2, device=device, dtype=torch.bool)
    lp = torch.zeros(2, width, device=device)
    ranks = torch.zeros(2, width, device=device, dtype=torch.int32)
    sampled = torch.zeros(1, width, device=device, dtype=torch.int32)
    count = torch.zeros(1, device=device, dtype=torch.int32)
    draft = torch.zeros_like(canvas)
    logits = torch.zeros(width, vocab, device=device)
    logits[:, mask] = -100
    logits[0, 2] = 10  # Only the first position clears 0.9.

    def advance():
        _compiled_masked_step(
            logits,
            slots,
            indices,
            slots,
            lengths,
            temps,
            canvas,
            step,
            phase,
            lp,
            ranks,
            sampled,
            count,
            draft,
            max_denoising_steps=max_steps,
            mask_token_id=mask,
            CL=width,
            all_greedy=True,
            capture_logprobs=True,
            leftmost=False,
            threshold_mode=True,
            log_threshold=math.log(0.9),
        )

    advance()
    assert count.item() == 0
    assert canvas[1, 0].item() == 2
    if max_steps > 1:
        assert (canvas[1, 1:] == mask).all()
    first_lp = lp[1, 0].clone()
    logits[0, 3] = 20  # An already revealed position must remain unchanged.
    for _ in range(valid_length - 1 if max_steps > 1 else 0):
        advance()
    assert phase[1].item()
    assert (canvas[1, :valid_length] != mask).all()
    assert (canvas[1, valid_length:] == mask).all()
    assert canvas[1, 0].item() == 2
    assert lp[1, 0] == first_lp
    completed = canvas[1].clone()
    advance()
    assert count.item() == valid_length
    assert torch.equal(sampled[0, :valid_length], completed[:valid_length])
    assert (canvas[1] == mask).all()
    assert not phase[1].item()
    assert step[1].item() == 0
    assert (canvas[0] == mask).all()  # Unscheduled request is untouched.


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


@pytest.fixture
def fp32_reference(monkeypatch):
    """Isolate architecture parity from BF16 rounding and Triton TF32 dots."""
    precision = torch.get_float32_matmul_precision()
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "highest")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _load_hf(dtype=torch.bfloat16, device="cuda"):
    _install_tf_compat()
    from transformers import AutoModel

    tok = _load_tokenizer()
    model = (
        AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    return tok, model


def _prompt_ids(tok, device="cuda"):
    msgs = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
    return ids.to(device)


def _vllm_manual_forward(obj, token_ids, causal_flag):
    """Single forward of the loaded vLLM model over `token_ids` as one request
    with per-request `causal=causal_flag`, driving the GPU model runner
    (KV-cache slots, attn metadata, forward context). Returns fp32 logits
    `[len(token_ids), vocab]` on CPU. Runs under `collective_rpc`.
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
    block_size = kv_cache_config.kv_cache_groups[0].block_size

    length = len(token_ids)
    n_blocks = (length + block_size - 1) // block_size
    # Block 0 is the null block; use fresh blocks from 1.
    block_ids = torch.arange(1, n_blocks + 1, dtype=torch.int32, device=device)
    block_table = block_ids.unsqueeze(0)
    pos = torch.arange(length, dtype=torch.int64, device=device)
    slot_mapping = (
        block_ids[pos // block_size].to(torch.int64) * block_size + pos % block_size
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
def test_logits_parity(monkeypatch, fp32_reference):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    tok, hf = _load_hf(dtype=torch.float32)
    prompt = _prompt_ids(tok)
    canvas = torch.full(
        (1, CANVAS_LENGTH), MASK_TOKEN_ID, dtype=prompt.dtype, device=prompt.device
    )
    seq = torch.cat([prompt, canvas], dim=1)
    token_ids = seq[0].tolist()

    with torch.no_grad():
        hf_logits = hf(seq).logits[0].float().cpu()  # fully bidirectional
    del hf
    torch.accelerator.empty_cache()

    llm = LLM(
        model=MODEL_PATH,
        dtype="float32",
        max_num_batched_tokens=128,
        max_num_seqs=4,
        attention_config={"backend": "TRITON_ATTN"},
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    # causal_flag=False -> bidirectional canvas (matches HF single forward).
    vllm_logits = llm.llm_engine.collective_rpc(
        partial(_vllm_manual_forward, token_ids=token_ids, causal_flag=False)
    )[0]

    llm.llm_engine.engine_core.shutdown()
    del llm
    cleanup_dist_env_and_memory()

    canvas_slice = slice(prompt.shape[1], seq.shape[1])
    hf_c = hf_logits[canvas_slice]
    vl_c = vllm_logits[canvas_slice]

    argmax_match = (hf_c.argmax(-1) == vl_c.argmax(-1)).float().mean().item()
    max_diff = (hf_c - vl_c).abs().max().item()

    assert argmax_match == 1.0, f"argmax mismatch on canvas: {argmax_match:.4f}"
    assert max_diff < 1e-3, f"logit max|diff| too large: {max_diff:.6f}"


# ---------------------------------------------------------------------------
# GPU + weights: greedy end-to-end generation
# ---------------------------------------------------------------------------
@requires_gpu
@requires_weights
def test_greedy_generation(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt

    tok = _load_tokenizer()
    prompt_ids = _prompt_ids(tok, device="cpu")[0].tolist()

    llm = LLM(
        model=MODEL_PATH,
        attention_config={"backend": "TRITON_ATTN"},
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    params = [
        SamplingParams(temperature=0.0, max_tokens=CANVAS_LENGTH, logprobs=n)
        for n in (None, 0, 5)
    ]
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * 3, params)
    del llm
    cleanup_dist_env_and_memory()
    for o in outs:
        text = o.outputs[0].text
        assert "4" in text, f"unexpected generation: {text!r}"
    # Batched with a top-k request, logprobs=0 still reports each greedy token
    # at the step it was revealed.
    seq = outs[1].outputs[0]
    assert seq.logprobs is not None
    ranks = [lp[t].rank if lp else None for lp, t in zip(seq.logprobs, seq.token_ids)]
    assert ranks == [1] * len(ranks), ranks


# ---------------------------------------------------------------------------
# GPU + weights: engine-wide stochastic sampling (RL rollouts)
# ---------------------------------------------------------------------------
@requires_gpu
@requires_weights
def test_stochastic_rollouts(monkeypatch):
    """Mixed request temperatures override defaults and return reveal logprobs."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt

    tok = _load_tokenizer()
    msgs = [{"role": "user", "content": "Tell me a short story about a robot."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True)
    prompt_ids = enc["input_ids"] if not isinstance(enc, list) else enc

    llm = LLM(
        model=MODEL_PATH,
        attention_config={"backend": "TRITON_ATTN"},
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
        diffusion_config={"model_options": {"temperature": 0.0}},
    )
    assert llm.get_default_sampling_params().temperature == 0.0
    temperatures = [0.0, 0.7, 1.0, 1.5]
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))] * len(temperatures),
        [
            SamplingParams(
                temperature=t, top_p=0.9, max_tokens=2 * CANVAS_LENGTH, logprobs=0
            )
            for t in temperatures
        ],
    )
    del llm
    cleanup_dist_env_and_memory()
    texts = {o.outputs[0].text for o in outs}
    assert len(texts) >= 2, f"Mixed-temperature rollouts were identical: {texts}"
    for o in outs:
        c = o.outputs[0]
        assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
        for tid, entry in zip(c.token_ids, c.logprobs):
            assert entry is not None
            lp = entry[tid]
            assert lp.rank is not None
            assert lp.logprob <= 1e-6 and lp.rank >= 1


def _assert_ar_runner(obj):
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    runner = getattr(obj, "worker", obj).model_runner
    assert type(runner.model_state) is DefaultModelState
    assert type(runner.sampler) is Sampler


@requires_gpu
@requires_weights
@pytest.mark.parametrize(
    "hf_overrides",
    [
        {"ar_mode": True},
        {"architectures": ["NemotronLabsDiffusionForCausalLM"]},
    ],
    ids=["ar_mode", "causal_architecture"],
)
def test_ar_logits_and_generation(monkeypatch, hf_overrides, fp32_reference):
    """Both AR selectors use causal HF logits and ordinary cached generation."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt

    tok, hf = _load_hf(dtype=torch.float32)
    prompt = _prompt_ids(tok)
    with torch.inference_mode():
        # Native AR generation disables diffusion_lm on the HF attention
        # layers; otherwise they discard even an explicitly causal mask.
        hf_output, _ = hf.ar_generate(prompt, max_new_tokens=16, temperature=0.0)
        hf_logits = hf(prompt, use_causal_mask=True).logits[0].float().cpu()
    hf_tokens = hf_output[0, prompt.shape[1] :].tolist()
    del hf
    torch.accelerator.empty_cache()

    llm = LLM(
        model=MODEL_PATH,
        dtype="float32",
        max_num_batched_tokens=128,
        max_num_seqs=4,
        hf_overrides=hf_overrides,
        attention_config={"backend": "TRITON_ATTN"},
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    try:
        config = llm.llm_engine.vllm_config
        assert not config.model_config.is_diffusion
        assert config.diffusion_config is None
        assert config.speculative_config is None
        llm.llm_engine.collective_rpc(_assert_ar_runner)
        prompt_ids = prompt[0].tolist()
        vllm_logits = llm.llm_engine.collective_rpc(
            partial(_vllm_manual_forward, token_ids=prompt_ids, causal_flag=True)
        )[0]
        # Compare every prompt position, so a bidirectional prefill cannot pass
        # by agreeing only on the last position.
        torch.testing.assert_close(vllm_logits, hf_logits, atol=1e-3, rtol=0.0)
        assert vllm_logits[-1].argmax() == hf_logits[-1].argmax()

        greedy = llm.generate(
            TokensPrompt(prompt_token_ids=prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=16, logprobs=0),
        )[0].outputs[0]
        assert list(greedy.token_ids) == hf_tokens
        assert greedy.logprobs is not None
        # Exercise fractional-temperature sampling and logprobs as well.
        sampled = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)] * 2,
            SamplingParams(
                temperature=0.7, top_p=0.9, top_k=20, max_tokens=8, logprobs=0
            ),
        )
        for output in sampled:
            completion = output.outputs[0]
            assert 0 < len(completion.token_ids) <= 8
            assert completion.logprobs is not None
            assert len(completion.logprobs) == len(completion.token_ids)
            for token_id, entry in zip(completion.token_ids, completion.logprobs):
                assert entry is not None
                assert entry[token_id].logprob <= 1e-6
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        torch._dynamo.reset()
        cleanup_dist_env_and_memory()


def test_temperature_precedes_top_p(monkeypatch):
    from vllm.model_executor.models import nemotron_dllm
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    monkeypatch.setattr(nemotron_dllm, "apply_top_k_top_p", apply_top_k_top_p_pytorch)
    logits = torch.tensor([[0.6, 0.3, 0.1, 10.0]]).log().repeat(3, 1)
    filtered = nemotron_dllm._filter_sampling_logits(
        logits, torch.tensor([0.0, 0.5, 2.0]), None, torch.full((3,), 0.7), 3
    )
    # At T=0.5 the leading token alone exceeds top_p; at T=1 or T=2 it
    # takes two. The mask must not consume the nucleus probability budget.
    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    assert torch.equal(filtered.isfinite(), expected)
    torch.testing.assert_close(filtered[expected], logits[expected])


def test_mixed_temperature_distribution_and_logprobs(eager_sampler):
    from vllm.model_executor.models.nemotron_dllm import _compiled_masked_step

    # Many independent positions give a low-variance check of the actual
    # sampling law, not just the temperatures attached to returned logprobs.
    rows, width, vocab, mask = 3, 4096, 3, 2
    temperatures = torch.tensor([0.0, 0.5, 2.0])
    logits = torch.tensor([0.0, 1.0, float("-inf")]).expand(rows, width, vocab)
    slots = torch.arange(rows)
    canvas = torch.full((rows, width), mask, dtype=torch.int64)
    logprobs = torch.zeros(rows, width)
    torch.manual_seed(17)
    processed = _compiled_masked_step(
        logits.reshape(-1, vocab),
        slots,
        slots,
        slots,
        torch.full((rows,), width),
        temperatures,
        canvas,
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.bool),
        logprobs,
        torch.zeros(rows, width, dtype=torch.int32),
        torch.zeros(rows, width, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros_like(canvas),
        max_denoising_steps=1,
        mask_token_id=mask,
        CL=width,
        all_greedy=False,
        capture_logprobs=True,
        leftmost=False,
        threshold_mode=True,
        log_threshold=0.0,
    )
    assert (canvas[0] == 1).all()
    for row, temperature in enumerate(temperatures.tolist()):
        distribution = torch.distributions.Categorical(
            logits=torch.tensor([0.0, 1.0]) / (temperature or 1.0)
        )
        torch.testing.assert_close(logprobs[row], distribution.log_prob(canvas[row]))
        torch.testing.assert_close(
            processed[row].log_softmax(-1).gather(-1, canvas[row, :, None]).squeeze(-1),
            logprobs[row],
        )
        if temperature > 0:
            frequency = (canvas[row] == 1).float().mean()
            assert abs(frequency - distribution.probs[1]) < 0.03


@requires_gpu
@requires_weights
def test_bidirectional_read_matches_hf(monkeypatch, fp32_reference):
    """`is_causal=False` reads a trailing mask after one bidirectional pass
    over the whole prompt, matching the HF model's bidirectional logits."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.exceptions import VLLMValidationError
    from vllm.inputs import TokensPrompt

    tok, hf = _load_hf(dtype=torch.float32)
    prompt_ids = _prompt_ids(tok)[0].tolist() + [MASK_TOKEN_ID]
    candidates = sorted(
        {tok.encode(c, add_special_tokens=False)[0] for c in "ABCD0123456789"}
    )
    with torch.inference_mode():
        for layer in hf.encoder.layers:
            layer.self_attn.diffusion_lm = True  # no causal mask anywhere
        logits = hf(torch.tensor([prompt_ids], device="cuda")).logits[0, -1]
        bidirectional = logits.float().log_softmax(-1).cpu()
        for layer in hf.encoder.layers:
            layer.self_attn.diffusion_lm = False
        causal_logits = hf(
            torch.tensor([prompt_ids], device="cuda"), use_causal_mask=True
        ).logits[0, -1]
        causal = causal_logits.float().log_softmax(-1).cpu()
    del hf
    torch.accelerator.empty_cache()
    # The prompt tokens attend to later tokens, so the read must differ from
    # a causal pass; otherwise this test could not detect a causal prefill.
    assert (bidirectional[candidates] - causal[candidates]).abs().max() > 0.05

    llm = LLM(
        model=MODEL_PATH,
        dtype="float32",
        hf_overrides={"ar_mode": True, "is_causal": False},
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=4,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.6,
    )
    try:
        config = llm.llm_engine.vllm_config
        assert not config.model_config.is_diffusion
        assert not config.cache_config.enable_prefix_caching
        assert not config.scheduler_config.enable_chunked_prefill
        assert llm.get_default_sampling_params().max_tokens == 1
        params = SamplingParams(
            temperature=0.0, max_tokens=1, logprob_token_ids=candidates
        )
        prompts = [TokensPrompt(prompt_token_ids=prompt_ids)] * 2
        for output in llm.generate(prompts, params):
            logprobs = output.outputs[0].logprobs
            assert logprobs is not None and len(logprobs) == 1
            for token in candidates:
                assert logprobs[0][token].logprob == pytest.approx(
                    bidirectional[token].item(), abs=2e-3
                )
        with pytest.raises(VLLMValidationError, match="max_tokens=1"):
            llm.generate(prompts[0], SamplingParams(temperature=0.0, max_tokens=2))
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()
