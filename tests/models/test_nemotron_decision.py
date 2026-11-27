# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronForDecision: the model, its config hook and the packed layout."""

import json

import pytest
import regex as re
import torch
from transformers import Ministral3Config

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.model_executor.models.nemotron_packed_layout import (
    ALL_QUESTIONS,
    ATTENTION_LAYOUT,
    NO_QUESTIONS,
    OWN_QUESTION,
    STATE_END_TOKEN_ID,
    attention_layout,
    packed_read,
    state_end_token_id,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionType


def save_checkpoint(path, **config):
    """Write the checkpoint's `config.json`, with the keys of `config` added."""
    Ministral3Config(
        architectures=["NemotronForDecision"],
        is_causal=False,
        use_cache=False,
        mask_token_id=31,
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 8,
            "llama_4_scaling_beta": 0.1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        **config,
    ).save_pretrained(path)


STATE_END = {STATE_END_TOKEN_ID: 30}
"""A checkpoint whose prompt closes the state with a token. No test here has a
tokenizer, so any id of the vocabulary does but the mask's, 31."""

PACKED = {ATTENTION_LAYOUT: NO_QUESTIONS, **STATE_END}
"""What a checkpoint that is read packed declares: the layout, and the
state-end token a packed read needs."""

both_reads = pytest.mark.parametrize(
    "config",
    [{}, PACKED],
    indirect=True,
    ids=["one-read-each", "packed"],
)
"""The test runs on a checkpoint that does not name a layout and on one that
declares packed reads."""

FLASH_ATTENTION_AVAILABLE = (
    "vllm.v1.attention.backends.fa_utils.is_flash_attn_varlen_func_available"
)


@pytest.fixture
def config(tmp_path, request, monkeypatch):
    # Without a parameter the checkpoint does not name a layout.
    declared = getattr(request, "param", {})
    save_checkpoint(tmp_path, **declared)
    with monkeypatch.context() as startup:
        if declared.get(ATTENTION_LAYOUT) == NO_QUESTIONS:
            # Packed reads start only where FlashAttention runs. No test here
            # runs a forward, so a host without it builds the model as well.
            startup.setattr(FLASH_ATTENTION_AVAILABLE, lambda: True)
        return VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=128,
                max_num_seqs=2,
                max_num_batched_tokens=64,
                max_num_scheduled_tokens=32,
                is_encoder_decoder=False,
                enable_chunked_prefill=True,
            ),
            cache_config={"enable_prefix_caching": True},
        )


@pytest.fixture
def backbone(config, dist_init):
    cls, _ = ModelRegistry.resolve_model_cls(
        config.model_config.architectures, config.model_config
    )
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        return cls(vllm_config=config)


@both_reads
def test_model_uses_encoder_only_attention(config, backbone):
    from vllm.model_executor.models.nemotron_decision import (
        NemotronForDecision,
        NemotronPackedReadImpl,
        NemotronPackedReadModelState,
    )
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

    model_config = config.model_config
    assert isinstance(model_config.hf_config, Ministral3Config)
    assert model_config.runner_type == "generate"
    assert not model_config.hf_config.is_causal
    assert not config.cache_config.enable_prefix_caching
    assert not config.scheduler_config.enable_chunked_prefill
    assert type(backbone) is NemotronForDecision
    # With packed reads, the encoder-only state and FlashAttention that also
    # run them. Without, those of any encoder-only model.
    packed = packed_read(model_config.hf_config)
    model_state_cls = resolve_model_state_cls(config, backbone)
    assert model_state_cls is (
        NemotronPackedReadModelState if packed else EncoderOnlyModelState
    )
    assert issubclass(model_state_cls, EncoderOnlyModelState)
    for layer in backbone.model.layers:
        assert layer.self_attn.attn.attn_type == AttentionType.ENCODER_ONLY
        assert (type(layer.self_attn.attn.impl) is NemotronPackedReadImpl) == packed
    # Transformers' Ministral3 rounds the gain before multiplying BF16 queries. Test
    # positions on both sides of the original context-length boundary.
    attention = backbone.model.layers[0].self_attn
    positions = torch.tensor([0, 7, 8, 16, 96])
    expected = (1 + 0.1 * torch.log1p(positions.div(8, rounding_mode="floor"))).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        attention._get_llama_4_attn_scale(positions).flatten(), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("checkpoint_names", [True, False])
@both_reads
def test_checkpoint_weights_keep_their_tensor_layout(backbone, checkpoint_names):
    prefix = "encoder" if checkpoint_names else "model"
    head = "diffusion_head" if checkpoint_names else "lm_head"
    layer = backbone.model.layers[0]
    hidden = backbone.config.hidden_size
    qkv = layer.self_attn.qkv_proj
    generator = torch.Generator().manual_seed(123)

    # Distinct rows detect Q/K rotary-head permutations as well as wrong shards.
    def weight(rows):
        return torch.randn(rows, hidden, generator=generator).to(torch.bfloat16)

    query = weight(layer.self_attn.q_size)
    key = weight(layer.self_attn.kv_size)
    value = weight(layer.self_attn.kv_size)
    head_weight = weight(backbone.config.vocab_size)
    weights = {
        f"{prefix}.layers.0.self_attn.q_proj.weight": query,
        f"{prefix}.layers.0.self_attn.k_proj.weight": key,
        f"{prefix}.layers.0.self_attn.v_proj.weight": value,
        f"{head}.weight": head_weight,
    }
    assert backbone.load_weights(weights.items()) == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "lm_head.weight",
    }
    torch.testing.assert_close(qkv.weight, torch.cat([query, key, value]))
    torch.testing.assert_close(backbone.lm_head.weight[: len(head_weight)], head_weight)


@pytest.mark.parametrize(
    ("declared", "overrides", "expected"),
    [
        ({}, {}, OWN_QUESTION),
        (STATE_END, {}, OWN_QUESTION),
        (PACKED, {}, NO_QUESTIONS),
        ({ATTENTION_LAYOUT: OWN_QUESTION, **STATE_END}, {}, OWN_QUESTION),
        (STATE_END, {ATTENTION_LAYOUT: NO_QUESTIONS}, NO_QUESTIONS),
        ({}, PACKED, NO_QUESTIONS),
        (
            {ATTENTION_LAYOUT: OWN_QUESTION, **STATE_END},
            {ATTENTION_LAYOUT: NO_QUESTIONS},
            NO_QUESTIONS,
        ),
        (PACKED, {ATTENTION_LAYOUT: OWN_QUESTION}, OWN_QUESTION),
    ],
    ids=[
        "absent",
        "absent-with-state-end-token",
        "declared-packed",
        "declared-own-question",
        "overridden-packed",
        "both-keys-overridden",
        "packed-over-own-question",
        "own-question-over-packed",
    ],
)
def test_attention_layout_is_own_question_unless_declared_or_overridden(
    tmp_path, declared, overrides, expected
):
    """The checkpoint's `config.json` names the layout, `--hf-overrides` wins
    over it, and without the key each question is read on its own."""
    save_checkpoint(tmp_path, **declared)
    hf_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
    ).hf_config
    assert attention_layout(hf_config) == expected
    assert packed_read(hf_config) is (expected == NO_QUESTIONS)


@pytest.mark.parametrize(
    ("declared", "overrides"),
    [
        ({}, {ATTENTION_LAYOUT: "packed"}),
        # What `--hf-overrides.decision_attention_layout true` parses to.
        (STATE_END, {ATTENTION_LAYOUT: True}),
        (PACKED, {ATTENTION_LAYOUT: NO_QUESTIONS.replace("_", "-")}),
        ({ATTENTION_LAYOUT: NO_QUESTIONS.upper(), **STATE_END}, {}),
        ({ATTENTION_LAYOUT: 1}, {}),
    ],
    ids=[
        "override-not-a-layout",
        "override-true",
        "override-hyphens",
        "declared-uppercase",
        "declared-1",
    ],
)
def test_attention_layout_takes_only_a_layout_it_reads(tmp_path, declared, overrides):
    """A value that is not a layout stops the server, also one that names a
    layout but for its case or a hyphen."""
    save_checkpoint(tmp_path, **declared)
    value = (declared | overrides)[ATTENTION_LAYOUT]
    message = (
        f"NemotronForDecision takes {ATTENTION_LAYOUT} {OWN_QUESTION} or "
        f"{NO_QUESTIONS}, not {value!r}."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
        )


@pytest.mark.parametrize(
    ("declared", "overrides"),
    [
        ({ATTENTION_LAYOUT: ALL_QUESTIONS, **STATE_END}, {}),
        (PACKED, {ATTENTION_LAYOUT: ALL_QUESTIONS}),
    ],
    ids=["declared", "overridden"],
)
def test_state_sees_all_questions_is_not_served_yet(tmp_path, declared, overrides):
    """The checkpoint format names a third layout. The server does not start
    with it, and says which layouts it reads."""
    save_checkpoint(tmp_path, **declared)
    message = (
        f"NemotronForDecision does not serve {ATTENTION_LAYOUT} {ALL_QUESTIONS} "
        f"yet. Use {OWN_QUESTION} or {NO_QUESTIONS}."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
        )


@pytest.mark.parametrize(
    ("declared", "overrides"),
    [
        ({ATTENTION_LAYOUT: NO_QUESTIONS}, {}),
        ({}, {ATTENTION_LAYOUT: NO_QUESTIONS}),
        # JSON null in `--hf-overrides` serves the prompt without the token.
        (PACKED, {STATE_END_TOKEN_ID: None}),
    ],
    ids=["declared", "overridden", "token-overridden-to-null"],
)
def test_packed_reads_need_a_state_end_token(tmp_path, declared, overrides):
    """A packed read shares the tokens through the state-end token, so the
    server does not start with packed reads of a prompt that has none."""
    save_checkpoint(tmp_path, **declared)
    message = (
        f"NemotronForDecision reads {ATTENTION_LAYOUT} {NO_QUESTIONS} only from "
        f"a checkpoint whose prompt closes the state with a token: its "
        f"config.json has no {STATE_END_TOKEN_ID}."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
        )


@pytest.mark.parametrize(
    ("declared", "overrides"),
    [
        ({STATE_END_TOKEN_ID: "30"}, {}),
        # The token's text, which is not its id.
        ({}, {STATE_END_TOKEN_ID: "<state-end>"}),
        ({STATE_END_TOKEN_ID: True}, {}),
        (PACKED, {STATE_END_TOKEN_ID: 30.0}),
        ({STATE_END_TOKEN_ID: -1}, {}),
    ],
    ids=[
        "declared-quoted",
        "override-token-text",
        "declared-true",
        "override-float",
        "declared-negative",
    ],
)
def test_state_end_token_id_is_a_token_id(tmp_path, declared, overrides):
    """An id that is not a non-negative integer stops the server, with either
    layout."""
    save_checkpoint(tmp_path, **declared)
    value = (declared | overrides)[STATE_END_TOKEN_ID]
    message = (
        f"NemotronForDecision takes {STATE_END_TOKEN_ID} as a token id, not "
        f"{value!r} ({type(value).__name__})."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
        )


def test_checkpoint_names_its_state_end_token_id(tmp_path):
    save_checkpoint(tmp_path, **PACKED)

    def served(**overrides):
        return ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, hf_overrides=overrides
        ).hf_config

    assert state_end_token_id(served()) == 30
    assert packed_read(served())
    # Read one question at a time, the prompt keeps its token.
    own_question = served(**{ATTENTION_LAYOUT: OWN_QUESTION})
    assert state_end_token_id(own_question) == 30
    assert not packed_read(own_question)
    # The prompt without the token, for the same checkpoint. It is not read
    # packed.
    plain = served(**{ATTENTION_LAYOUT: OWN_QUESTION, STATE_END_TOKEN_ID: None})
    assert state_end_token_id(plain) is None
    assert not packed_read(plain)


@pytest.mark.parametrize(
    ("key", "name", "value"),
    [
        (ATTENTION_LAYOUT, ATTENTION_LAYOUT.replace("_", "-"), OWN_QUESTION),
        (ATTENTION_LAYOUT, ATTENTION_LAYOUT.title(), OWN_QUESTION),
        (STATE_END_TOKEN_ID, STATE_END_TOKEN_ID.replace("_", "-"), 29),
        (STATE_END_TOKEN_ID, STATE_END_TOKEN_ID.title(), 29),
    ],
)
@pytest.mark.parametrize("source", ["declared", "overridden"])
def test_misspelled_read_key_is_refused(tmp_path, key, name, value, source):
    """A key that is a read key but for a hyphen or its case would be
    ignored: `state_sees_own_question` under it would leave a checkpoint's
    packed reads on, and a token id under it would not change the token that
    closes the state."""
    declared = PACKED | ({name: value} if source == "declared" else {})
    save_checkpoint(tmp_path, **declared)
    message = (
        f"NemotronForDecision reads {key}, so {name!r} would be ignored. "
        f"Spell the key {key} in --hf-overrides and in the "
        "checkpoint's config.json."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            hf_overrides={name: value} if source == "overridden" else {},
        )


@pytest.mark.parametrize(
    ("declared", "argv", "expected"),
    [
        (
            STATE_END,
            ["--hf-overrides", json.dumps({ATTENTION_LAYOUT: NO_QUESTIONS})],
            NO_QUESTIONS,
        ),
        (STATE_END, [f"--hf-overrides.{ATTENTION_LAYOUT}", NO_QUESTIONS], NO_QUESTIONS),
        (PACKED, [f"--hf-overrides.{ATTENTION_LAYOUT}", OWN_QUESTION], OWN_QUESTION),
        ({}, ["--hf-overrides", json.dumps(PACKED)], NO_QUESTIONS),
        (
            {},
            [
                f"--hf-overrides.{ATTENTION_LAYOUT}",
                NO_QUESTIONS,
                f"--hf-overrides.{STATE_END_TOKEN_ID}",
                "30",
            ],
            NO_QUESTIONS,
        ),
        (
            {},
            [f"--hf-overrides.{ATTENTION_LAYOUT}", NO_QUESTIONS],
            f"its config.json has no {STATE_END_TOKEN_ID}.",
        ),
        (
            PACKED,
            [f"--hf-overrides.{ATTENTION_LAYOUT}", ALL_QUESTIONS],
            f"does not serve {ATTENTION_LAYOUT} {ALL_QUESTIONS} yet.",
        ),
        (
            PACKED,
            [f"--hf-overrides.{ATTENTION_LAYOUT}", "true"],
            f"{OWN_QUESTION} or {NO_QUESTIONS}, not True.",
        ),
        (
            PACKED,
            [f"--hf-overrides.{ATTENTION_LAYOUT.replace('_', '-')}", OWN_QUESTION],
            "would be ignored",
        ),
        (
            PACKED,
            [f"--hf-overrides.{STATE_END_TOKEN_ID}", "<state-end>"],
            "as a token id, not '<state-end>' (str).",
        ),
    ],
    ids=[
        "json-packed",
        "dotted-packed",
        "dotted-own-question",
        "json-both-keys",
        "dotted-both-keys",
        "dotted-packed-without-token",
        "dotted-all-questions",
        "dotted-true",
        "dotted-hyphen",
        "dotted-token-text",
    ],
)
def test_attention_layout_from_the_command_line(tmp_path, declared, argv, expected):
    """Both spellings of `--hf-overrides` reach the hook as `vllm serve`
    parses them: a layout as a string, a token id as a number, `true` as a
    bool."""
    save_checkpoint(tmp_path, **declared)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        [str(tmp_path), "--skip-tokenizer-init", *argv]
    )
    args.model = args.model_tag
    engine_args = AsyncEngineArgs.from_cli_args(args)
    if expected in (OWN_QUESTION, NO_QUESTIONS):
        hf_config = engine_args.create_model_config().hf_config
        assert attention_layout(hf_config) == expected
        assert state_end_token_id(hf_config) == 30
    else:
        with pytest.raises(ValueError, match=re.escape(expected)):
            engine_args.create_model_config()


def test_one_read_per_question_is_the_stock_encoder_only_model(config, backbone):
    """Without packed reads the layers have the backend the selector chooses
    and the model runner resolves the model state."""
    from vllm.model_executor.layers.attention.encoder_only_attention import (
        create_encoder_only_attention_backend,
    )
    from vllm.model_executor.models.nemotron_decision import (
        NemotronPackedReadImpl,
    )
    from vllm.v1.attention.selector import get_attn_backend
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

    assert attention_layout(config.model_config.hf_config) == OWN_QUESTION
    assert not packed_read(config.model_config.hf_config)
    assert not hasattr(backbone, "get_model_state_cls")
    assert resolve_model_state_cls(config, backbone) is EncoderOnlyModelState
    with set_current_vllm_config(config):
        selected = get_attn_backend(
            backbone.config.head_dim,
            torch.bfloat16,
            config.cache_config.cache_dtype,
            attn_type=AttentionType.ENCODER_ONLY,
        )
    for layer in backbone.model.layers:
        attn = layer.self_attn.attn
        assert attn.attn_type == AttentionType.ENCODER_ONLY
        assert attn.attn_backend is create_encoder_only_attention_backend(selected)
        assert type(attn.impl) is selected.get_impl_cls()
        assert not isinstance(attn.impl, NemotronPackedReadImpl)


@pytest.mark.parametrize(
    ("dtype", "backend", "refusal"),
    [
        ("float32", None, "float16 and bfloat16"),
        ("bfloat16", "TRITON_ATTN", "not on --attention-backend TRITON_ATTN"),
    ],
    ids=["float32", "TRITON_ATTN"],
)
def test_only_packed_reads_need_flash_attention(
    tmp_path, caplog_vllm, dtype, backend, refusal
):
    """Without packed reads the server starts with a dtype and an
    `--attention-backend` that packed reads refuse, and says which read it
    does."""
    from vllm.config.attention import AttentionConfig

    def start(**checkpoint):
        save_checkpoint(tmp_path, **checkpoint)
        return VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path), skip_tokenizer_init=True, dtype=dtype
            ),
            attention_config=AttentionConfig(backend=backend),
        )

    config = start()
    assert not config.cache_config.enable_prefix_caching
    assert not config.scheduler_config.enable_chunked_prefill
    assert (
        f"prompts have no state-end token (the checkpoint has no {STATE_END_TOKEN_ID})"
        in caplog_vllm.text
    )
    assert (
        f"in a forward of its own ({ATTENTION_LAYOUT} {OWN_QUESTION})"
        in caplog_vllm.text
    )
    # A state-end token alone does not pack the questions.
    caplog_vllm.clear()
    start(**STATE_END)
    assert (
        f"prompts close the state with token 30 ({STATE_END_TOKEN_ID})"
        in caplog_vllm.text
    )
    assert (
        f"in a forward of its own ({ATTENTION_LAYOUT} {OWN_QUESTION})"
        in caplog_vllm.text
    )
    with pytest.raises(ValueError, match=refusal) as error:
        start(**PACKED)
    # The refusal names the way to serve without packed reads.
    assert (
        f"--hf-overrides.{ATTENTION_LAYOUT} {OWN_QUESTION} serves without packed reads."
        in str(error.value)
    )


def test_packed_read_sequences_follow_the_segment_mask():
    """A packed read's attention sequences give the segment layout: the shared
    tokens see themselves; each question sees them and itself, at the
    positions it has in its own prompt; requests stay apart."""
    from vllm.model_executor.models.nemotron_packed_layout import (
        packed_sequences,
        segment_positions,
        sequence_groups,
    )

    # A plain request of 3 tokens, a packed read of 4 shared tokens and
    # questions of 2, 1 and 3, and a plain request of 2.
    lengths = [4, 2, 1, 3]
    query_start_loc = [0, 3, 13, 15]
    q_lens, k_lens, kv_rows = packed_sequences(query_start_loc, [None, lengths, None])
    assert q_lens == [3, 4, 2, 1, 3, 2]
    assert k_lens == [3, 4, 6, 5, 7, 2]
    num_tokens = query_start_loc[-1]
    attends = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    query, key = 0, 0
    for q_len, k_len in zip(q_lens, k_lens):
        attends[query : query + q_len, kv_rows[key : key + k_len]] = True
        query, key = query + q_len, key + k_len
    assert (query, key) == (num_tokens, len(kv_rows))

    request = torch.tensor([0] * 3 + [1] * 10 + [2] * 2)
    segment = torch.tensor([0] * 3 + [0] * 4 + [1] * 2 + [2] + [3] * 3 + [0] * 2)
    same_request = request[:, None] == request[None, :]
    expected = same_request & (
        (segment[None, :] == 0) | (segment[:, None] == segment[None, :])
    )
    assert torch.equal(attends, expected)
    # Sanity of the expectation: the shared tokens do not see a question, and
    # a question does not see another one.
    assert not expected[3, 7] and not expected[7, 9] and expected[9, 3]

    assert segment_positions(lengths).tolist() == [0, 1, 2, 3, 4, 5, 4, 4, 5, 6]

    # One attention call gathers a bounded number of key rows: sequences in
    # order, as many as fit, and always at least one.
    assert sequence_groups(k_lens, 100) == [(0, 6)]
    assert sequence_groups(k_lens, 13) == [(0, 3), (3, 5), (5, 6)]
    assert sequence_groups(k_lens, 1) == [(i, i + 1) for i in range(6)]


@pytest.mark.parametrize(
    "lengths,valid",
    [
        ([4, 2, 1, 3], True),
        ([9, 1], True),
        ([10], False),
        ([4, 2, 1, 2], False),
        ([4, 0, 3, 3], False),
        ([4, 2.0, 1, 3], False),
        ((4, 2, 1, 3), False),
        (None, False),
    ],
)
def test_packed_read_takes_only_lengths_that_cover_the_prompt(lengths, valid):
    """The lengths come with a request, so a wrong layout is a plain read."""
    from vllm import SamplingParams
    from vllm.model_executor.models.nemotron_packed_layout import (
        SEGMENT_LENGTHS,
        segment_lengths,
    )

    params = SamplingParams(max_tokens=1, extra_args={SEGMENT_LENGTHS: lengths})
    assert (segment_lengths(params, list(range(10))) is not None) == valid
    assert segment_lengths(SamplingParams(max_tokens=1), list(range(10))) is None


def test_packed_reads_run_on_the_model_backend_only(monkeypatch, caplog_vllm):
    """The model gives its FlashAttention backend, which the backend selector
    does not validate: the server starts with it where FlashAttention runs,
    and another requested backend or dtype is refused."""
    from types import SimpleNamespace
    from typing import cast

    from vllm.config.attention import AttentionConfig
    from vllm.model_executor.models.config import NemotronDecisionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    def verify(backend, dtype):
        config = cast(
            VllmConfig,
            SimpleNamespace(
                attention_config=AttentionConfig(backend=backend),
                model_config=SimpleNamespace(
                    dtype=dtype, hf_config=SimpleNamespace(**PACKED)
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
            ),
        )
        NemotronDecisionConfig.verify_and_update_config(config)
        return config

    def flash_attention(available):
        monkeypatch.setattr(FLASH_ATTENTION_AVAILABLE, lambda: available)

    flash_attention(True)
    for backend in (None, AttentionBackendEnum.FLASH_ATTN):
        for dtype in (torch.float16, torch.bfloat16):
            config = verify(backend, dtype)
            assert not config.cache_config.enable_prefix_caching
            assert not config.scheduler_config.enable_chunked_prefill
    assert (
        f"prompts close the state with token 30 ({STATE_END_TOKEN_ID})"
        in caplog_vllm.text
    )
    assert f"the state once ({ATTENTION_LAYOUT} {NO_QUESTIONS})" in caplog_vllm.text
    # Each refusal names the way to serve without packed reads.
    hint = (
        f"--hf-overrides.{ATTENTION_LAYOUT} {OWN_QUESTION} serves without packed reads."
    )
    with pytest.raises(
        ValueError, match="not on --attention-backend TRITON_ATTN"
    ) as error:
        verify(AttentionBackendEnum.TRITON_ATTN, torch.bfloat16)
    assert hint in str(error.value)
    # The selector would have skipped FlashAttention for float32.
    for backend in (None, AttentionBackendEnum.FLASH_ATTN):
        with pytest.raises(ValueError, match="float16 and bfloat16") as error:
            verify(backend, torch.float32)
        assert hint in str(error.value)
    flash_attention(False)
    with pytest.raises(ValueError, match="needs FlashAttention") as error:
        verify(None, torch.bfloat16)
    assert hint in str(error.value)
