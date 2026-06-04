# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColBERT embedding_mode query/document encoding.

Reviewer-facing coverage map:
- A correctness: marker/truncation/expansion + GPU asymmetry + ranking parity
- B API safety: validate_colbert_embedding_mode + PoolingParams.verify
- C cache: template/config hash + canonical pre-tokenization text + config change
- D batch: batch independence (stateless) + mixed-mode stress + batch-vs-single
- E IO: finalize immutability (no shared EngineInput mutation)
- Gate: pylate ranking parity — exact top-k order (no score tolerance)
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# bf16 GPU embeddings are not bitwise-stable; use relaxed checks here.
# Ranking parity (merge gate) compares order only, not embedding values.
COLBERT_GPU_EMBED_RTOL = 1e-2
COLBERT_GPU_EMBED_ATOL = 1e-2
COLBERT_PYLATE_EMBED_COS_MIN = 0.95

from vllm.model_executor.models.colbert_encoding import (  # noqa: E402
    COLBERT_ATTENTION_MASK_ENFORCED_IN_FORWARD,
    ColBERTEncoder,
    ColBERTEncodingConfig,
    build_colbert_attention_mask,
    canonicalize_colbert_input_text,
    compute_colbert_cache_key,
    compute_colbert_cache_salt,
    compute_colbert_config_hash,
    finalize_colbert_engine_input,
    is_colbert_pooling_model,
    resolve_colbert_chat_template_hash,
    validate_colbert_embedding_mode,
    validate_colbert_engine_inputs,
)

pytest.importorskip("transformers")


def _import_pylate_models():
    import importlib

    pylate = pytest.importorskip("pylate")
    if hasattr(pylate, "models"):
        return pylate.models
    return importlib.import_module("pylate.models")


def _pooling_params(*args, **kwargs):
    """Defer import so CPU-only tests can collect without loading vllm._C."""
    from vllm.pooling_params import PoolingParams

    return PoolingParams(*args, **kwargs)


def _colbert_encode(llm, prompts, pooling_params):
    """ColBERT models expose token_embed only; vLLM 0.22+ requires pooling_task."""
    return llm.encode(prompts, pooling_params, pooling_task="token_embed")


def _mean_per_token_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 1.0
    return F.cosine_similarity(a[:n].float(), b[:n].float(), dim=-1).mean().item()


JINA_MODEL = "jinaai/jina-colbert-v2"
JINA_OVERRIDES = {"hf_overrides": {"architectures": ["ColBERTJinaRobertaModel"]}}
QUERY_TEXT = "What is the capital of France?"
DOC_TEXT = "The capital of France is Paris."


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def _colbert_test_seed():
    set_seed(1234)


def test_colbert_attention_mask_not_enforced_constant():
    assert COLBERT_ATTENTION_MASK_ENFORCED_IN_FORWARD is False


def test_pooling_params_verify_rejects_embedding_mode_on_non_colbert():
    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        runner="pooling",
    )
    params = _pooling_params(task="token_embed", embedding_mode="query")
    with pytest.raises(ValueError, match="only supported for ColBERT"):
        params.verify(model_config)


def test_validate_colbert_embedding_mode_rejects_non_colbert():
    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        runner="pooling",
    )
    with pytest.raises(ValueError, match="only supported for ColBERT"):
        validate_colbert_embedding_mode(model_config, "query")

    validate_colbert_embedding_mode(model_config, None)


def test_is_colbert_pooling_model_detects_jina():
    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    assert is_colbert_pooling_model(model_config)


def test_colbert_insert_prefix_token():
    assert ColBERTEncoder.insert_prefix_token([10, 20, 30], 99) == [10, 99, 20, 30]


def test_colbert_marker_tokens_in_encoded_ids():
    """Query vs document encodings must insert distinct marker token ids."""
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    encoder = ColBERTEncoder(cfg)
    base_ids = tokenizer.encode(QUERY_TEXT, add_special_tokens=True)

    q_ids = encoder.encode_token_ids(list(base_ids), "query")
    d_ids = encoder.encode_token_ids(list(base_ids), "document")

    assert cfg.query_prefix_id in q_ids
    assert cfg.document_prefix_id in d_ids
    assert cfg.query_prefix_id not in d_ids
    assert cfg.document_prefix_id not in q_ids


def test_colbert_query_expansion_padding_only_for_query():
    """Query mode pads with mask tokens to query_length; document does not expand."""
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    encoder = ColBERTEncoder(cfg)
    short_ids = tokenizer.encode("hello", add_special_tokens=True)

    q_ids = encoder.encode_token_ids(list(short_ids), "query")
    d_ids = encoder.encode_token_ids(list(short_ids), "document")

    assert len(q_ids) == cfg.query_length
    assert q_ids.count(cfg.mask_token_id) > 0
    assert len(d_ids) <= cfg.document_length
    assert len(d_ids) < len(q_ids)


def test_pooling_params_embedding_mode_extra_kwargs():
    params = _pooling_params(task="token_embed", embedding_mode="query")
    assert params.extra_kwargs is not None
    assert params.extra_kwargs["colbert_embedding_mode"] == "query"


def test_colbert_config_hash_changes_cache_key():
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    template_hash = resolve_colbert_chat_template_hash(None, model_config=model_config)
    config_hash_a = compute_colbert_config_hash(cfg, model_config=model_config)

    cfg_mutated = ColBERTEncodingConfig(
        query_prefix=cfg.query_prefix,
        document_prefix=cfg.document_prefix,
        query_prefix_id=cfg.query_prefix_id,
        document_prefix_id=cfg.document_prefix_id,
        query_length=cfg.query_length + 1,
        document_length=cfg.document_length,
        mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id,
        do_query_expansion=cfg.do_query_expansion,
        attend_to_expansion_tokens=cfg.attend_to_expansion_tokens,
    )
    config_hash_b = compute_colbert_config_hash(cfg_mutated, model_config=model_config)
    assert config_hash_a != config_hash_b

    key_a = compute_colbert_cache_key(
        model_name=JINA_MODEL,
        embedding_mode="query",
        raw_text=QUERY_TEXT,
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash_a,
    )
    key_b = compute_colbert_cache_key(
        model_name=JINA_MODEL,
        embedding_mode="query",
        raw_text=QUERY_TEXT,
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash_b,
    )
    assert key_a != key_b


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_same_text_query_vs_document_embeddings_differ(
    vllm_runner,
    colbert_model: str,
):
    """Same text must encode differently under query vs document (not cosine-only)."""
    text = "identical passage for asymmetry check"
    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()
        q = _colbert_encode(
            llm,
            [text],
            _pooling_params(task="token_embed", embedding_mode="query"),
        )[0].outputs.data
        d = _colbert_encode(
            llm,
            [text],
            _pooling_params(task="token_embed", embedding_mode="document"),
        )[0].outputs.data

    q_t = torch.as_tensor(q).float()
    d_t = torch.as_tensor(d).float()
    assert q_t.shape != d_t.shape or not torch.allclose(q_t, d_t, atol=1e-3)


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_query_differs_from_document(
    vllm_runner,
    colbert_model: str,
):
    """Query and document modes must produce different embeddings."""
    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()
        q_out = _colbert_encode(
            llm,
            [QUERY_TEXT],
            _pooling_params(task="token_embed", embedding_mode="query"),
        )
        d_out = _colbert_encode(
            llm,
            [DOC_TEXT],
            _pooling_params(task="token_embed", embedding_mode="document"),
        )

    q_emb = torch.as_tensor(q_out[0].outputs.data).float()
    d_emb = torch.as_tensor(d_out[0].outputs.data).float()
    assert q_emb.shape != d_emb.shape or not torch.allclose(q_emb, d_emb, atol=1e-3)


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_query_length_fixed(
    vllm_runner,
    colbert_model: str,
):
    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()
        outputs = _colbert_encode(
            llm,
            [QUERY_TEXT],
            _pooling_params(task="token_embed", embedding_mode="query"),
        )

    q_emb = torch.as_tensor(outputs[0].outputs.data)
    assert q_emb.shape[0] <= 32


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_batch_modes_no_cross_contamination(
    vllm_runner,
    colbert_model: str,
):
    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()
        single_q = _colbert_encode(
            llm,
            [QUERY_TEXT],
            _pooling_params(task="token_embed", embedding_mode="query"),
        )[0].outputs.data
        single_d = _colbert_encode(
            llm,
            [DOC_TEXT],
            _pooling_params(task="token_embed", embedding_mode="document"),
        )[0].outputs.data

        batch = _colbert_encode(
            llm,
            [QUERY_TEXT, DOC_TEXT, QUERY_TEXT],
            [
                _pooling_params(task="token_embed", embedding_mode="query"),
                _pooling_params(task="token_embed", embedding_mode="document"),
                _pooling_params(task="token_embed", embedding_mode="query"),
            ],
        )

    q0 = torch.as_tensor(batch[0].outputs.data).float()
    d1 = torch.as_tensor(batch[1].outputs.data).float()
    q2 = torch.as_tensor(batch[2].outputs.data).float()
    ref_q = torch.as_tensor(single_q).float()
    ref_d = torch.as_tensor(single_d).float()

    torch.testing.assert_close(q0, ref_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q2, ref_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d1, ref_d, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_pylate_parity(
    vllm_runner,
    colbert_model: str,
):
    pylate_models = _import_pylate_models()

    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=8192,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()
        v_q = _colbert_encode(
            llm,
            [QUERY_TEXT],
            _pooling_params(task="token_embed", embedding_mode="query"),
        )[0].outputs.data
        v_d = _colbert_encode(
            llm,
            [DOC_TEXT],
            _pooling_params(task="token_embed", embedding_mode="document"),
        )[0].outputs.data

    model = pylate_models.ColBERT(
        model_name_or_path=colbert_model,
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    )
    p_q = model.encode([QUERY_TEXT], is_query=True, convert_to_tensor=True)[0].float()
    p_d = model.encode([DOC_TEXT], is_query=False, convert_to_tensor=True)[0].float()

    v_q_t = torch.as_tensor(v_q).float().cpu()
    v_d_t = torch.as_tensor(v_d).float().cpu()
    p_q = p_q.detach().cpu()
    p_d = p_d.detach().cpu()

    assert _mean_per_token_cosine(v_q_t, p_q) > COLBERT_PYLATE_EMBED_COS_MIN
    assert _mean_per_token_cosine(v_d_t, p_d) > COLBERT_PYLATE_EMBED_COS_MIN


def test_colbert_cache_key_aligns_with_pre_tokenization():
    from vllm.config import ModelConfig
    from vllm.renderers.params import TokenizeParams
    from vllm.tokenizers.registry import cached_tokenizer_from_config

    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    tokenizer = cached_tokenizer_from_config(model_config)
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    encoder = ColBERTEncoder(cfg)
    base = TokenizeParams(max_total_tokens=8192, do_lower_case=True)
    colbert_tok = encoder.build_tokenize_params("query", base)

    assert canonicalize_colbert_input_text("Hello", colbert_tok, tokenizer) == "hello"

    template_hash = resolve_colbert_chat_template_hash(None, model_config=model_config)
    config_hash = compute_colbert_config_hash(cfg, model_config=model_config)
    key_hello = compute_colbert_cache_key(
        model_name=JINA_MODEL,
        embedding_mode="query",
        raw_text="hello",
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash,
    )
    key_canonical = compute_colbert_cache_key(
        model_name=JINA_MODEL,
        embedding_mode="query",
        raw_text=canonicalize_colbert_input_text("Hello", colbert_tok, tokenizer),
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash,
    )
    key_request_casing = compute_colbert_cache_key(
        model_name=JINA_MODEL,
        embedding_mode="query",
        raw_text="Hello",
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash,
    )
    assert key_hello == key_canonical
    assert key_hello != key_request_casing


def test_colbert_batch_independence_is_stateless():
    """Same (text, mode) tokenizes identically across batch position and grouping."""
    from vllm.config import ModelConfig, VllmConfig
    from vllm.entrypoints.chat_utils import ChatTemplateConfig
    from vllm.entrypoints.pooling.embed.io_processor import (
        ColBERTTokenEmbedIOProcessor,
    )
    from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
    from vllm.renderers.hf import HfRenderer
    from vllm.tokenizers.registry import cached_tokenizer_from_config

    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
        max_model_len=512,
    )
    vllm_config = VllmConfig(model_config=model_config)
    renderer = HfRenderer(vllm_config, cached_tokenizer_from_config(model_config))
    processor = ColBERTTokenEmbedIOProcessor(
        vllm_config,
        renderer,
        ChatTemplateConfig(),
    )
    proxy = PoolingCompletionRequest(
        model=None,
        input="",
        add_special_tokens=True,
    )
    tok_params = proxy.build_tok_params(model_config)

    items: list[tuple[str, str]] = [
        ("query", "What is ML?"),
        ("document", "Machine learning is a field of study."),
        ("query", "What is ML?"),
        ("document", "Deep learning uses neural networks."),
        ("query", "What is DL?"),
    ]
    prompts = [text for _, text in items]
    modes = [mode for mode, _ in items]

    batch_inputs = processor._render_colbert_cmpl_batch(
        proxy, prompts, modes, tok_params=tok_params
    )
    assert len(batch_inputs) == len(items)

    singles = [
        processor._render_colbert_cmpl(proxy, [text], mode, tok_params=tok_params)[0]
        for mode, text in items
    ]

    for i, (batched, single) in enumerate(zip(batch_inputs, singles)):
        assert batched["prompt_token_ids"] == single["prompt_token_ids"], (
            f"token ids differ at index {i} (batch vs single)"
        )
        assert batched.get("colbert_cache_key") == single.get("colbert_cache_key"), (
            f"cache key differ at index {i}"
        )

    assert batch_inputs[0]["prompt_token_ids"] == batch_inputs[2]["prompt_token_ids"]
    assert batch_inputs[0]["colbert_cache_key"] == batch_inputs[2]["colbert_cache_key"]

    perm = [4, 1, 0, 3, 2]
    perm_prompts = [prompts[i] for i in perm]
    perm_modes = [modes[i] for i in perm]
    perm_batch = processor._render_colbert_cmpl_batch(
        proxy, perm_prompts, perm_modes, tok_params=tok_params
    )
    for out_j, orig_i in enumerate(perm):
        perm_ids = perm_batch[out_j]["prompt_token_ids"]
        single_ids = singles[orig_i]["prompt_token_ids"]
        assert perm_ids == single_ids
        perm_key = perm_batch[out_j]["colbert_cache_key"]
        single_key = singles[orig_i]["colbert_cache_key"]
        assert perm_key == single_key


def test_colbert_cache_key_uses_template_hash():
    text = QUERY_TEXT
    template_hash = resolve_colbert_chat_template_hash("my-template")
    assert len(template_hash) == 64
    from vllm.config import ModelConfig
    from vllm.tokenizers.registry import cached_tokenizer_from_config

    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    tokenizer = cached_tokenizer_from_config(model_config)
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    config_hash = compute_colbert_config_hash(cfg, model_config=model_config)
    base_kwargs = dict(
        model_name=JINA_MODEL,
        raw_text=text,
        chat_template_hash=template_hash,
        colbert_config_hash=config_hash,
    )
    q_key = compute_colbert_cache_key(embedding_mode="query", **base_kwargs)
    d_key = compute_colbert_cache_key(embedding_mode="document", **base_kwargs)
    same_key_again = compute_colbert_cache_key(embedding_mode="query", **base_kwargs)
    other_hash = resolve_colbert_chat_template_hash("other-template")
    other_template = compute_colbert_cache_key(
        embedding_mode="query",
        model_name=JINA_MODEL,
        raw_text=text,
        chat_template_hash=other_hash,
        colbert_config_hash=config_hash,
    )
    assert q_key != d_key
    assert q_key == same_key_again
    assert q_key != other_template

    assert (
        compute_colbert_cache_salt(
            "query",
            text,
            model_name=JINA_MODEL,
            chat_template_hash=template_hash,
            colbert_config_hash=config_hash,
        )
        == q_key
    )


def test_colbert_token_truncation_query_shorter_than_document():
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
        max_model_len=512,
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    encoder = ColBERTEncoder(cfg)
    long_text = "word " * 5000
    token_ids = tokenizer.encode(long_text, add_special_tokens=True)
    q_ids = encoder.encode_token_ids(token_ids, "query")
    d_ids = encoder.encode_token_ids(token_ids, "document")
    assert len(q_ids) <= cfg.query_length
    assert len(d_ids) <= cfg.document_length
    assert len(q_ids) < len(d_ids)


def test_colbert_engine_input_finalize_is_immutable():
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)

    base = {
        "type": "token",
        "prompt_token_ids": [1, 2, 3],
        "cache_salt": "shared",
    }
    a = finalize_colbert_engine_input(
        base, mode="query", cache_key="salt-a", config=cfg
    )
    b = finalize_colbert_engine_input(
        base, mode="document", cache_key="salt-b", config=cfg
    )
    assert a["colbert_embedding_mode"] == "query"
    assert b["colbert_embedding_mode"] == "document"
    assert a["cache_salt"] == "salt-a"
    assert a["colbert_cache_key"] == "salt-a"
    assert b["cache_salt"] == "salt-b"
    assert a["colbert_attention_mask"] == build_colbert_attention_mask(
        [1, 2, 3], "query", cfg
    )
    a["prompt_token_ids"].append(99)
    assert base["prompt_token_ids"] == [1, 2, 3]


def test_colbert_query_attention_mask_all_ones():
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    token_ids = [0, 1, 2, cfg.mask_token_id, cfg.mask_token_id]
    mask = build_colbert_attention_mask(token_ids, "query", cfg)
    assert mask == [1] * len(token_ids)


def test_colbert_parity_check_env(monkeypatch):
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    monkeypatch.setenv("VLLM_COLBERT_PARITY_CHECK", "1")
    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    engine_inputs = [
        {
            "type": "token",
            "prompt_token_ids": [0] * 40,
            "colbert_embedding_mode": "query",
            "colbert_cache_key": "k1",
        }
    ]
    with pytest.raises(ValueError, match="query length"):
        validate_colbert_engine_inputs(engine_inputs, cfg)


def test_colbert_encoding_config_from_tokenizer():
    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    model_config = ModelConfig(
        model=JINA_MODEL,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    assert cfg.query_prefix == "[QueryMarker]" or cfg.query_prefix_id is not None
    assert (
        cfg.document_prefix == "[DocumentMarker]" or cfg.document_prefix_id is not None
    )


STRESS_BATCH = [
    ("query", "What is ML?"),
    ("document", "Machine learning is ..."),
    ("query", "What is DL?"),
    ("document", "Deep learning is ..."),
]


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_mixed_mode_stress_batch(
    vllm_runner,
    colbert_model: str,
):
    """Stress: interleaved query/doc batch — no cross-contamination."""
    texts = [text for _, text in STRESS_BATCH]
    modes = [mode for mode, _ in STRESS_BATCH]
    params = [
        _pooling_params(task="token_embed", embedding_mode=mode) for mode in modes
    ]

    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        llm = vllm_model.get_llm()

        singles = []
        for mode, text in STRESS_BATCH:
            singles.append(
                _colbert_encode(
                    llm,
                    [text],
                    _pooling_params(task="token_embed", embedding_mode=mode),
                )[0].outputs.data
            )

        batch_out = _colbert_encode(llm, texts, params)

        for _ in range(3):
            repeat = _colbert_encode(llm, texts, params)
            for i, (a, b) in enumerate(zip(batch_out, repeat)):
                ta = torch.as_tensor(a.outputs.data).float()
                tb = torch.as_tensor(b.outputs.data).float()
                torch.testing.assert_close(
                    ta,
                    tb,
                    rtol=COLBERT_GPU_EMBED_RTOL,
                    atol=COLBERT_GPU_EMBED_ATOL,
                    msg=f"deterministic replay failed index {i}",
                )

    from transformers import AutoTokenizer

    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model=colbert_model,
        runner="pooling",
        trust_remote_code=True,
        hf_overrides=JINA_OVERRIDES["hf_overrides"],
    )
    tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    cfg = ColBERTEncodingConfig.from_model_config(model_config, tokenizer)
    chat_hash = resolve_colbert_chat_template_hash(None, model_config=model_config)
    config_hash = compute_colbert_config_hash(cfg, model_config=model_config)
    cache_keys = [
        compute_colbert_cache_key(
            model_name=colbert_model,
            embedding_mode=mode,
            raw_text=text,
            chat_template_hash=chat_hash,
            colbert_config_hash=config_hash,
        )
        for mode, text in STRESS_BATCH
    ]
    assert len(cache_keys) == len(set(cache_keys))

    for i, (single, batched) in enumerate(zip(singles, batch_out)):
        ref = torch.as_tensor(single).float()
        got = torch.as_tensor(batched.outputs.data).float()
        torch.testing.assert_close(
            ref,
            got,
            rtol=COLBERT_GPU_EMBED_RTOL,
            atol=COLBERT_GPU_EMBED_ATOL,
            msg=f"batch vs single index {i}",
        )


RANKING_QUERY = "What is machine learning?"
RANKING_DOCS = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks with many layers.",
]


@pytest.mark.parametrize("colbert_model", [JINA_MODEL])
def test_colbert_pylate_ranking_parity(vllm_runner, colbert_model: str):
    """CI correctness gate: vLLM /score top-k order must match PyLate exactly.

    Uses strict list equality on rankings only (no cosine checks, no score tolerance).
    """
    pylate_models = _import_pylate_models()

    with vllm_runner(
        colbert_model,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        **JINA_OVERRIDES,
    ) as vllm_model:
        vllm_scores = vllm_model.score(RANKING_QUERY, RANKING_DOCS)

    assert len(vllm_scores) == len(RANKING_DOCS)
    vllm_ranking = sorted(
        range(len(RANKING_DOCS)), key=lambda i: vllm_scores[i], reverse=True
    )

    pylate_model = pylate_models.ColBERT(
        model_name_or_path=colbert_model,
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    )
    q_emb = pylate_model.encode([RANKING_QUERY], is_query=True, convert_to_tensor=True)[
        0
    ].float()
    pylate_scores = []
    from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score

    for doc in RANKING_DOCS:
        d_emb = pylate_model.encode([doc], is_query=False, convert_to_tensor=True)[
            0
        ].float()
        pylate_scores.append(compute_maxsim_score(q_emb, d_emb).item())

    pylate_ranking = sorted(
        range(len(RANKING_DOCS)), key=lambda i: pylate_scores[i], reverse=True
    )

    assert list(vllm_ranking) == list(pylate_ranking), (
        f"vLLM ranking {vllm_ranking} != PyLate ranking {pylate_ranking}; "
        f"vLLM scores={vllm_scores}, PyLate scores={pylate_scores}"
    )
    assert torch.equal(
        torch.tensor(vllm_ranking),
        torch.tensor(pylate_ranking),
    )

    k = min(5, len(RANKING_DOCS))
    assert vllm_ranking[:k] == pylate_ranking[:k]
