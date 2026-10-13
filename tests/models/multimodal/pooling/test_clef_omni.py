# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Clef-Omni pooler against a port of the checkpoint's joint schema head,
which loops over questions where the pooler batches them."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.clef_omni import JointSchemaHead, JointSchemaPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

HIDDEN, VOCAB = 64, 500
TYPES = [0, 1, 1, 2]
COUNTS = [2, 3, 1, 5]


def reference_logits(head, hidden, token_ids, embedding, question_spans, spans):
    """``JointSchemaHead.forward`` of the checkpoint's joint_schema_model.py
    for one record, with ``spans`` the option spans of each question."""
    hidden = head.hidden_norm(hidden)
    memory = head.memory_projection(hidden).unsqueeze(0)
    global_vector = hidden[-1]
    questions = torch.stack([hidden[s:e].mean(0) for s, e in question_spans])
    contexts = [torch.stack([hidden[s:e].mean(0) for s, e in q]) for q in spans]
    lexicals = [
        torch.stack([embedding[token_ids[s:e]].mean(0) for s, e in q]) for q in spans
    ]
    options = torch.cat(
        [
            head.option_context_projection(c)
            + head.option_lexical_projection(lex)
            + head.option_question_projection(questions[i]).unsqueeze(0)
            for i, (c, lex) in enumerate(zip(contexts, lexicals))
        ]
    ).unsqueeze(0)
    for layer in head.evidence_layers:
        options = layer(options, memory)
    routed = torch.split(options[0], [len(q) for q in spans])

    base = head.question_projection(questions)
    summaries = []
    for field, opts in zip(base, routed):
        weights = torch.softmax(opts @ field / math.sqrt(opts.shape[-1]), dim=0)
        summaries.append((weights[:, None] * opts).sum(0))
    fields = (
        base
        + head.option_summary_norm(torch.stack(summaries))
        + head.global_projection(global_vector)
        + head.type_embedding(torch.tensor(TYPES))
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    fields = head.field_norm(fields[0])

    logits = []
    for i, (field, lexical, opts) in enumerate(zip(fields, lexicals, routed)):
        anchor = F.normalize(questions[i] + global_vector, dim=-1)
        prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * (F.normalize(lexical, dim=-1) @ anchor)
        opts = head.option_norm(opts)
        field = field.expand_as(opts)
        features = torch.cat([field, opts, field * opts, (field - opts).abs()], -1)
        joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        joint = joint_scale * F.cosine_similarity(field, opts, dim=-1)
        joint = joint + head.residual_scorer(features).squeeze(-1)
        logits.append(prior + torch.sigmoid(head.residual_gate) * joint)
    return logits


@pytest.fixture
def pooler():
    torch.manual_seed(0)
    head = JointSchemaHead(
        hidden_size=HIDDEN,
        width=32,
        routing_layers=2,
        layers=2,
        heads=4,
        feedforward=48,
    ).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    embedding = torch.randn(VOCAB, HIDDEN)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False, async_scheduling=False
        )
    )
    with patch(
        "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
        return_value=config,
    ):
        pooler = JointSchemaPooler(head, lambda ids: embedding[ids])
    return pooler, embedding


def schema(length: int):
    """Questions after a 20-token state, with options of 1 to 4 tokens."""
    pos = 20
    question_spans: list[tuple[int, int]] = []
    spans: list[list[tuple[int, int]]] = []
    for count in COUNTS:
        question_spans.append((pos, pos + 4))
        pos += 6
        spans.append([])
        for k in range(count):
            n = 1 + (k * 3) % 4
            spans[-1].append((pos, pos + n))
            pos += n + 1
    assert pos < length
    extra = {
        "question_types": TYPES,
        "question_spans": [[s - length, e - length] for s, e in question_spans],
        "option_spans": [[s - length, e - length] for q in spans for s, e in q],
        "option_counts": COUNTS,
    }
    return {"joint_schema": extra}, question_spans, spans


@torch.inference_mode()
def test_pooler_matches_reference(pooler):
    pooler, embedding = pooler
    lengths = [120, 90, 100]
    token_ids = [torch.randint(0, VOCAB, (n,)) for n in lengths]
    hidden = torch.randn(sum(lengths), HIDDEN)
    first, first_questions, first_spans = schema(lengths[0])
    last, last_questions, last_spans = schema(lengths[2])
    params = [
        PoolingParams(task="token_classify", use_activation=True, extra_kwargs=first),
        # A prompt without a schema, like the profiling run's.
        PoolingParams(task="token_classify", use_activation=True),
        PoolingParams(task="token_classify", use_activation=False, extra_kwargs=last),
    ]
    padded = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for i, ids in enumerate(token_ids):
        padded[i, : len(ids)] = ids
    metadata = PoolingMetadata(
        prompt_lens=torch.tensor(lengths),
        prompt_token_ids=None,
        prompt_token_ids_cpu=padded,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in lengths],
    )
    metadata.build_pooling_cursor(
        torch.tensor(lengths).numpy(),
        seq_lens_cpu=torch.tensor(lengths),
        device=torch.device("cpu"),
    )
    probs, empty, logits = pooler(hidden, metadata)

    first_hidden, _, last_hidden = torch.split(hidden, lengths)
    expected = reference_logits(
        pooler.head, first_hidden, token_ids[0], embedding, first_questions, first_spans
    )
    expected_probs = torch.cat([q.softmax(-1) for q in expected])
    torch.testing.assert_close(probs, expected_probs, atol=1e-5, rtol=1e-4)
    assert empty.numel() == 0
    expected = reference_logits(
        pooler.head, last_hidden, token_ids[2], embedding, last_questions, last_spans
    )
    torch.testing.assert_close(logits, torch.cat(expected), atol=1e-5, rtol=1e-4)
