# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for modular Sentence Transformers CrossEncoders."""

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from vllm.entrypoints.pooling.scoring.typing import ScoreInput, ScoreMultiModalParam


def _create_modular_bert_cross_encoder(
    path: Path,
    *,
    structured: bool = False,
    pooling_mode: str = "mean",
) -> tuple[str, list[tuple[str, str]], list[float]]:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.sentence_transformer.modules import (
        Dense,
        Pooling,
        Transformer,
    )

    base_path = path / "base"
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "query",
                "document",
                ":",
                ";",
                "0",
                "1",
                "2",
                "3",
                "4",
            ]
        )
    }
    config = BertConfig(
        architectures=["BertModel"],
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=len(vocab),
    )
    torch.manual_seed(0)
    bert = BertModel(config)
    with torch.no_grad():
        token_types = bert.embeddings.token_type_embeddings.weight
        token_types[0].zero_()
        token_types[1].copy_(torch.linspace(-1.0, 1.0, config.hidden_size))
    bert.save_pretrained(base_path)

    tokenizer = BertTokenizer(
        vocab=vocab,
        do_lower_case=False,
        model_max_length=16,
    )
    if structured:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }}:"
            "{% for item in message['content'] %}{{ item['text'] }}{% endfor %};"
            "{% endfor %}[SEP]"
        )
    tokenizer.save_pretrained(base_path)

    transformer = Transformer(
        str(base_path),
        max_seq_length=16,
        module_output_name="token_embeddings",
        modality_config={
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            "message": {
                "method": "forward",
                "method_output_name": "last_hidden_state",
                "format": "structured",
            },
        }
        if structured
        else None,
    )
    pooling = Pooling(
        config.hidden_size, pooling_mode=pooling_mode, include_prompt=True
    )
    dense = Dense(
        config.hidden_size,
        1,
        activation_function=torch.nn.Identity(),
        init_weight=torch.linspace(-0.5, 0.5, config.hidden_size).unsqueeze(0),
        init_bias=torch.tensor([0.1]),
        module_output_name="scores",
    )
    cross_encoder = CrossEncoder(
        modules=[transformer, pooling, dense],
        activation_fn=torch.nn.Identity(),
        device="cpu",
    )
    export_path = path / "export"
    cross_encoder.save_pretrained(export_path)

    document = " ".join(["document"] * 21)
    pairs = [
        ("query", document),
        ("query query query", document),
        ("query", "document"),
    ]
    reference_scores = cross_encoder.predict(pairs).tolist()
    return str(export_path), pairs, reference_scores


@pytest.mark.parametrize(
    ("model_impl", "enforce_eager"),
    [("vllm", True), ("transformers", True), ("transformers", False)],
)
def test_modular_bert_cross_encoder_score_parity(
    vllm_runner, tmp_path: Path, model_impl: str, enforce_eager: bool
) -> None:
    """A current modular BERT export must preserve pair and truncation semantics."""
    pytest.importorskip(
        "sentence_transformers",
        minversion="5.7.0",
        reason="Modular CrossEncoder construction requires sentence-transformers 5.7",
    )
    model_path, pairs, reference_scores = _create_modular_bert_cross_encoder(tmp_path)

    with vllm_runner(
        model_path,
        runner="pooling",
        model_impl=model_impl,
        trust_remote_code=False,
        max_model_len=None,
        dtype="float32",
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.1,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        compilation_config={"cudagraph_capture_sizes": [8, 16, 32]},
    ) as model:
        assert model.llm.llm_engine.model_config.max_model_len == 16
        # Replay equal-sized requests with different segment boundaries, then
        # exercise a shorter request whose graph input includes padding.
        for pair, expected_length, reference_score in zip(
            pairs, [16, 16, 5], reference_scores
        ):
            output = model.llm.score(*pair)[0]
            assert len(output.prompt_token_ids) == expected_length
            assert output.outputs.score == pytest.approx(
                reference_score, abs=1e-4, rel=1e-4
            )


@pytest.mark.parametrize("pooling_mode", ["cls", "mean", "lasttoken"])
def test_structured_cross_encoder_export_preserves_truncated_scores(
    vllm_runner, tmp_path: Path, pooling_mode: str
) -> None:
    """Saved structured exports retain the template tail used by the trained head."""
    pytest.importorskip("sentence_transformers", minversion="5.7.0")
    from sentence_transformers import CrossEncoder

    model_path, pairs, reference_scores = _create_modular_bert_cross_encoder(
        tmp_path, structured=True, pooling_mode=pooling_mode
    )
    reference = CrossEncoder(model_path, device="cpu", trust_remote_code=False)
    features = reference.preprocess(pairs)
    expected_ids = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(features["input_ids"], features["attention_mask"])
    ]
    with vllm_runner(
        model_path,
        trust_remote_code=False,
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        max_model_len=None,
    ) as model:
        outputs = model.llm.score(
            [query for query, _ in pairs], [document for _, document in pairs]
        )
    assert [output.prompt_token_ids for output in outputs] == expected_ids
    assert [output.outputs.score for output in outputs] == pytest.approx(
        reference_scores, abs=1e-4, rel=1e-4
    )


def _create_qwen_cross_encoder(
    path: Path,
    false_id: int | None,
    tied: bool,
    *,
    multimodal: bool = False,
    dense: bool = False,
) -> tuple[
    str,
    list[tuple[ScoreInput, ScoreInput]],
    list[float],
    list[float],
    list[list[int]],
]:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.modules import LogitScore, Transformer
    from sentence_transformers.sentence_transformer.modules import Dense, Pooling
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config, Qwen3ForCausalLM, TokenizersBackend

    base_path = path / "base"
    vision_tokens = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "query",
                "document",
                "no",
                "yes",
                "system",
                "assistant",
                ":",
                ";",
                "match",
                "extra",
            ]
            + vision_tokens
        )
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        model_max_length=64,
        model_input_names=["input_ids", "attention_mask"],
        additional_special_tokens=vision_tokens,
        chat_template=(
            "{% for message in messages %}{{ message['role'] }}:"
            "{% for item in message['content'] %}"
            "{% if item['type'] == 'image' %}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{ item['text'] }}{% endif %}{% endfor %};"
            "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
        ),
    )
    text_kwargs = dict(
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=len(vocab),
        max_position_embeddings=128,
    )
    torch.manual_seed(123)
    if multimodal:
        from transformers import (
            Qwen2VLImageProcessor,
            Qwen3VLConfig,
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
            Qwen3VLProcessor,
            Qwen3VLTextConfig,
            Qwen3VLVideoProcessor,
            Qwen3VLVisionConfig,
        )

        text_config = Qwen3VLTextConfig(
            **text_kwargs,
            is_causal=not dense,
            rope_parameters={
                "rope_type": "default",
                "mrope_section": [6, 5, 5],
                "mrope_interleaved": True,
            },
        )
        vision_config = Qwen3VLVisionConfig(
            depth=1,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            out_hidden_size=128,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        )
        vl_config = Qwen3VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=vocab["<|image_pad|>"],
            video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"],
            vision_end_token_id=vocab["<|vision_end|>"],
            tie_word_embeddings=tied,
        )
        model_cls = Qwen3VLModel if dense else Qwen3VLForConditionalGeneration
        model_cls(vl_config).save_pretrained(base_path)
        processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
            image_processor=Qwen2VLImageProcessor(
                patch_size=16,
                size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
            ),
            video_processor=Qwen3VLVideoProcessor(),
        )
        processor.save_pretrained(base_path)
    else:
        config = Qwen3Config(
            **text_kwargs, tie_word_embeddings=tied, pad_token_id=0, eos_token_id=2
        )
        Qwen3ForCausalLM(config).save_pretrained(base_path)
        tokenizer.save_pretrained(base_path)
    output_name = "last_hidden_state" if dense else "logits"
    transformer = Transformer(
        str(base_path),
        transformer_task=(
            "feature-extraction"
            if dense
            else "any-to-any"
            if multimodal
            else "text-generation"
        ),
        module_output_name="token_embeddings" if dense else "causal_logits",
        modality_config={
            "text": {"method": "forward", "method_output_name": output_name},
            **(
                {"image": {"method": "forward", "method_output_name": output_name}}
                if multimodal
                else {}
            ),
            "message": {
                "method": "forward",
                "method_output_name": output_name,
                "format": "structured",
            },
        },
        processing_kwargs={}
        if dense
        else {"chat_template": {"add_generation_prompt": True}},
    )
    head = (
        [
            Pooling(128, pooling_mode="mean", include_prompt=True),
            Dense(
                128, 1, activation_function=torch.nn.Tanh(), module_output_name="scores"
            ),
        ]
        if dense
        else [LogitScore(true_token_id=6, false_token_id=false_id)]
    )
    cross_encoder = CrossEncoder(
        modules=[transformer, *head],
        num_labels=1,
        device="cpu",
        prompts={} if dense else {"match": "match"},
        default_prompt_name=None if dense else "match",
    )
    export_path = path / "export"
    cross_encoder.save_pretrained(export_path)
    pairs = [
        ("query", "document"),
        ("query extra", "document extra extra"),
        ("query", " ".join(["document"] * 70)),
    ]
    if multimodal:
        from PIL import Image

        image = Image.new("RGB", (64, 64), color=(35, 100, 180))
        pairs = [("query", image), (image, "document")]
    scores = cross_encoder.predict(pairs).tolist()
    raw_scores = cross_encoder.predict(
        pairs, activation_fn=torch.nn.Identity()
    ).tolist()
    features = cross_encoder.preprocess(pairs, prompt=None if dense else "match")
    token_ids = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(
            features["input_ids"],
            features["attention_mask"],
        )
    ]
    if multimodal:
        from vllm.multimodal.utils import encode_image_url

        image_input: ScoreMultiModalParam = {
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_url(image)}}
            ]
        }
        score_pairs: list[tuple[ScoreInput, ScoreInput]] = [
            ("query", image_input),
            (image_input, "document"),
        ]
    else:
        score_pairs = [(query, document) for query, document in pairs]
    return str(export_path), score_pairs, scores, raw_scores, token_ids


@pytest.mark.parametrize(
    "false_id,tied,multimodal",
    [
        (None, False, False),
        (5, False, False),
        (None, True, False),
        (5, True, False),
        (5, False, True),
        (5, True, True),
    ],
)
def test_logit_score_matches_sentence_transformers_export(
    vllm_runner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    false_id: int | None,
    tied: bool,
    multimodal: bool,
) -> None:
    """Score real exports through both logit modes, including tied LM heads."""
    pytest.importorskip("sentence_transformers", minversion="5.7.0")
    # Compare checkpoint semantics without Triton's default TF32 attention rounding.
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    export = _create_qwen_cross_encoder(tmp_path, false_id, tied, multimodal=multimodal)
    _check_export_scores(vllm_runner, export, multimodal=multimodal)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_multimodal_dense_matches_sentence_transformers_export(
    vllm_runner, tmp_path: Path, monkeypatch, enforce_eager: bool
) -> None:
    """A bidirectional VLM + mean Pooling + Dense needs no remote model code."""
    pytest.importorskip("sentence_transformers", minversion="5.7.0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    export = _create_qwen_cross_encoder(
        tmp_path, None, False, multimodal=True, dense=True
    )
    _check_export_scores(
        vllm_runner,
        export,
        multimodal=True,
        model_impl="transformers",
        enforce_eager=enforce_eager,
    )


def _check_export_scores(vllm_runner, export, *, multimodal, **kwargs):
    from vllm import PoolingParams

    model_path, pairs, expected, expected_raw, token_ids = export
    with vllm_runner(
        model_path,
        dtype="float32",
        trust_remote_code=False,
        enforce_eager=kwargs.pop("enforce_eager", True),
        gpu_memory_utilization=0.1,
        max_model_len=None,
        limit_mm_per_prompt={"image": 1, "video": 0} if multimodal else None,
        **kwargs,
    ) as model:
        results = model.llm.score(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        )
        assert [result.prompt_token_ids for result in results] == token_ids
        actual = [result.outputs.score for result in results]
        raw_results = model.llm.score(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            pooling_params=PoolingParams(use_activation=False),
        )
        actual_raw = [result.outputs.score for result in raw_results]
    assert actual == pytest.approx(expected, abs=1e-4, rel=1e-4)
    assert actual_raw == pytest.approx(expected_raw, abs=1e-4, rel=1e-4)
