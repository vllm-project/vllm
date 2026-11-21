# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["vllm-project/sentiment-polish-gpt2-small"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for head_dtype_str in ["float32", "model"]:
        with vllm_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            hf_overrides={"head_dtype": head_dtype_str},
        ) as vllm_model:
            model_config = vllm_model.llm.llm_engine.model_config
            model_dtype = model_config.dtype
            head_dtype = model_config.head_dtype

            if head_dtype_str == "float32":
                assert head_dtype == torch.float32
            elif head_dtype_str == "model":
                assert head_dtype == model_dtype

            vllm_outputs = vllm_model.classify(example_prompts)

        for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
            hf_output = torch.tensor(hf_output).float()
            vllm_output = torch.tensor(vllm_output).float()

            assert torch.allclose(hf_output, vllm_output, atol=1e-2)


class MLMHeadDtypeInspectionWorker:
    model_runner: Any

    def inspect_mlm_head_dtypes(self):
        return sorted(
            {
                str(parameter.dtype)
                for parameter in self.model_runner.model.mlm_head.parameters()
            }
        )


@pytest.mark.parametrize(
    "architecture", ["BertForMaskedLM", "BertSpladeSparseEmbeddingModel"]
)
@pytest.mark.parametrize(
    "model_dtype,head_dtype",
    [
        ("float32", None),
        ("float32", "float32"),
        ("float32", "model"),
        ("bfloat16", None),
        ("bfloat16", "float32"),
        ("bfloat16", "model"),
        ("float32", "bfloat16"),
    ],
)
def test_mlm_head_dtype_controls_saved_weights_and_outputs(
    tmp_path, vllm_runner, architecture, model_dtype, head_dtype
):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForMaskedLM, TokenizersBackend

    from vllm import PoolingParams

    config = BertConfig(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(89)
        reference = BertForMaskedLM(config).eval()
        with torch.no_grad():
            reference.cls.predictions.decoder.weight.zero_()
            bias = torch.linspace(9999.5, 10001.5, 16)
            reference.cls.predictions.bias.copy_(bias)
            reference.cls.predictions.decoder.bias.copy_(bias)
    config.save_pretrained(tmp_path)
    state = {
        name: value.detach().clone().contiguous()
        for name, value in reference.state_dict().items()
    }
    # Explicit decoder names isolate the dtype contract from tied checkpoint aliases.
    if architecture == "BertSpladeSparseEmbeddingModel":
        del state["cls.predictions.bias"]
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})

    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "alpha": 4, "beta": 5}
    vocabulary.update({f"unused{i}": i for i in range(6, 16)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(tmp_path)
    head_dtype_name = model_dtype if head_dtype == "model" else head_dtype or "float32"
    expected_dtype = getattr(torch, head_dtype_name)
    expected_bias = reference.cls.predictions.decoder.bias.detach().to(expected_dtype)
    prompts = ["alpha", "beta alpha beta"]
    overrides = {"architectures": [architecture]}
    if head_dtype is not None:
        overrides["head_dtype"] = head_dtype
    task = "token_classify" if architecture == "BertForMaskedLM" else "embed"

    with vllm_runner(
        str(tmp_path),
        runner="pooling",
        dtype=model_dtype,
        hf_overrides=overrides,
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=False,
        enforce_eager=True,
        worker_extension_cls=(
            "tests.models.language.pooling.test_head_dtype.MLMHeadDtypeInspectionWorker"
        ),
    ) as model:
        outputs = model.llm.encode(
            prompts,
            pooling_task=task,
            pooling_params=PoolingParams(use_activation=False),
            use_tqdm=False,
        )
        actual_head_dtypes = model.llm.collective_rpc("inspect_mlm_head_dtypes")

    for prompt, output in zip(prompts, outputs):
        if task == "token_classify":
            target = expected_bias.expand(len(tokenizer.encode(prompt)), -1).float()
        else:
            target = expected_bias.relu().log1p().float()
        torch.testing.assert_close(
            torch.as_tensor(output.outputs.data).float(), target, rtol=0, atol=0
        )
    assert actual_head_dtypes == [[str(expected_dtype)]]
