# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import numpy as np
import pybase64
import pytest
import torch

from vllm.multimodal.media.image import ImageEmbeddingMediaIO
from vllm.multimodal.media.video import VideoEmbeddingMediaIO


def _load_array(handler, array, route, tmp_path):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    if route == "bytes":
        result = handler.load_bytes(payload)
    elif route == "base64":
        result = handler.load_base64("", pybase64.b64encode(payload).decode("ascii"))
    else:
        path = tmp_path / "embeddings.npy"
        path.write_bytes(payload)
        result = handler.load_file(path)
        assert path.read_bytes() == payload
    return result


@pytest.mark.parametrize("handler_type", [ImageEmbeddingMediaIO, VideoEmbeddingMediaIO])
@pytest.mark.parametrize("route", ["bytes", "base64", "file"])
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
@pytest.mark.parametrize("byteorder", ["=", ">"])
def test_numpy_embedding_values_preserve_byteorder(
    handler_type, route, dtype, byteorder, tmp_path
):
    # np.save must preserve logical values of transposed/strided arrays too.
    values = np.arange(120, dtype=np.float64).reshape(4, 5, 6)
    values = (values / 8 - 3).astype(np.dtype(dtype).newbyteorder(byteorder))
    view = values.transpose(2, 0, 1)[::2]
    original = view.copy()
    expected = torch.tensor(
        view.tolist(),
        dtype=getattr(
            torch, {"f2": "float16", "f4": "float32", "f8": "float64"}[dtype]
        ),
    )

    actual = _load_array(handler_type(), view, route, tmp_path)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(view, original)


@pytest.mark.parametrize("handler_type", [ImageEmbeddingMediaIO, VideoEmbeddingMediaIO])
@pytest.mark.parametrize("route", ["bytes", "base64", "file"])
def test_numpy_embedding_metadata_preserves_large_integers(
    handler_type, route, tmp_path
):
    values = [[2**53 + 1, -(2**53) - 3], [2**63 - 1, -(2**63)]]
    array = np.array(values, dtype=">i8")
    actual = _load_array(handler_type(), array, route, tmp_path)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, torch.tensor(values), rtol=0, atol=0)


@pytest.mark.parametrize("handler_type", [ImageEmbeddingMediaIO, VideoEmbeddingMediaIO])
@pytest.mark.parametrize("byteorder", ["=", ">"])
def test_numpy_embedding_empty_shape(handler_type, byteorder, tmp_path):
    array = np.empty((2, 0, 4), dtype=np.dtype("f4").newbyteorder(byteorder))
    actual = _load_array(handler_type(), array, "bytes", tmp_path)
    assert actual.shape == (2, 0, 4)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("handler_type", [ImageEmbeddingMediaIO, VideoEmbeddingMediaIO])
def test_numpy_embedding_rejects_object_arrays(handler_type):
    buffer = io.BytesIO()
    np.save(buffer, np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        handler_type().load_bytes(buffer.getvalue())


@pytest.mark.parametrize("byteorder", ["=", ">"])
def test_numpy_image_embedding_is_consumed_by_native_llava(
    tmp_path, vllm_runner, byteorder
):
    from safetensors.torch import load_file, save_file
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        TokenizersBackend,
    )

    from vllm import SamplingParams

    text_config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        image_size=32,
        patch_size=16,
    )
    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=31,
        image_seq_length=4,
    )
    model_path = tmp_path / "model"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        LlavaForConditionalGeneration(config).eval().save_pretrained(model_path)

    # Keep the saved fixture in the CLIP wrapper format accepted by this model.
    checkpoint = model_path / "model.safetensors"
    state = {}
    for name, tensor in load_file(checkpoint).items():
        for prefix in ("model.vision_tower.", "vision_tower."):
            if name.startswith(prefix) and not name.startswith(
                prefix + "vision_model."
            ):
                name = prefix + "vision_model." + name[len(prefix) :]
                break
        state[name] = tensor
    save_file(state, checkpoint, metadata={"format": "pt"})
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "alpha": 4,
        "beta": 5,
        "<image>": 31,
    }
    vocabulary.update({f"unused{i}": i for i in range(6, 31)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        additional_special_tokens=[AddedToken("<image>", special=True)],
    )
    processor = LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 32},
            crop_size={"height": 32, "width": 32},
        ),
        tokenizer=tokenizer,
        patch_size=16,
        num_additional_image_tokens=1,
        vision_feature_select_strategy="default",
    )
    processor.save_pretrained(model_path)

    values = (
        np.linspace(-0.5, 0.5, 256)
        .reshape(4, 64)
        .astype(np.dtype("f4").newbyteorder(byteorder))
    )
    loaded = _load_array(ImageEmbeddingMediaIO(), values, "base64", tmp_path)
    direct = torch.tensor(values.tolist(), dtype=torch.float32)
    torch.testing.assert_close(loaded, direct, rtol=0, atol=0)
    with vllm_runner(
        str(model_path),
        dtype="float32",
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=False,
        enable_mm_embeds=True,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        max_logprobs=32,
    ) as model:
        results = model.llm.generate(
            [
                {
                    "prompt": "<image> alpha beta",
                    "multi_modal_data": {"image": direct.unsqueeze(0)},
                },
                {
                    "prompt": "<image> alpha beta",
                    "multi_modal_data": {"image": loaded.unsqueeze(0)},
                },
            ],
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, logprobs=-1),
            use_tqdm=False,
        )
    reference, actual = [result.outputs[0] for result in results]
    assert actual.token_ids == reference.token_ids
    assert len(actual.token_ids) == 2
    for observed, expected in zip(actual.logprobs, reference.logprobs):
        assert observed.keys() == expected.keys()
        assert [value.logprob for value in observed.values()] == pytest.approx(
            [value.logprob for value in expected.values()], rel=1e-5, abs=1e-5
        )
