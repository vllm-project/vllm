# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColBERT late interaction scoring.

Tests are parametrized across multiple ColBERT backbones to ensure the
generic ColBERT support works with different encoder architectures.
"""

from contextlib import contextmanager

import pytest
import torch

from tests.utils import wait_for_memory_to_settle
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score

# -----------------------------------------------------------------------
# Model definitions: (model_name, colbert_dim, extra vllm_runner kwargs)
# -----------------------------------------------------------------------
COLBERT_MODELS = {
    "bert": {
        "model": "answerdotai/answerai-colbert-small-v1",
        "colbert_dim": 96,
        "max_model_len": 512,
        "extra_kwargs": {},
        "hf_comparison": {
            "weights_file": "model.safetensors",
            "weights_key": "linear.weight",
            "trust_remote_code": False,
            "model_cls": "BertModel",
        },
    },
    "modernbert": {
        "model": "lightonai/GTE-ModernColBERT-v1",
        "colbert_dim": 128,
        "max_model_len": 299,
        "extra_kwargs": {
            "hf_overrides": {
                "architectures": ["ColBERTModernBertModel"],
            },
        },
        "hf_comparison": {
            "weights_file": "1_Dense/model.safetensors",
            "weights_key": "linear.weight",
            "trust_remote_code": False,
            "model_cls": "AutoModel",
        },
    },
    "jina": {
        "model": "jinaai/jina-colbert-v2",
        "colbert_dim": 128,
        "max_model_len": 8192,
        "extra_kwargs": {
            "hf_overrides": {
                "architectures": ["ColBERTJinaRobertaModel"],
            },
        },
        "hf_comparison": {
            "weights_file": "model.safetensors",
            "weights_key": "linear.weight",
            "trust_remote_code": True,
            "model_cls": "AutoModel",
        },
    },
    "lfm2": {
        "model": "LiquidAI/LFM2-ColBERT-350M",
        "colbert_dim": 128,
        "max_model_len": 511,
        "extra_kwargs": {
            "hf_overrides": {
                "architectures": ["ColBERTLfm2Model"],
            },
        },
        "hf_comparison": {
            "weights_file": "1_Dense/model.safetensors",
            "weights_key": "linear.weight",
            "trust_remote_code": False,
            "model_cls": "AutoModel",
        },
    },
}


TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "half"


def _load_hf_model(model_name: str, hf_spec: dict, device: torch.device):
    """Load HF model on the given device with a compatible attention impl."""
    from transformers import AutoModel, BertModel

    cls = BertModel if hf_spec["model_cls"] == "BertModel" else AutoModel
    trust = hf_spec.get("trust_remote_code", False)

    # Flash / Triton kernels require GPU tensors; fall back to eager on CPU.
    extra = {}
    if device.type == "cpu":
        extra["attn_implementation"] = "eager"

    model = cls.from_pretrained(
        model_name,
        trust_remote_code=trust,
        **extra,
    ).to(device)
    model.eval()

    # Transformers 5.0 weight materialization can clear non-persistent
    # buffers (e.g. rotary inv_freq) that were registered with
    # persistent=False.  Re-compute them so the model produces valid output.
    for mod in model.modules():
        if hasattr(mod, "_compute_inv_freq") and hasattr(mod, "inv_freq"):
            mod.inv_freq = mod._compute_inv_freq(device=device)

    return model


def _load_projection_weight(model_name: str, hf_spec: dict, device: torch.device):
    """Download and return the ColBERT linear projection weight."""
    from safetensors.torch import load_file

    from vllm.transformers_utils.repo_utils import hf_api, with_retry

    path = with_retry(
        lambda: hf_api().hf_hub_download(model_name, filename=hf_spec["weights_file"]),
        f"Error downloading {hf_spec['weights_file']} from {model_name}",
    )
    weights = load_file(path)
    return weights[hf_spec["weights_key"]].to(device)


def _compute_hf_colbert_embeddings(model, tokenizer, linear_weight, texts, device):
    """Run HF model + projection and return L2-normalised token embeddings."""
    import torch.nn.functional as F

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state.float()
            projected = F.linear(hidden, linear_weight.float())
            normalised = F.normalize(projected, p=2, dim=-1)
            embeddings.append(normalised.squeeze(0).cpu())
    return embeddings


@contextmanager
def _hf_colbert_model(model_name: str, hf_spec: dict, device: torch.device):
    """Load the HF backbone + ColBERT projection, freeing the GPU on exit.

    These live outside any runner context, so without explicit cleanup ROCm
    keeps the VRAM resident and the next backend parametrization (or test)
    OOMs on startup.
    """
    hf_model = _load_hf_model(model_name, hf_spec, device)
    linear_weight = _load_projection_weight(model_name, hf_spec, device)
    try:
        yield hf_model, linear_weight
    finally:
        del hf_model, linear_weight
        cleanup_dist_env_and_memory()
        wait_for_memory_to_settle()


def _assert_embeddings_close(vllm_outputs, hf_embeddings):
    """Assert that vLLM and HuggingFace embeddings match."""
    for i, (hf_emb, vllm_out) in enumerate(zip(hf_embeddings, vllm_outputs)):
        vllm_emb = torch.as_tensor(vllm_out).float()

        assert hf_emb.shape == vllm_emb.shape, (
            f"Shape mismatch for text {i}: HF {hf_emb.shape} vs vLLM {vllm_emb.shape}"
        )

        torch.testing.assert_close(
            vllm_emb,
            hf_emb,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Embedding mismatch for text {i}",
        )


@pytest.fixture(params=list(COLBERT_MODELS.keys()), scope="class")
def colbert_spec(request):
    """Return the model spec dict for the current parametrization."""
    return COLBERT_MODELS[request.param]


@pytest.fixture(scope="class")
def colbert_model_name(colbert_spec):
    return colbert_spec["model"]


@pytest.fixture(scope="class")
def colbert_dim(colbert_spec):
    return colbert_spec["colbert_dim"]


@pytest.fixture(scope="class")
def colbert_max_model_len(colbert_spec):
    return colbert_spec["max_model_len"]


@pytest.fixture(scope="class")
def colbert_extra_kwargs(colbert_spec):
    return colbert_spec["extra_kwargs"]


@pytest.fixture(scope="class")
def colbert_model(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        yield vllm_model


class TestColbertSharedEngine:
    """Tests sharing one engine per model.

    Class-scoped so the engine is released before `test_colbert_hf_comparison`,
    which needs a runner of its own and cannot start while this one holds VRAM.
    """

    def test_colbert_token_embed(self, colbert_model, colbert_dim):
        """Test that ColBERT model produces token embeddings."""
        outputs = colbert_model.token_embed([TEXTS_1[0]])

        assert len(outputs) == 1
        emb = torch.as_tensor(outputs[0])
        assert emb.dim() == 2
        assert emb.shape[1] == colbert_dim
        assert emb.shape[0] > 1

    def test_colbert_late_interaction_1_to_1(self, colbert_model):
        """Test ColBERT late interaction scoring with 1:1 query-document pair."""
        q_outputs = colbert_model.token_embed([TEXTS_1[0]])
        d_outputs = colbert_model.token_embed([TEXTS_2[0]])

        q_emb = torch.as_tensor(q_outputs[0])
        d_emb = torch.as_tensor(d_outputs[0])

        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        vllm_scores = colbert_model.score(TEXTS_1[0], TEXTS_2[0])

        assert len(vllm_scores) == 1
        assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)

    def test_colbert_late_interaction_1_to_N(self, colbert_model):
        """Test ColBERT late interaction scoring with 1:N query-documents."""
        q_outputs = colbert_model.token_embed([TEXTS_1[0]])
        d_outputs = colbert_model.token_embed(TEXTS_2)

        q_emb = torch.as_tensor(q_outputs[0])

        manual_scores = []
        for d_out in d_outputs:
            d_emb = torch.as_tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        vllm_scores = colbert_model.score(TEXTS_1[0], TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)

    def test_colbert_late_interaction_N_to_N(self, colbert_model):
        """Test ColBERT late interaction scoring with N:N query-documents."""
        q_outputs = colbert_model.token_embed(TEXTS_1)
        d_outputs = colbert_model.token_embed(TEXTS_2)

        manual_scores = []
        for q_out, d_out in zip(q_outputs, d_outputs):
            q_emb = torch.as_tensor(q_out)
            d_emb = torch.as_tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        vllm_scores = colbert_model.score(TEXTS_1, TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)

    def test_colbert_relevance_ordering(self, colbert_model):
        """Test that ColBERT scores relevant documents higher than irrelevant."""
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
        ]

        scores = colbert_model.score(query, documents)

        assert len(scores) == 3
        assert scores[0] > scores[1], "ML doc should score higher than Python doc"
        assert scores[2] > scores[1], "DL doc should score higher than Python doc"

    def test_colbert_embed_task_not_supported(self, colbert_model):
        """Test that ColBERT model does not support the embed task."""
        with pytest.raises(ValueError, match="Embedding API is not supported"):
            colbert_model.llm.encode([TEXTS_1[0]], pooling_task="embed")


@pytest.mark.parametrize("backend", ["bert", "modernbert", "jina", "lfm2"])
def test_colbert_hf_comparison(vllm_runner, backend):
    """Test that vLLM ColBERT embeddings match HuggingFace for each backend."""
    from transformers import AutoTokenizer

    spec = COLBERT_MODELS[backend]
    hf_spec = spec["hf_comparison"]
    extra_kwargs = spec["extra_kwargs"]
    model_name = spec["model"]
    assert isinstance(model_name, str)
    assert isinstance(hf_spec, dict)
    assert isinstance(extra_kwargs, dict)
    test_texts = [TEXTS_1[0], TEXTS_2[0]]

    with vllm_runner(
        model_name,
        runner="pooling",
        dtype="float32",
        max_model_len=spec["max_model_len"],
        enforce_eager=True,
        **extra_kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(test_texts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=hf_spec.get("trust_remote_code", False),
    )
    with _hf_colbert_model(model_name, hf_spec, device) as (hf_model, linear_weight):
        hf_embeddings = _compute_hf_colbert_embeddings(
            hf_model,
            hf_tokenizer,
            linear_weight,
            test_texts,
            device,
        )

    _assert_embeddings_close(vllm_outputs, hf_embeddings)


@pytest.mark.parametrize(
    "dimension_field", [None, "colbert_dim", "dim", "projection_dim"]
)
@pytest.mark.parametrize("projection_name", ["linear.weight", "colbert_linear.weight"])
def test_colbert_configured_projection(
    tmp_path, vllm_runner, dimension_field, projection_name
):
    """Load configured and inferred projection widths through the public LLM API."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, TokenizersBackend

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        config = BertConfig(
            vocab_size=16,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
            pad_token_id=0,
        )
        reference = BertModel(config, add_pooling_layer=False).eval()
        projection = torch.randn(8, 64) * 0.1

    config.architectures = ["HF_ColBERT"]
    if dimension_field is not None:
        setattr(config, dimension_field, projection.shape[0])
    config.save_pretrained(tmp_path)
    state = {
        "bert." + name: value.contiguous()
        for name, value in reference.state_dict().items()
    }
    state[projection_name] = projection
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
    }
    vocabulary.update({f"unused{i}": i for i in range(7, 16)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(tmp_path)
    texts = ["alpha beta", "gamma"]
    expected = _compute_hf_colbert_embeddings(
        reference, tokenizer, projection, texts, torch.device("cpu")
    )
    expected_score = (expected[0] @ expected[1].T).amax(-1).sum()

    with vllm_runner(
        str(tmp_path),
        runner="pooling",
        dtype="float32",
        max_model_len=64,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=False,
        enforce_eager=True,
    ) as model:
        actual = model.token_embed(texts)
        scores = model.score(texts[0], texts[1])

    for result, target in zip(actual, expected):
        torch.testing.assert_close(
            torch.as_tensor(result), target, rtol=1e-5, atol=1e-5
        )
    assert scores == pytest.approx([expected_score.item()], rel=1e-5, abs=1e-5)
