# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import importlib.util
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from tests.utils import create_new_process_for_each_test, multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)


def get_and_check_output(output, expected_shape):
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None

    obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
    token_ids = obj["token_ids"]
    hidden_states = obj["hidden_states"]

    prompt_token_ids = output.prompt_token_ids
    assert torch.equal(token_ids, torch.tensor(prompt_token_ids))

    assert hidden_states.shape == expected_shape

    # Verify hidden_states are not all zeros (i.e., they were actually computed)
    assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))

    return token_ids, hidden_states


def save_synthetic_tokenizer(config_dir):
    """Save a tiny local tokenizer so tests do not depend on HF downloads."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[UNK]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[PAD]": 3,
    }
    vocab.update({f"token_{idx}": idx for idx in range(4, 1000)})

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        model_max_length=128,
    )
    fast_tokenizer.save_pretrained(config_dir)


@pytest.fixture(scope="module")
def require_predictable_model_plugin():
    """Skip unless the spawn-safe predictable-model plugin is installed.

    The predictable dummy models are registered through the
    ``vllm.general_plugins`` entry point exposed by the
    ``vllm_add_predictable_models`` package under ``tests/plugins``. vLLM re-runs
    plugin registration in every process it creates -- the short-lived
    model-inspection subprocess and each engine worker -- so this is the one
    mechanism that works under both ``fork`` and ``spawn`` and regardless of the
    directory pytest was launched from. A plain in-process
    ``ModelRegistry.register_model`` call does not survive a ``spawn``ed worker.

    Install once with::

        pip install -e tests/plugins/vllm_add_predictable_models
    """
    if importlib.util.find_spec("vllm_add_predictable_models") is None:
        pytest.skip(
            "Predictable test models come from a vLLM general plugin. Install it "
            "with: pip install -e tests/plugins/vllm_add_predictable_models"
        )


@pytest.fixture(scope="module")
def predictable_llama_config_path(tmp_path_factory, require_predictable_model_plugin):
    """Create a minimal LlamaConfig for PredictableLlamaForCausalLM."""
    from transformers import LlamaConfig

    config_dir = tmp_path_factory.mktemp("predictable_llama")

    # Create a minimal Llama config with small dimensions
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,  # Enough layers to test various layer_ids
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        architectures=["PredictableLlamaForCausalLM"],
    )

    # Save config
    config.save_pretrained(config_dir)
    save_synthetic_tokenizer(config_dir)

    return str(config_dir)


@pytest.fixture(scope="module")
def predictable_hybrid_config_path(tmp_path_factory, require_predictable_model_plugin):
    """Create a minimal LlamaConfig for PredictableHybridForCausalLM.

    Uses 2 hidden-state layer IDs so that CacheOnlyAttentionLayer's
    MLAAttentionSpec page size is compatible with the verifier's hybrid cache:
      page_size = block_size(16) * num_hidden_states(2) * hidden_size(256)
                  * dtype_size(2) = 16384 bytes per page.
    """
    from transformers import LlamaConfig

    config_dir = tmp_path_factory.mktemp("predictable_hybrid")

    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        architectures=["PredictableHybridForCausalLM"],
    )
    config.save_pretrained(config_dir)
    save_synthetic_tokenizer(config_dir)

    return str(config_dir)


def test_extract_hidden_states_with_predictable_dummy_model(
    predictable_llama_config_path, tmp_path
):
    """Test hidden-state extraction with a predictable dummy model.

    Tests 3 scenarios:

    1. **Basic extraction**: non-sequential layer ordering, multiple prompts
       of varying length — verifies correct layer association and
       deterministic values.
    2. **Chunked prefill**: max_num_batched_tokens=128 with ~500-token
       prompts so each is split across multiple scheduler iterations —
       verifies hidden states are reassembled correctly.
    3. **Per-request options**: custom hidden_states_path and
       include_output_tokens — verifies per-request kv_transfer_params
       plumbing.

    Model registration is provided by the ``vllm_add_predictable_models`` general
    plugin, so the worker resolves the architecture under any multiprocessing
    start method (this suite runs under ``spawn`` in CI).
    """
    layer_ids = [5, 2, 10]
    num_layers = len(layer_ids)
    max_num_batched_tokens = 128

    llm = LLM(
        model=predictable_llama_config_path,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmp_path,
                "allow_custom_save_path": True,
            },
        },
        max_model_len=1024,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    # --- Scenario 1: basic extraction with non-sequential layers ----------
    prompts = [
        "Short",
        "Medium length",
        "Much longer prompt with many tokens",
        "Much longer prompt with many tokens",  # repeated prompt
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        expected_shape = (
            len(output.prompt_token_ids),
            num_layers,
            hidden_size,
        )
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max={layer_hidden.max():.3f}"
            )

    # --- Scenario 2: chunked prefill with long prompts --------------------
    long_prompt = " ".join(["word"] * 500)
    chunked_prompts = [
        long_prompt,
        long_prompt + " extra tokens here",
        "Short",
    ]
    outputs = llm.generate(chunked_prompts, sampling_params)

    assert len(outputs) == len(chunked_prompts)
    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        expected_shape = (prompt_len, num_layers, hidden_size)
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max="
                f"{layer_hidden.max():.3f}. "
                f"prompt_len={prompt_len}, "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )

    # --- Scenario 3: per-request options ----------------------------------
    max_tokens = 5
    custom_path = os.path.join(tmp_path, "subdir", "custom.safetensors")

    sampling_params_list = [
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": custom_path,
                    "include_output_tokens": True,
                }
            },
        ),
    ]
    per_req_prompts = ["Short", "Medium length"]
    outputs = llm.generate(per_req_prompts, sampling_params_list)

    # First output: prompt-only hidden states, default path
    out0 = outputs[0]
    path0 = out0.kv_transfer_params["hidden_states_path"]
    assert path0 != custom_path
    obj0 = example_hidden_states_connector.load_hidden_states(path0)
    assert torch.equal(obj0["token_ids"], torch.tensor(out0.prompt_token_ids))
    assert obj0["hidden_states"].shape == (
        len(out0.prompt_token_ids),
        num_layers,
        hidden_size,
    )
    example_hidden_states_connector.cleanup_hidden_states(path0)

    # Second output: prompt + output tokens, custom path
    out1 = outputs[1]
    assert out1.kv_transfer_params["hidden_states_path"] == custom_path
    obj1 = example_hidden_states_connector.load_hidden_states(custom_path)
    token_ids = obj1["token_ids"]
    hidden_states = obj1["hidden_states"]
    # The final output token was never an input to the model, so its hidden
    # state is not in the cache — hence the -1.
    total_tokens = len(out1.prompt_token_ids) + len(out1.outputs[0].token_ids) - 1
    assert token_ids.shape[0] == total_tokens
    assert hidden_states.shape == (total_tokens, num_layers, hidden_size)

    # Verify predictable layer values hold for all tokens (prompt + output)
    for idx, layer_id in enumerate(layer_ids):
        layer_hidden = hidden_states[:, idx, :]
        assert torch.allclose(
            layer_hidden,
            torch.full_like(layer_hidden, layer_id),
            atol=1e-5,
        )
    example_hidden_states_connector.cleanup_hidden_states(custom_path)


@create_new_process_for_each_test()
def test_extract_hidden_states_qwen35_hybrid_smoke(tmp_path):
    """Smoke test for Qwen3.5 hybrid (mamba + full-attention) models.
    Uses load_format="dummy" to just check shape/plumbing.
    """
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3.5-0.8B hidden_size

    llm = LLM(
        model="Qwen/Qwen3.5-0.8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        expected_shape = (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )
        get_and_check_output(output, expected_shape)


@create_new_process_for_each_test()
def test_extract_hidden_states_with_hybrid_model(
    predictable_hybrid_config_path, tmp_path
):
    """Integration test for extract_hidden_states with a hybrid (Mamba+attn) model.

    PredictableHybridForCausalLM has is_hybrid=True and contains a FakeMambaLayer,
    which causes vLLM to:
      - Run HybridAttentionMambaModelConfig.verify_and_update_config()
        (sets mamba_block_size)
      - Run Platform._align_hybrid_block_size()
        (sets mamba_page_size_padded so Mamba + attention page sizes agree)
      - Enable the Hybrid Memory Allocator (HMA)
      - Use ExampleHiddenStatesConnector.request_finished_all_groups() instead
        of request_finished() — the SupportsHMA code path fixed in our changes

    Uses 2 layer IDs so that CacheOnlyAttentionLayer's MLAAttentionSpec page
    size (block_size * 2 * hidden_size * dtype) matches the verifier model's
    padded Mamba page size.
    """
    # 2 layer ids: ensures CacheOnly page size is compatible with Mamba padding
    layer_ids = [3, 7]
    num_layers = len(layer_ids)

    llm = LLM(
        model=predictable_hybrid_config_path,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": tmp_path},
        },
        max_model_len=128,
        enforce_eager=True,
        enable_chunked_prefill=False,
        trust_remote_code=True,
        load_format="dummy",
        gpu_memory_utilization=0.04,
        # Keep this dummy-model test independent of transient GPU memory changes
        # from other processes sharing the machine.
        kv_cache_memory_bytes=64 * 1024 * 1024,
        # Disable prefix caching to keep mamba_cache_mode="none" for
        # this fake hybrid model (which has no real Mamba prefix-cache support).
        enable_prefix_caching=False,
    )

    prompts = [
        "Hello",
        "Longer hybrid prompt test",
        "Hybrid model with Mamba layers",
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    outputs = llm.generate(prompts, sampling_params)
    del llm
    gc.collect()

    assert len(outputs) == len(prompts)

    for output in outputs:
        # hidden_states shape is [prompt_len, num_hidden_layers, hidden_size]
        expected_shape = (
            len(output.prompt_token_ids),
            num_layers,
            hidden_size,
        )
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output {float(layer_id)}, "
                f"but got mean={layer_hidden.mean():.3f}, "
                f"min={layer_hidden.min():.3f}, max={layer_hidden.max():.3f}"
            )


@pytest.mark.timeout(60)
@multi_gpu_test(num_gpus=2)
@create_new_process_for_each_test()
def test_extract_hidden_states_tp2():
    """Test that hidden states extraction works with tensor_parallel_size=2."""
    tmp_dir = tempfile.mkdtemp()
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3-0.6B hidden_size

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=2,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": tmp_dir},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        expected_shape = (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )
        get_and_check_output(output, expected_shape)


def test_basic_cache_ignores_padding_slots():
    from vllm.model_executor.models.extract_hidden_states import basic_cache

    kv_cache = torch.full((1, 4, 1, 1), -1, dtype=torch.float32)
    to_cache = torch.tensor(
        [
            [[10.0]],
            [[20.0]],
            [[999.0]],
        ],
        dtype=torch.float32,
    )
    slot_mapping = torch.tensor([0, 1, -1], dtype=torch.int64)

    basic_cache(to_cache, kv_cache, slot_mapping)

    assert kv_cache[:, :, 0, 0].tolist() == [[10.0, 20.0, -1.0, -1.0]]


# ---------------------------------------------------------------------------
# Unit test: unify_kv_cache_spec_page_size() MambaSpec fix
# ---------------------------------------------------------------------------


def test_unify_kv_cache_spec_page_size_mamba_fix():
    """Directly tests the MambaSpec handling in unify_kv_cache_spec_page_size().

    MambaSpec.page_size_bytes is computed from shapes/dtypes and ignores
    block_size, so it is unified by padding (page_size_padded) rather than by
    scaling block_size. block_size must stay untouched so that
    cdiv(max_model_len, block_size) accounting in mamba_cache_mode="all" is not
    distorted.
    """
    import torch

    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    BLOCK_SIZE = 16

    # MLAAttentionSpec: 16 * 2 * 64 * 2 = 4096 bytes
    attn_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )

    # MambaSpec with smaller page_size (256 bytes < 4096 bytes); unified by
    # padding to 4096 (page_size_padded), not by scaling block_size.
    # A single ssm_state of shape (8, 8) in float32 = 64 * 4 = 256 bytes.
    mamba_spec = MambaSpec(
        shapes=((8, 8),),
        dtypes=(torch.float32,),
        block_size=BLOCK_SIZE,
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )

    assert mamba_spec.page_size_bytes == 256  # 8*8*4 = 256 bytes
    assert attn_spec.page_size_bytes == 4096  # 16*2*64*2 = 4096 bytes
    assert 4096 % 256 == 0  # ratio = 16, must be an integer

    kv_cache_spec = {
        "fake_mamba": mamba_spec,
        "cache_only_attn": attn_spec,
    }

    unified = unify_kv_cache_spec_page_size(kv_cache_spec)

    # Both specs must now report the same page_size_bytes
    assert unified["fake_mamba"].page_size_bytes == 4096
    assert unified["cache_only_attn"].page_size_bytes == 4096

    # MambaSpec is padded to the common page; block_size must stay untouched
    # (scaling it would distort cdiv(max_model_len, block_size) accounting).
    assert unified["fake_mamba"].page_size_padded == 4096
    assert unified["fake_mamba"].block_size == BLOCK_SIZE

    # The attention spec should be unchanged
    assert unified["cache_only_attn"] is attn_spec

    # Lock in the safety property: in mamba_cache_mode="all", block budgeting
    # uses the original (un-scaled) block_size. With BLOCK_SIZE=16 and
    # max_model_len=64 that is cdiv(64, 16)=4 pages, whereas a 16x-scaled
    # block_size would have produced only cdiv(64, 256)=1.
    all_mode_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        model_config=SimpleNamespace(max_model_len=64),
    )
    assert unified["fake_mamba"].max_memory_usage_bytes(all_mode_config) == 4 * 4096


def test_extract_hidden_states_uses_cache_only_kv_group():
    """Regression test for hybrid models where cache-only is not KV group 0."""
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        ExampleHiddenStatesConnector,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        HiddenStateCacheSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.mamba_mixer"],
                kv_cache_spec=MambaSpec(
                    shapes=((8, 8),),
                    dtypes=(torch.float32,),
                    block_size=16,
                    mamba_type=MambaAttentionBackendEnum.MAMBA2,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["draft_model.cache_only_layers.24"],
                kv_cache_spec=HiddenStateCacheSpec(
                    block_size=32,
                    num_kv_heads=2,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            ),
        ]
    )

    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(kv_cache_config) == 1

    proposer = ExtractHiddenStatesProposer.__new__(ExtractHiddenStatesProposer)
    proposer.attn_layer_names = ["draft_model.cache_only_layers.24"]
    proposer.kv_cache_gid = -1
    proposer.validate_same_kv_cache_group(kv_cache_config)
    assert proposer.kv_cache_gid == 1


def test_extract_hidden_states_uses_cache_only_group_with_gdn_hybrid():
    """Qwen3.5-style GDN hybrids must still route extract cache ops to draft.

    This is intentionally unit-level: it verifies the hybrid KV-group invariant
    without requiring GDN kernels or a full Qwen3.5 model load.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        ExampleHiddenStatesConnector,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        HiddenStateCacheSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.linear_attn"],
                kv_cache_spec=MambaSpec(
                    shapes=((16, 64),),
                    dtypes=(torch.float32,),
                    block_size=16,
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.self_attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["draft_model.cache_only_layers.24"],
                kv_cache_spec=HiddenStateCacheSpec(
                    block_size=32,
                    num_kv_heads=2,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            ),
        ]
    )

    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(kv_cache_config) == 2

    connector = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    connector._cache_kv_group_id = 2
    assert connector._cache_block_ids(([10], [20], [30, 31])) == [30, 31]

    proposer = ExtractHiddenStatesProposer.__new__(ExtractHiddenStatesProposer)
    proposer.attn_layer_names = ["draft_model.cache_only_layers.24"]
    proposer.kv_cache_gid = -1
    proposer.validate_same_kv_cache_group(kv_cache_config)
    assert proposer.kv_cache_gid == 2
