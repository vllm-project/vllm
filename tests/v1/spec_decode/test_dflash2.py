# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_groups,group_size,block_size,taps",
    [
        (7, 13, 11, 5, 1),
        (7, 13, 11, 5, 3),
        (7, 13, 11, 8, 2),
        (16, 64, 16, 8, 2),
        (16, 160, 16, 8, 2),
    ],
)
def test_grouped_conv_triton_matches_reference(
    dtype: torch.dtype,
    batch: int,
    num_groups: int,
    group_size: int,
    block_size: int,
    taps: int,
):
    torch.manual_seed(0)
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size, device="cuda", dtype=dtype)
    base = torch.randn(taps, num_groups * group_size, device="cuda", dtype=dtype)
    projected = torch.randn(rows, 2, taps, num_groups, device="cuda", dtype=dtype)
    delta = projected[:, 1]

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )

    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(
        actual,
        expected.flatten(0, 1).flatten(-2).to(dtype),
        rtol=1e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype is torch.bfloat16 else 1e-5,
    )


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skip_global_cleanup
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 3,
            "conv_group_size": 2,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model = DFlash2Qwen3Model(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)


@pytest.mark.parametrize("excluded", [False, True])
def test_selector_projection_loads_draft_quantization(monkeypatch, excluded):
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    prefix = "model.candidate_selector"
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=[f"{prefix}.hidden_projection"] if excluded else [],
    )
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=0),
        )
    ):
        selector = CandidateSelector(
            hidden_size=16,
            vocab_size=32,
            rank=8,
            top_k=4,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=prefix,
        )
    weight = torch.ones(
        8,
        16 if excluded else 8,
        dtype=torch.bfloat16 if excluded else torch.uint8,
    )
    AutoWeightsLoader(selector).load_weights([("hidden_projection.weight", weight)])
    torch.testing.assert_close(selector.hidden_projection.weight, weight)
    assert selector.predecessor_codebook.dtype == torch.bfloat16
    assert selector.successor_codebook.dtype == torch.bfloat16


@pytest.fixture(params=["dflash", "dspark"])
def draft_lm_head_model(request, monkeypatch):
    from torch import nn

    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models import qwen3_dflash, qwen3_dspark
    from vllm.model_executor.models import utils as model_utils

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    for module in (qwen3_dflash, qwen3_dspark, model_utils):
        monkeypatch.setattr(module, "get_draft_quant_config", lambda _: quant_config)

    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.quant_config = quant_config
            self.use_aux_hidden_state = True
            self.has_separate_mask_embedding = False
            self.confidence_head = None

        def _build_fused_kv_buffers(self):
            pass

    monkeypatch.setattr(qwen3_dflash.DFlashQwen3ForCausalLM, "model_cls", Backbone)
    monkeypatch.setattr(qwen3_dspark, "Qwen3DSparkModel", Backbone)
    monkeypatch.setattr(
        qwen3_dflash.DFlashQwen3ForCausalLM, "_read_mask_embedding", lambda _: None
    )
    cls = (
        qwen3_dflash.DFlashQwen3ForCausalLM
        if request.param == "dflash"
        else qwen3_dspark.Qwen3DSparkForCausalLM
    )

    def build(owned):
        hf_config = SimpleNamespace(
            vocab_size=64,
            draft_vocab_size=64,
            hidden_size=16,
            has_own_lm_head=owned,
            num_hidden_layers=0,
            dflash_config={},
        )
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(
                    hf_config=hf_config, get_vocab_size=lambda: 64
                ),
                attention_backend=None,
                kv_cache_dtype=None,
                draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
            ),
            model_config=SimpleNamespace(
                get_total_num_hidden_layers=lambda: 0, get_vocab_size=lambda: 64
            ),
            attention_config=SimpleNamespace(backend=None),
            cache_config=SimpleNamespace(),
            load_config=SimpleNamespace(),
            parallel_config=SimpleNamespace(),
        )
        with set_current_vllm_config(
            VllmConfig(
                device_config=DeviceConfig("cpu"),
                compilation_config=CompilationConfig(mode=0),
            )
        ):
            model = cls(vllm_config=config)
        return model, config

    return request.param, build


@pytest.mark.parametrize("owned", [False, True])
def test_declared_draft_lm_head_loads_and_is_not_replaced(
    monkeypatch, draft_lm_head_model, owned
):
    from vllm.model_executor import model_loader
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
    from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    kind, build = draft_lm_head_model
    model, config = build(owned)
    original_head = model.lm_head
    if owned:
        with pytest.raises(ValueError, match="has_own_lm_head=true"):
            model.load_weights([])
        assert original_head.weight.dtype == torch.uint8
        assert original_head.weight.shape == (64, 8)
        weights = {
            "lm_head." + name: torch.ones_like(value)
            for name, value in original_head.state_dict().items()
            if name != "input_scale"
        }
        model.load_weights(weights.items())
        for name, value in weights.items():
            torch.testing.assert_close(
                getattr(original_head, name.removeprefix("lm_head.")), value
            )
    else:
        model.load_weights([])
        assert original_head.weight.dtype == torch.float32

    target_head = build(owned)[0].lm_head
    with torch.no_grad():
        original_head.weight.fill_(1)
        target_head.weight.fill_(1)
        if owned:
            target_head.weight_scale_2.fill_(2)
    embedding = VocabParallelEmbedding(64, 16)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=embedding), lm_head=target_head
    )
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    for module in (dflash_utils, dspark_utils):
        monkeypatch.setattr(
            module, "replace", lambda obj, **kw: SimpleNamespace(**(vars(obj) | kw))
        )
        monkeypatch.setattr(module, "get_pp_safe_draft_load_config", lambda c: c)
    monkeypatch.setattr(dflash_utils, "get_model", lambda **kw: model)
    monkeypatch.setattr(model_loader, "get_model", lambda **kw: model)
    monkeypatch.setattr(dspark_utils, "_get_dspark_parallel_config", lambda c, tp: c)
    monkeypatch.setattr(
        dspark_utils, "_resolve_dspark_attention_backend", lambda *a: None
    )
    load_model = (
        dflash_utils.load_dflash_model
        if kind == "dflash"
        else dspark_utils.load_dspark_model
    )
    assert load_model(target, config) is model
    assert model.lm_head is (original_head if owned else target_head)
