# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU regression tests for Qwen4Exp token-sharded GR and PLE."""

from dataclasses import MISSING, fields
from itertools import accumulate, product
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("Qwen4Exp SP requires CUDA", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm.config import (
    EngramConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpModel,
    Qwen4ExpSparseMoeBlock,
)
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from .test_ple import _ConvBatchCase, _make_conv_metadata


class _TPAttentionProjection(nn.Module):
    """Exercise decoder communication with distinct TP output contributions."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=False,
            reduce_results=False,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        """Return this TP rank's partial output projection."""
        return self.proj(hidden_states)[0]


def _check_vocab_embedding() -> None:
    """Check default AR and deferred RS for both fused and eager vocabulary lookups."""
    for dtype, use_fused in product(
        (torch.bfloat16, torch.float8_e4m3fn), (False, True)
    ):
        embedding = VocabParallelEmbedding(128, 16, params_dtype=dtype)
        partial_embedding = VocabParallelEmbedding(
            128, 16, params_dtype=dtype, reduce_results=False
        )
        embedding.use_fused_embedding = use_fused
        partial_embedding.use_fused_embedding = use_fused
        values = torch.arange(128 * 16).reshape(128, 16)
        table = ((values % 29 - 14) / 16).to(dtype)
        embedding.weight_loader(embedding.weight, table)
        partial_embedding.load_state_dict(embedding.state_dict())
        # Include vocabulary rows owned by both ranks and a padded SP token count.
        ids = torch.tensor([0, 63, 64, 127, 12])
        expected = table[ids].float()
        torch.testing.assert_close(embedding(ids).float(), expected, atol=0, rtol=0)
        partial = partial_embedding(ids)
        if dtype == torch.float8_e4m3fn:
            local = sp_reduce_scatter(partial.view(torch.int8)).view(dtype)
        else:
            local = sp_reduce_scatter(partial)
        torch.testing.assert_close(local.float(), sp_shard(expected), atol=0, rtol=0)


def _make_decoder(
    vllm_config: VllmConfig, rank: int, prefix: str
) -> Qwen4ExpDecoderLayer:
    """Build a decoder with real GR and MoE kernels and a TP attention projection."""
    config = vllm_config.model_config.hf_text_config
    layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = "full_attention"
    layer.ple = None
    layer.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
    hc_config = HyperConnectionConfig(
        hc_count=config.hc_count,
        hidden_size=config.hidden_size,
        hc_lowrank=config.hc_lowrank,
        params_dtype=torch.bfloat16,
        hc_per_branch_norm=True,
    )
    layer.attn_hyper_connection = GatedResidual(hc_config)
    layer.mlp_hyper_connection = GatedResidual(hc_config)
    layer.self_attn = _TPAttentionProjection(config.hidden_size)
    layer.mlp = Qwen4ExpSparseMoeBlock(
        vllm_config, prefix=f"{prefix}.mlp", reduce_results=False
    )
    torch.manual_seed(12)
    for param in layer.parameters():
        param.normal_(std=0.08)
    layer.self_attn.proj.weight.add_(rank * 0.01)
    experts = layer.mlp.experts.routed_experts
    if layer.use_sequence_parallel:
        # Distinct expert owners expose wrong EP routing or duplicate reductions.
        experts.w13_weight.add_(rank * 0.01)
    experts.quant_method.process_weights_after_loading(experts)
    return layer


def _check_decoder(vllm_config: VllmConfig, rank: int) -> None:
    """Compare GR state across position layouts and MoE reduction modes."""
    config = vllm_config.model_config.hf_text_config
    layer = _make_decoder(vllm_config, rank, "decoder")

    # Cover padding, both MoE reduction contracts, and MRoPE's token axis.
    for num_tokens, defer_moe_reduce, use_mrope in (
        (1, True, False),
        (3, False, True),
        (8, True, True),
    ):
        hidden = torch.randn(num_tokens, config.hc_count * config.hidden_size)
        positions = torch.arange(num_tokens)
        if use_mrope:
            positions = positions.unsqueeze(0).expand(3, -1)
        reference = (hidden, None, None)
        local = (sp_shard(hidden), None, None)
        for _ in range(3):
            layer.use_hc_sequence_parallel = False
            layer.self_attn.proj.reduce_results = True
            layer.mlp.experts.moe_config.skip_final_all_reduce = False
            reference = layer(
                *reference,
                positions,
                input_ids=None,
                query_start_loc=None,
                ngram_context=None,
            )
            layer.use_hc_sequence_parallel = True
            layer.self_attn.proj.reduce_results = False
            layer.mlp.experts.moe_config.skip_final_all_reduce = defer_moe_reduce
            local = layer(
                *local,
                positions,
                input_ids=None,
                query_start_loc=None,
                ngram_context=None,
            )
            for index, (actual, expected) in enumerate(zip(local, reference)):
                torch.testing.assert_close(
                    sp_all_gather(actual)[:num_tokens],
                    expected,
                    atol=0.02 if index == 0 else 0.002,
                    rtol=0.02,
                )


def _make_mtp(vllm_config: VllmConfig, rank: int) -> Qwen4ExpMultiTokenPredictor:
    """Build an MTP wrapper around real GR and MoE kernels."""
    config = vllm_config.model_config.hf_text_config
    model = Qwen4ExpMultiTokenPredictor.__new__(Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(model)
    model.hc_count = config.hc_count
    model.hidden_size = config.hidden_size
    model.num_mtp_layers = 1
    model.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.pre_fc_norm_embedding = nn.Identity()
    model.pre_fc_norm_hidden = nn.Identity()
    model.fc_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    model.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    model.hyper_connection_mixer = GatedResidual(
        HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            hc_lowrank=config.hc_lowrank,
            params_dtype=torch.bfloat16,
            hc_per_branch_norm=True,
        ),
        use_combine=False,
    )
    torch.manual_seed(24)
    for param in model.parameters():
        param.normal_(std=0.08)
    layer = _make_decoder(vllm_config, rank, "mtp")
    model.layers = nn.ModuleList([layer])
    return model


def _check_mtp(vllm_config: VllmConfig, rank: int) -> None:
    """Preserve full, contiguous sample and multi-stream outputs across draft steps."""
    config = vllm_config.model_config.hf_text_config
    model = _make_mtp(vllm_config, rank)
    layer = model.layers[0]
    for num_tokens in (3, 8):
        hidden = torch.randn(num_tokens, config.hc_count * config.hidden_size)
        reference_hidden = local_hidden = hidden
        positions = torch.arange(num_tokens)
        for step in range(2):
            ids = torch.randint(0, config.vocab_size, (num_tokens,))
            model.use_hc_sequence_parallel = layer.use_hc_sequence_parallel = False
            layer.self_attn.proj.reduce_results = True
            layer.mlp.experts.moe_config.skip_final_all_reduce = False
            expected = model(ids, positions, reference_hidden, spec_step_idx=step)
            model.use_hc_sequence_parallel = layer.use_hc_sequence_parallel = True
            layer.self_attn.proj.reduce_results = False
            layer.mlp.experts.moe_config.skip_final_all_reduce = True
            actual = model(ids, positions, local_hidden, spec_step_idx=step)
            for output, reference in zip(actual, expected):
                assert output.is_contiguous()
                torch.testing.assert_close(output, reference, atol=0.002, rtol=0.02)
            reference_hidden = expected[1]
            local_hidden = actual[1]


def _check_ple(vllm_config: VllmConfig) -> None:
    """Compare vocabulary RS, full PLE outputs, and persistent convolution state."""
    config = vllm_config.model_config.hf_text_config
    device = torch.device("cuda", torch.accelerator.current_device_index())
    for cpu_offload, fp8 in product((False, True), (False, True)):
        vllm_config.engram_config = EngramConfig(cpu_offload=cpu_offload)
        ple = Qwen4ExpPLELayer(config, vllm_config, prefix=f"ple_{cpu_offload}_{fp8}")
        sp_ple = Qwen4ExpPLELayer(
            config,
            vllm_config,
            prefix=f"ple_sp_{cpu_offload}_{fp8}",
            use_sequence_parallel=True,
        )
        torch.manual_seed(42)
        for param in ple.parameters():
            param.normal_(std=0.02)
        embedding = ple.ple_embedding.ngram_embedding
        if fp8:
            embedding_cls = (
                Qwen4ExpPLEPinnedHostEmbedding
                if cpu_offload
                else Qwen4ExpPLEDeviceEmbedding
            )
            for module in (ple, sp_ple):
                module.ple_embedding.ngram_embedding = embedding_cls(
                    embedding.org_vocab_size,
                    embedding.embedding_dim,
                    params_dtype=torch.bfloat16,
                    padding_size=embedding.padding_size,
                    prefix=f"{module.prefix}.ple_embedding.ngram_embedding",
                    embedding_method=Qwen4ExpPLEFp8EmbeddingMethod(),
                    num_ngram_heads=module.ple_embedding.ngram_heads,
                    max_total_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                    reduce_results=not module.ple_embedding.use_reduce_scatter,
                )
            embedding = ple.ple_embedding.ngram_embedding
            embedding.weight_scale.fill_(0.25)
        # Distinct global vocabulary rows force contributions from remote ranks.
        start = embedding.shard_indices.org_vocab_start_index
        values = torch.arange(embedding.weight.numel()).reshape(embedding.weight.shape)
        values = ((values + start * embedding.embedding_dim) % 29 - 14) / 16
        embedding.weight.copy_(values.to(embedding.weight.dtype))
        # Compare fixed SP configurations with identical weights and separate caches.
        sp_ple.load_state_dict(ple.state_dict())
        ple.num_spec_tokens = sp_ple.num_spec_tokens = 3
        width = ple.conv_state_len + ple.num_spec_tokens
        shape = (64, ple.hc_hidden_size, width)
        state = torch.zeros(shape)
        if not is_conv_state_dim_first():
            state = state.transpose(-1, -2).contiguous()
        ref_state, sp_state = state.clone(), state.clone()
        ple.kv_cache = (ref_state,)
        sp_ple.kv_cache = (sp_state,)
        cases = [
            _ConvBatchCase(prefill_query_lens=(3, 4, 2), include_null_state=False),
            _ConvBatchCase(num_decodes=3),
            _ConvBatchCase(
                spec_query_lens=(4, 3), num_accepted=(2, 1), spec_query_len=4
            ),
            _ConvBatchCase(
                spec_query_lens=(1, 1), num_accepted=(1, 2), spec_query_len=4
            ),
            _ConvBatchCase(num_decodes=1),
        ]
        for case in cases:
            layout, num_tokens = _make_conv_metadata(case, device)
            kwargs = {
                f.name: None
                for f in fields(PleShortConvAttentionMetadata)
                if f.default is MISSING and f.default_factory is MISSING
            }
            kwargs.update(vars(layout))
            metadata = PleShortConvAttentionMetadata(**kwargs)
            lengths = case.spec_query_lens or (
                (1,) * case.num_decodes + case.prefill_query_lens
            )
            qsl = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
            context = torch.randint(
                0, config.vocab_size, (len(lengths), config.ngram_size - 1)
            )
            ids = torch.randint(0, config.vocab_size, (num_tokens,))
            hidden = torch.randn(num_tokens, ple.hc_hidden_size)
            with set_forward_context(
                {ple.prefix: metadata, sp_ple.prefix: metadata}, vllm_config
            ):
                ple.start_prefetch(ids, qsl, context)
                sp_ple.start_prefetch(ids, qsl, context)
                expected = ple(hidden, ids, qsl, context)
                actual = sp_ple(sp_shard(hidden), ids, qsl, context)
            torch.testing.assert_close(
                sp_all_gather(actual)[:num_tokens], expected, atol=0.002, rtol=0.02
            )
            torch.testing.assert_close(sp_state, ref_state, atol=0.002, rtol=0.02)


def _check_moe_sp(vllm_config: VllmConfig, rank: int) -> None:
    """Compare native SP with the existing MoE wrapper across unequal DP batches."""
    config = vllm_config.model_config.hf_text_config
    mtp = _make_mtp(vllm_config, rank)
    layer = mtp.layers[0]

    target = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(target)
    target.config = config
    target.embed_tokens = mtp.embed_tokens
    target.hyper_connection_mixer = mtp.hyper_connection_mixer
    # Reuse a block to exercise delayed GR state across three decoder layers.
    target.layers = nn.ModuleList([layer] * 3)
    target.start_layer, target.end_layer = 0, 3
    target._mtp_hidden_buffer = torch.empty(32, config.hc_count * config.hidden_size)
    dp_rank = vllm_config.parallel_config.data_parallel_rank

    def forward(model, use_sp: bool, **kwargs):
        """Use the current batch with full-token or local-token GR."""
        model.use_hc_sequence_parallel = layer.use_hc_sequence_parallel = use_sp
        model.use_sequence_parallel = layer.use_sequence_parallel = use_sp
        layer.self_attn.proj.reduce_results = not use_sp
        # The baseline MoE wrapper chunks tokens internally and needs a local mask.
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=counts,
            is_padding=padding if use_sp else local_padding,
        ):
            output = model(input_ids=ids, positions=positions, **kwargs)
            torch.testing.assert_close(get_forward_context().is_padding, local_padding)
            return output

    for base_tokens in (1, 8):
        counts = torch.tensor([base_tokens, base_tokens + 2], device="cpu")
        num_tokens = int(counts[dp_rank])
        torch.manual_seed(30 + dp_rank)
        ids = torch.randint(0, config.vocab_size, (num_tokens,))
        positions = torch.arange(num_tokens)
        embeds = torch.randn(num_tokens, config.hidden_size)
        padding = None
        if num_tokens > 1:
            padding = torch.arange(num_tokens) == num_tokens - 1
        local_padding = sp_padding_mask(padding, embeds)

        expected = forward(target, False, inputs_embeds=embeds)
        expected_multi = target._mtp_hidden_buffer[:num_tokens].clone()
        actual = forward(target, True, inputs_embeds=embeds)
        actual_multi = target._mtp_hidden_buffer[:num_tokens].clone()
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
        torch.testing.assert_close(actual_multi, expected_multi, atol=0.02, rtol=0.02)

        for step in range(2):
            expected = forward(
                mtp, False, hidden_states=expected_multi, spec_step_idx=step
            )
            actual = forward(mtp, True, hidden_states=actual_multi, spec_step_idx=step)
            for output, reference in zip(actual, expected):
                assert output.is_contiguous()
                torch.testing.assert_close(output, reference, atol=0.002, rtol=0.02)
            expected_multi, actual_multi = expected[1], actual[1]


def _run_tpsp(rank: int, port: int, use_moe_sp: bool = False) -> None:
    """Run first-stage TP or second-stage TP=2, DP=2, EP regressions."""
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2 if use_moe_sp else 1,
            data_parallel_rank=rank // 2,
            enable_expert_parallel=use_moe_sp,
            all2all_backend="allgather_reducescatter",
        )
    )
    init_distributed_environment(
        world_size=4 if use_moe_sp else 2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # Spawned ranks bypass the GPU worker's workspace initialization.
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=2)
            text_config = Qwen4ExpTextConfig(
                vocab_size=128,
                eos_token_id=0,
                hidden_size=128,
                hc_count=2,
                hc_lowrank=16,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=256,
                shared_expert_intermediate_size=256,
                ple_embed_dim=128,
                heads_per_ngram=2,
                ngram_vocab_size_base=32,
                make_ngram_vocab_size_divisible_by=64,
            )
            config.model_config = SimpleNamespace(
                hf_text_config=text_config, dtype=torch.bfloat16
            )
            with torch.device(f"cuda:{rank}"), torch.inference_mode():
                torch.set_default_dtype(torch.bfloat16)
                if use_moe_sp:
                    _check_moe_sp(config, rank)
                else:
                    with set_forward_context(None, config):
                        _check_vocab_embedding()
                        _check_decoder(config, rank)
                        _check_mtp(config, rank)
                    _check_ple(config)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
def test_tpsp_decoder_and_ple() -> None:
    """Catch duplicate reductions, token misalignment, and diverging PLE/MTP state."""
    mp.spawn(_run_tpsp, args=(get_open_port(),), nprocs=2)


@multi_gpu_test(num_gpus=4)
def test_moe_sp_target_and_mtp(monkeypatch) -> None:
    """Validate automatic MoE SP, padding, and native EP outputs for target and MTP."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    mp.spawn(_run_tpsp, args=(get_open_port(), True), nprocs=4)
