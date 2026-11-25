# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused TP all-reduce + mHC boundary against an
all-gather and the stock PyTorch reference of mHC post, pre and RMSNorm."""

import pytest
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform

from .tp4 import TP, run_tp4

HIDDEN, HC, SINKHORN, EPS, TOP_K = 4096, 4, 20, 1e-6, 8


def _reference(peers, i):
    # The fused kernel sums the ranks in FP32 and rounds once, as one-shot
    # all-reduces do.
    x = (
        (peers[0].float() + peers[1].float()) + (peers[2].float() + peers[3].float())
    ).bfloat16()
    if current_platform.is_device_capability_family(90):
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_fused_post_pre_tilelang,
        )

        # The deployed TileLang path computes small-batch mix coefficients
        # from the FP32 post output, and norm variance before BF16 collapse.
        return mhc_fused_post_pre_tilelang(
            x, i["residual"], i["post"], i["comb"], i["fn"].float(),
            i["scale"], i["base"], EPS, EPS, EPS, 2.0, SINKHORN,
            norm_weight=i["norm"], norm_eps=EPS,
        )  # fmt: skip
    residual = mhc_post_torch(x, i["residual"], i["post"], i["comb"])
    post, comb, collapsed = mhc_pre_torch(
        residual, i["fn"].float(), i["scale"], i["base"],
        EPS, EPS, EPS, 2.0, SINKHORN,
    )  # fmt: skip
    v = collapsed.float()
    rstd = torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS)
    return residual, post, comb, (v * rstd * i["norm"].float()).bfloat16()


def _inputs(tokens, rank, device):
    g = torch.Generator(device=device).manual_seed(tokens)

    def randn(*shape, generator=g):
        return torch.randn(*shape, generator=generator, device=device)

    x = randn(tokens, HIDDEN, generator=torch.Generator(device).manual_seed(rank))
    # Packed +0/-0 pairs collide with the Lamport sentinel.
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    return dict(
        x=x.bfloat16(),
        residual=randn(tokens, HC, HIDDEN).bfloat16(),
        post=torch.rand(tokens, HC, 1, generator=g, device=device) * 2,
        comb=torch.softmax(randn(tokens, HC, HC), -1),
        fn=(randn(2 * HC + HC * HC, HC * HIDDEN) * 0.02).bfloat16(),
        scale=torch.tensor([0.5, 0.5, 1.0], device=device),
        base=randn(2 * HC + HC * HC) * 0.1,
        norm=(1 + 0.1 * randn(HIDDEN)).bfloat16(),
    )


def _boundary(rank, device):
    from vllm.models.glm5next.nvidia.specialized_decode.all_reduce_mhc import (
        AllReduceMhcBoundary,
    )

    # Larger batches share each cluster between tokens; 37 and 64 end on a
    # short group.
    op = AllReduceMhcBoundary(
        hidden_size=HIDDEN, max_num_tokens=64, top_k=TOP_K, rms_eps=EPS,
        hc_eps=EPS, post_mult=2.0, sinkhorn_repeat=SINKHORN, norm_eps=EPS,
        device=device, cluster_tokens={6: 2, 16: 2, 37: 3, 48: 4, 64: 5},
    )  # fmt: skip

    def check(outputs, i):
        peers = get_tp_group().all_gather(i["x"], dim=0).view(TP, -1, HIDDEN)
        ref = _reference(peers, i)
        # BF16 outputs are one rounding apart at most; the mixes are FP32
        # and differ only in summation order.
        torch.testing.assert_close(outputs[0], ref[0], atol=2**-6, rtol=2**-7)
        torch.testing.assert_close(outputs[1], ref[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(outputs[2], ref[2], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(outputs[3], ref[3], atol=2**-5, rtol=2**-7)

    # Changing shapes cover shrinking and growing batches in one mailbox.
    for tokens in (1, 6, 16, 3, 2, 64, 37, 48, 8):
        i = _inputs(tokens, rank, device)
        args = [i[k] for k in ("x", "residual", "post", "comb", "fn")]
        args += [i[k] for k in ("scale", "base", "norm")]
        eager = op(*args)
        op.join()
        check(eager, i)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                op(*args)
            captured = op(*args)
            op.join()
        for _ in range(10):
            graph.replay()
        check(captured, i)
        # Replays pick up in-place input changes.
        i["x"].mul_(0.5)
        i["residual"].mul_(2)
        graph.replay()
        check(captured, i)

        # A MoE output with its top-k reduction open: the fused finalize
        # must match finalizing first (one FP32 FMA per route in route
        # order, the shared add, one BF16 rounding).
        rows = tokens * TOP_K + 5
        gemm2 = torch.randn(rows, HIDDEN, device=device).bfloat16()
        permuted = torch.randperm(rows, device=device)[: tokens * TOP_K]
        permuted = permuted.view(tokens, TOP_K).int()
        permuted[0, -1] = -1  # an expert this rank does not hold
        weights = torch.rand(tokens, TOP_K, device=device)
        routed = UnfinalizedMoEOutput(gemm2, weights, permuted)
        shared = i["x"]
        moe = MoEOutput(routed=routed, shared_output=shared)
        fused = op(moe, *args[1:])
        acc = torch.zeros(tokens, HIDDEN, device=device)
        for k in range(TOP_K):
            valid = (permuted[:, k] >= 0).unsqueeze(-1)
            rows_k = gemm2[permuted[:, k].clamp_min(0).long()].float()
            torch.addcmul(
                acc,
                torch.where(valid, rows_k, 0.0),
                torch.where(valid, weights[:, k : k + 1], 0.0),
                out=acc,
            )
        plain = op((acc + shared.float()).bfloat16(), *args[1:])
        op.join()
        for a, b in zip(fused, plain):
            assert torch.equal(a, b)

    # Connected boundaries exercise the next call's reads of the preceding
    # residual and side-stream coefficients, rather than reusing fixed inputs.
    i = _inputs(6, rank, device)

    def connected(check_each=False):
        state = i
        for _ in range(4):
            outputs = op(
                *(state[k] for k in ("x", "residual", "post", "comb")),
                *(i[k] for k in ("fn", "scale", "base", "norm")),
            )
            if check_each:
                op.join()
                check(outputs, state)
            residual, post, comb, normalized = outputs
            state = dict(
                i, x=normalized * ((rank + 1) / 16),
                residual=residual, post=post, comb=comb,
            )  # fmt: skip
        op.join()
        return outputs

    expected = connected(check_each=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = connected()
    for _ in range(100):
        graph.replay()
    for actual, want in zip(captured, expected):
        assert torch.equal(actual, want)
    i["x"].mul_(0.5)
    i["residual"].mul_(2)
    expected = connected(check_each=True)
    graph.replay()
    for actual, want in zip(captured, expected):
        assert torch.equal(actual, want)


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ),
    reason="Requires SM90 or SM100 with NVLink multicast",
)
def test_all_reduce_mhc_boundary_matches_reference():
    run_tp4(_boundary)


@pytest.fixture
def hopper_model_config(tmp_path):
    from transformers import Glm5NextConfig, Glm5NextTextConfig

    from vllm.models.glm5next.nvidia.specialized_decode.runtime import GEOMETRY

    text_config = Glm5NextTextConfig(
        **GEOMETRY,
        num_hidden_layers=3,
        layer_types=["linear_attention", "indexed_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        num_nextn_predict_layers=1,
        index_share_for_mtp_iteration=True,
        indexer_rope_interleave=False,
        mla_use_nope=True,
        topk_method="noaux_tc",
        intermediate_size=12288,
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        mhc=True,
    )
    Glm5NextConfig(
        text_config=text_config,
        architectures=["Glm5NextForCausalLM"],
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                f"model.layers.{layer}.self_attn.{name}"
                for layer in (0, 2)
                for name in (
                    "q_proj",
                    "in_proj_qkvbfg_a",
                    "k_proj",
                    "v_proj",
                    "b_proj",
                    "f_a_proj",
                    "g_a_proj",
                    "f_b_proj",
                    "g_b_proj",
                    "o_proj",
                )
            ],
        },
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="Hopper defaults",
)
@pytest.mark.parametrize(
    "mode",
    ["auto", "stock", "eager", "unsupported_policy", "other_attention", "bf16_ssm"],
)
def test_hopper_default_dispatch(hopper_model_config, mode):
    """Enable the Hopper stages with stock KV and retain stage-specific fallbacks."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.models.glm5next.nvidia.specialized_decode.policy import Policy
    from vllm.models.glm5next.nvidia.specialized_decode.runtime import (
        specialized_decode_config,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    extra = {}
    if mode == "stock":
        extra["decode_backend"] = "stock"
    if mode == "unsupported_policy":
        extra["policy"] = {"moe": "1:per_route"}
    config = EngineArgs(
        model=hopper_model_config,
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        enforce_eager=mode == "eager",
        attention_backend=(
            AttentionBackendEnum.FLASHMLA_SPARSE if mode == "other_attention" else None
        ),
        mamba_ssm_cache_dtype="bfloat16" if mode == "bf16_ssm" else "auto",
        additional_config=extra,
    ).create_engine_config()
    decode = specialized_decode_config(config)
    if mode in ("stock", "eager", "unsupported_policy"):
        assert decode is None
        return
    assert decode is not None
    assert config.cache_config.cache_dtype == "auto"
    policy = Policy(dict(decode.policy))
    assert policy("boundary", 1) == 1
    assert policy("boundary", 64) == 5
    assert policy("moe", 1) == "triton"
    assert policy("dense_mlp", 1) == "cute"
    assert policy("logits_gather", 1) == "nvlink"
    assert policy.enabled("kda") == (mode != "bf16_ssm")
    assert policy.enabled("dsa_attn") == (mode != "other_attention")
    assert policy.enabled("dsa_kpool") == (mode != "other_attention")
    if mode == "auto":
        config.additional_config = {"decode_backend": "latency_specialized"}
        assert decode == specialized_decode_config(config)


def _specialized_decode_stages(worker):
    runner = worker.model_runner
    decode = runner.model.specialized_decode
    target = bool(decode and decode.kda and decode.dense and decode.moe)
    speculator = runner.speculator
    if speculator is None:
        return target, None
    assert speculator.share_mtp_topk_indices
    draft = speculator.model.model.specialized_decode
    return target, bool(draft and draft.ar_norm)


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90)
    or torch.accelerator.device_count() < TP,
    reason="Requires four Hopper GPUs with NVLink multicast",
)
@pytest.mark.parametrize("mtp", [False, True], ids=["decode", "mtp5-long"])
def test_hopper_default_preserves_generation(
    vllm_runner, hopper_model_config, enable_pickle, mtp
):
    """Exercise Hopper dispatch and graph replay through LLM.generate."""
    from tests.models.utils import check_logprobs_close
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, logprobs=5, detokenize=False
    )
    # Exercise kpool remap beyond the point where it can select every pool.
    prompt = [11, 17, 23] * (686 if mtp else 1)
    prompts = [{"prompt_token_ids": prompt}, {"prompt_token_ids": [29, 31]}]
    spec = {"method": "mtp", "num_speculative_tokens": 5} if mtp else None
    capture_sizes = [1, 2, 4, 6, 12, 24] if mtp else [1, 2, 4]
    outputs = []
    for backend in ("stock", "auto"):
        with vllm_runner(
            hopper_model_config,
            load_format="dummy",
            skip_tokenizer_init=True,
            tensor_parallel_size=TP,
            block_size=64,
            dtype="bfloat16",
            max_model_len=2304 if mtp else 128,
            max_num_seqs=4,
            max_num_batched_tokens=512 if mtp else 128,
            gpu_memory_utilization=0.1,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            compilation_config={"cudagraph_capture_sizes": capture_sizes},
            speculative_config=spec,
            additional_config={"decode_backend": backend},
            seed=256,
        ) as runner:
            enabled = backend == "auto"
            expected = (enabled, enabled if mtp else None)
            assert runner.collective_rpc(_specialized_decode_stages) == [expected] * TP
            outputs.append(
                runner.generate_w_logprobs(prompts, sampling, use_tqdm=False)
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="stock",
        name_1="Hopper default",
    )
