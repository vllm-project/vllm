# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 specialized decode through the real model: a five-layer
DeepSeek-V4.1-Flash on random weights at TP4 with DSpark, served with
``decode_backend`` ``latency_specialized`` and ``auto``. Unlike
``test_kernels.py`` and ``test_numerics.py`` next to it, which
check each path against the op it replaces, this runs the engine: the
target's and the DSpark draft's executors and their per-step gates, the CUDA
graphs, the recorded metadata builds and the weights the executors take off
the layers.

The layers hold each variant: 0 sliding window only, 1 ratio-2 KV and index
source, 2 a ratio-2 consumer with an Engram, 3 ratio-1 KV, index and
candidate source, 4 a ratio-1 index source over 3's candidates. Twelve
requests decode together once the last prefill is done, finish one by one
and leave the last on its own: FULL decode graphs of 12 down to 1 requests
(6 rows each), above 48 rows (inline mHC), padded, row-sharded (12
requests) and low-batch (one request). The draft samples probabilistically
(the requests at temperature 0), so the fused and vocabulary-sharded Markov
samplers run, and the specialized-decode run replays its post-forward CUDA graphs.

Every decoder layer's residual streams and FFN input and the model's output
are kept for every decode token and every 16th prefill position, and the two
runs compared within the tolerances below. Neither run is bitwise
reproducible, not even on the stock paths: the engine batches requests as
they arrive, and two stock runs already differ. Prefill
positions are compared before the decoder's bounded replay only, which runs
the later layers of a prefill step on each request's last tokens.

Needs four GB300 (SM103) GPUs; about 10 minutes, mostly two engine starts.
"""

import zlib
from typing import TYPE_CHECKING

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
LOAD_FORMAT = "dsv41_nvidia_random"
TRUNCATED = {
    "num_hidden_layers": 5,
    # One entry a layer, the three MTP layers' included.
    "compress_ratios": [0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [1, 3],
    "index_source_layer_ids": [1, 3, 4],
    "candidate_source_layer_id": 3,
    "engram_layer_ids": [2],
    "engram_num_embeddings": [32768],
    "engram_vocab_size": 1000,
    "dspark_target_layer_ids": [2, 3, 4],
}


def _truncate(config):
    """``hf_overrides`` for the target and, being callable, the DSpark draft
    too: the target takes its aux hidden-state layers from the draft's
    ``dspark_target_layer_ids``. Module level, so engine processes unpickle it."""
    for key, value in TRUNCATED.items():
        setattr(config, key, value)
    return config


LAYERS = range(5)
DRAFTS = 5  # DSpark draft tokens: 6 rows a request each decode step
TOP_K = 6
FUSED_MHC_MAX_ROWS = 384
MAX_BATCHED_TOKENS = 4096
PREFILL_SAMPLE = 16  # keep every 16th prefill position

# The target and draft models this worker loaded, in load order.
_LOADED: list[torch.nn.Module] = []


@torch.no_grad()
def _fill(name: str, t: torch.Tensor) -> None:
    """A random value for every weight, seeded by its name (alike on every
    rank and in every run), in the ranges the checkpoint's formats hold."""
    if t.device.type == "meta":
        return
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype == torch.uint8 and "scale" in name:
        # E8M0 2^-12 .. 2^-8, alike over each 32 rows as in the checkpoint.
        rows = t.shape[-2] if t.dim() > 1 else 1
        shape = (*t.shape[:-2], -(-rows // 32), t.shape[-1]) if t.dim() > 1 else t.shape
        e8m0 = torch.randint(115, 120, shape, generator=g, device=t.device)
        if t.dim() > 1:
            e8m0 = e8m0.repeat_interleave(32, dim=-2)[..., :rows, :]
        t.copy_(e8m0)
    elif t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(randn().clamp(-6, 6))
    elif not torch.is_floating_point(t):
        return
    elif "norm" in name:
        t.copy_(randn(0.1, 1.0))
    elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
        t.copy_(randn(0.5).abs() + 0.5)
    elif name.endswith(("hc_attn_base", "hc_ffn_base", "attn_sink")):
        t.copy_(randn(0.1))
    elif name.endswith("embed_tokens.weight"):
        t.copy_(randn())
    else:
        t.copy_(randn(0.02))


class _Probe:
    """Persistent buffers that the target model fills with every decoder
    layer's residual streams (after attention) and FFN input, the vLLM
    routing margin of that input, and the model's output. Copies into
    persistent buffers, so CUDA graph replays fill them too.

    The stock layers fill them from hooks. The executor runs the layers'
    modules but not their forwards, so its boundary into each layer's FFN
    fills them."""

    def __init__(self, core: torch.nn.Module):
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer
        from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (  # noqa: E501
            Boundaries,
        )

        layers = [m for m in core.layers if isinstance(m, DeepseekV4DecoderLayer)]
        hc, hidden = layers[0].hc_mult, layers[0].hidden_size
        rows, device = MAX_BATCHED_TOKENS, layers[0].attn_norm.weight.device

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(rows, *shape, dtype=dtype, device=device)

        self.residual = [zeros(hc, hidden) for _ in layers]
        self.ffn_input = [zeros(hidden) for _ in layers]
        self.gap = [zeros(dtype=torch.float32) for _ in layers]
        self.output = zeros(hidden)
        self.gates = [layer.ffn.gate for layer in layers]
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._keep_residual(i))
            self._hook_ffn(i, layer.ffn)
        core.register_forward_hook(self._keep_output)

        index = {layer: i for i, layer in enumerate(layers)}
        sublayer = Boundaries.sublayer
        # Executor boundaries traced (the executor runs only in captures):
        # proves that the CUDA graphs hold the executor, not the stock layers.
        self.executor_sublayers = 0
        # The same for the DSpark draft's executor (layers outside ``index``).
        self.draft_sublayers = 0

        def sublayer_and_keep(boundaries, layer, ffn, state, step, **kwargs):
            i = index.get(layer)
            if i is None:
                self.draft_sublayers += 1
            else:
                self.executor_sublayers += 1
            out = sublayer(boundaries, layer, ffn, state, step, **kwargs)
            if ffn and i is not None:
                residual = out[0].residual
                self.residual[i][: residual.shape[0]].copy_(residual)
                self._keep_ffn_input(i, out[1])
            return out

        Boundaries.sublayer = sublayer_and_keep

    def _keep_residual(self, i):
        def hook(module, args, out):
            self.residual[i][: out[1].shape[0]].copy_(out[1])

        return hook

    def _keep_ffn_input(self, i: int, x: torch.Tensor) -> None:
        gate = self.gates[i]
        n = x.shape[0]
        self.ffn_input[i][:n].copy_(x)
        score = torch.nn.functional.softplus(x.float() @ gate.weight.float().T)
        score = score.sqrt() + gate.e_score_correction_bias.float()
        top = score.topk(TOP_K + 1, dim=-1).values
        self.gap[i][:n].copy_(top[:, TOP_K - 1] - top[:, TOP_K])

    def _hook_ffn(self, i, ffn):
        ffn.register_forward_pre_hook(
            lambda module, args: self._keep_ffn_input(i, args[0])
        )
        # The stock deferred-finalize MoE is called as forward_unfinalized.
        unfinalized = ffn.forward_unfinalized

        def forward_unfinalized(x, input_ids):
            self._keep_ffn_input(i, x)
            return unfinalized(x, input_ids)

        ffn.forward_unfinalized = forward_unfinalized

    def _keep_output(self, module, args, out):
        out = out[0] if isinstance(out, tuple) else out
        self.output[: out.shape[0]].copy_(out)

    def values(self, rows: torch.Tensor) -> dict:
        """The kept tensors of ``rows``, the residual streams' hc mean."""
        return {
            "residual": [
                t[rows].float().mean(1).bfloat16().cpu() for t in self.residual
            ],
            "ffn_input": [t[rows].cpu() for t in self.ffn_input],
            "gap": [t[rows].cpu() for t in self.gap],
            "output": self.output[rows].cpu(),
        }


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading with weights in the checkpoint's ranges, before post-load
    processing, so every processed copy follows. Also puts a ``_Probe`` on
    the target model before its CUDA graphs are captured."""

    def load_weights(self, model, model_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            _fill(name, t)
        # The routed experts' intermediate past this rank's share is the
        # loader's zero padding (576 -> 640 at TP4).
        inter = model_config.hf_text_config.moe_intermediate_size
        inter //= get_tensor_model_parallel_world_size()
        for m in model.modules():
            w13, w2 = getattr(m, "w13_weight", None), getattr(m, "w2_weight", None)
            if w13 is None or w2 is None or w13.device.type == "meta":
                continue
            padded = w13.shape[1] // 2
            if padded > inter:
                w13, w2 = w13.data.view(torch.uint8), w2.data.view(torch.uint8)
                for half in (0, padded):
                    w13[:, half + inter : half + padded] = 0
                w2[..., inter // 2 :] = 0
        _LOADED.append(model)
        cores = [m for m in model.modules() if type(m) is DeepseekV4Model]
        if cores:
            model._layer_probe = _Probe(cores[0])


class StepRecorder:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers. Rank 0 keeps every step's rows."""

    model_runner: "GPUModelRunner"
    rank: int

    def _target(self) -> torch.nn.Module:
        return next(m for m in _LOADED if hasattr(m, "_layer_probe"))

    def paths(self) -> dict:
        """Which implementation each path took on this rank."""
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.models.deepseek_v41.nvidia.model import (
            DeepseekV4DecoderLayer,
            DeepseekV4Model,
        )
        from vllm.models.deepseek_v41.nvidia.ops import mhc

        def layers(model):
            return [m for m in model.modules() if isinstance(m, DeepseekV4DecoderLayer)]

        target = self._target()
        (draft,) = [m for m in _LOADED if m is not target]
        assert not any(type(m) is DeepseekV4Model for m in draft.modules())
        logits = [m for m in target.modules() if isinstance(m, LogitsProcessor)]
        (core,) = [m for m in target.modules() if type(m) is DeepseekV4Model]
        language_model = (
            target.get_language_model()
            if hasattr(target, "get_language_model")
            else target
        )
        decode = language_model.specialized_decode
        draft_decode = draft.specialized_decode
        model_state = self.model_runner.model_state
        graphs = getattr(model_state, "graphs", None)
        speculator = self.model_runner.speculator
        return {
            # Prefill steps run the layers from here on a subset of the rows.
            "decoder_replay_start": core.decoder_replay_start,
            "decode": type(decode).__name__,
            "executor_sublayers": target._layer_probe.executor_sublayers,
            "draft_sublayers": target._layer_probe.draft_sublayers,
            "draft_decode": type(draft_decode).__name__,
            "model_state": type(model_state).__name__,
            "attention": [type(m.attn).__name__ for m in layers(target)],
            "ffn": [type(m.ffn).__name__ for m in layers(target)],
            "routed_experts": [
                plan.moe.gemm is not None for plan in getattr(decode, "plans", ())
            ],
            "logits": [type(m).__name__ for m in logits],
            "metadata_graphs": 0 if graphs is None else len(graphs._graphs),
            "fused_mhc_max_rows": (
                None if decode is None else decode.all_reduce.capacity
            ),
            "stock_fused_mhc_max_rows": getattr(mhc._all_reduce_mhc, "capacity", None),
            "draft_attention": [type(m.attn).__name__ for m in layers(draft)],
            "draft_routed_experts": [
                plan.moe.gemm is not None for plan in getattr(draft_decode, "plans", ())
            ],
            "draft_logits": type(draft.logits_processor).__name__,
            "markov_samplers": [
                speculator._fused_sampler is not None,
                speculator._sharded_sampler is not None,
            ],
            "post_forward_graphs": self.model_runner.post_forward_graphs is not None,
        }

    def record_steps(self, prompt_lens: list[int]) -> None:
        """From now on, keep each step's rows on rank 0."""
        runner = self.model_runner
        if self.rank != 0:
            return
        self._prompt_lens = prompt_lens
        self._steps: list[dict] = []
        self._full = False
        run_fullgraph = runner.cudagraph_manager.run_fullgraph

        def run_and_flag(desc):
            self._full = True
            return run_fullgraph(desc)

        runner.cudagraph_manager.run_fullgraph = run_and_flag
        self._post_forward = "eager"
        graphs = runner.post_forward_graphs
        if graphs is not None:
            run_region = graphs._run

            def run_region_and_flag(key, fn, stream):
                self._post_forward = "replay" if key in graphs.graphs else "graph"
                return run_region(key, fn, stream)

            graphs._run = run_region_and_flag
        self._recorded = False
        execute_model = runner.execute_model
        sample_tokens = runner.sample_tokens

        def execute_and_record(*args, **kwargs):
            self._full = False
            self._recorded = False
            output = execute_model(*args, **kwargs)
            state = runner.execute_model_state
            if not kwargs.get("dummy_run") and state is not None:
                self._record(state.input_batch)
                self._recorded = True
            return output

        def sample_and_flag(*args, **kwargs):
            # The engine samples, and so runs the post-forward region, in a
            # call of its own after execute_model.
            self._post_forward = "eager"
            output = sample_tokens(*args, **kwargs)
            if self._recorded:
                self._steps[-1]["post_forward"] = self._post_forward
                self._recorded = False
            return output

        runner.execute_model = execute_and_record
        runner.sample_tokens = sample_and_flag

    def _record(self, batch) -> None:
        torch.accelerator.synchronize()
        probe = self._target()._layer_probe
        n, padded = batch.num_tokens, batch.num_tokens_after_padding
        positions = batch.positions[:n].tolist()
        input_ids = batch.input_ids[:n].tolist()
        step = {
            "rows": n,
            "padded_rows": padded,
            "requests": batch.num_reqs,
            "full": self._full,
            # How the region from sampling to the draft inputs ran; set once
            # the step has sampled.
            "post_forward": None,
            "has_prefill": batch.has_prefill,
        }
        kept = []
        for i in range(batch.num_reqs):
            prompt = self._prompt_lens.index(int(batch.prefill_len_np[i]))
            decode = not batch.is_prefilling_np[i]
            start, end = batch.query_start_loc_np[i : i + 2]
            for row in range(start, end):
                key = (prompt, positions[row])
                if decode:
                    key += (tuple(input_ids[start : row + 1]),)
                elif positions[row] % PREFILL_SAMPLE:
                    continue
                kept.append((row, key))
        if kept:
            rows = torch.tensor([r for r, _ in kept], device=probe.output.device)
            values = probe.values(rows)
            step["tokens"] = {
                key: {
                    "residual": [t[j] for t in values["residual"]],
                    "ffn_input": [t[j] for t in values["ffn_input"]],
                    "gap": [float(t[j]) for t in values["gap"]],
                    "output": values["output"][j],
                }
                for j, (_, key) in enumerate(kept)
            }
        self._steps.append(step)

    def save_steps(self, path: str) -> None:
        if self.rank == 0:
            torch.save(self._steps, path)


def _serve(vllm_runner, specialized_decode: bool, prompts, max_tokens, path: str):
    """Greedy decode of ``prompts``: each prompt's output tokens and the
    paths each rank took; the steps' rows saved under ``path``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    with vllm_runner(
        MODEL,
        tensor_parallel_size=4,
        load_format=LOAD_FORMAT,
        hf_overrides=_truncate,
        worker_extension_cls=f"{__name__}.StepRecorder",
        additional_config={
            "decode_backend": "latency_specialized" if specialized_decode else "auto"
        },
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": DRAFTS,
            "draft_sample_method": "probabilistic",
            "enable_adaptive_verification": False,
        },
        attention_config={
            "backend": "FLASHINFER_MLA_SPARSE_DSV41",
            "indexer_kv_dtype": "mxfp4",
            "indexer_sparse_logits": True,
        },
        kv_cache_dtype="fp8",
        engram_config={"cpu_offload": True},
        # Autotuning times each run apart and may pick other tactics for the
        # stock paths; FlashInfer's defaults are the same in both.
        enable_flashinfer_autotune=False,
        # Decode graphs of 1, 2, 4, 8 and 16 requests of 6 rows.
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [6, 12, 24, 48, 96],
        },
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        max_model_len=4096,
        block_size=None,
        max_num_seqs=16,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.6,
    ) as runner:
        llm = runner.llm
        paths = llm.collective_rpc("paths")
        llm.collective_rpc("record_steps", args=([len(p) for p in prompts],))
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            [SamplingParams(temperature=0.0, max_tokens=n) for n in max_tokens],
        )
        llm.collective_rpc("save_steps", args=(path,))
    return [list(o.outputs[0].token_ids) for o in outputs], paths


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _assert_quantiles(errs: list[float], what, median: float, p99: float) -> None:
    assert errs, what
    q = torch.quantile(torch.tensor(errs), torch.tensor([0.5, 0.99])).tolist()
    # Any single token within 15%, five times what two stock runs differ by.
    assert q[0] < median and q[1] < p99 and max(errs) < 0.15, (what, q, max(errs))


def _check_close(got: dict, want: dict, keys: list, layers, output: bool) -> None:
    """The kept tensors of ``keys`` on both runs, close in distribution.

    Random weights amplify any difference layer by layer, and the stock paths
    do not reproduce either: single decode tokens of two stock runs
    differ by up to about 3%, with a median below 0.5% at every layer. Per
    layer, the residual streams' median within 1% and 99th percentile within
    5%. A token whose FFN input is off by 3% at layer 0 or
    8% past it must have a near-tie top-6 routing on the stock path: it may
    pick another expert (the low-batch experts route in the same FP32 order,
    but their input carries the earlier differences), and its later layers
    are exempt. A path the specialized-decode run no longer follows is off by far
    more, on every token.
    """
    flipped: set = set()
    for i in layers:
        residual = []
        for key in keys:
            if key in flipped:
                continue
            a, b = got[key], want[key]
            residual.append(_rel(a["residual"][i], b["residual"][i]))
            err = _rel(a["ffn_input"][i], b["ffn_input"][i])
            if err >= (0.03 if i == 0 else 0.08):
                assert b["gap"][i] < (5e-3 if i == 0 else 1e-2), (i, key, "x", err)
                flipped.add(key)
        _assert_quantiles(residual, (i, "residual"), median=0.01, p99=0.05)
    if output:
        errs = [
            _rel(got[k]["output"], want[k]["output"]) for k in keys if k not in flipped
        ]
        _assert_quantiles(errs, "output", median=0.02, p99=0.08)
    assert len(flipped) <= len(keys) // 4, sorted(flipped)


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_device_capability(103),
    reason="decode_backend=latency_specialized runs on GB300 (SM103)",
)
def test_specialized_decode_matches_the_model(vllm_runner, tmp_path):
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs four GPUs")
    g = torch.Generator().manual_seed(0)
    # Distinct prompt lengths past both ratios' 512 compressed entries.
    lens = (torch.randperm(900, generator=g)[:12] + 1100).tolist()
    prompts = [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]
    # More tokens than the prefill steps decode (at most 6 a step), so all
    # requests decode together; the last runs on alone.
    max_tokens = [60 + 2 * i for i in range(len(prompts) - 1)] + [120]

    on, paths_on = _serve(vllm_runner, True, prompts, max_tokens, f"{tmp_path}/on")
    off, paths_off = _serve(vllm_runner, False, prompts, max_tokens, f"{tmp_path}/off")

    # The executors run the target and DSpark draft models over their stock
    # layers, with every MoE layer on the low-batch GEMMs (the target's
    # deferring), and the draft samples on the fused and sharded samplers.
    for rank in paths_on:
        assert rank["decode"] == "SpecializedDecode", rank
        assert rank["executor_sublayers"] > 0, "no step ran the executor"
        assert rank["draft_sublayers"] > 0, "no draft step ran the executor"
        assert rank["draft_decode"] == "DraftDecode", rank
        assert rank["model_state"] == "DeepseekV41SpecializedDecodeModelState", rank
        assert rank["routed_experts"] == [True] * len(LAYERS), rank
        assert rank["draft_routed_experts"] == [True] * len(rank["draft_attention"])
        assert rank["logits"] == ["SpecializedDecodeLogitsProcessor"], rank
        assert rank["draft_logits"] == "SpecializedDecodeLogitsProcessor", rank
        assert rank["metadata_graphs"] > 0, rank
        assert rank["fused_mhc_max_rows"] == FUSED_MHC_MAX_ROWS, rank
        assert rank["markov_samplers"] == [True, True], rank
        assert rank["post_forward_graphs"], rank
    for rank in paths_off:
        assert rank["decode"] == "NoneType", rank
        assert rank["executor_sublayers"] == 0, rank
        assert rank["draft_sublayers"] == 0, rank
        assert rank["draft_decode"] == "NoneType", rank
        assert rank["model_state"] == "DeepseekV41ModelState", rank
        assert rank["logits"] == ["LogitsProcessor"], rank
        assert rank["draft_logits"] == "LogitsProcessor", rank
        assert rank["metadata_graphs"] == 0, rank
        assert rank["markov_samplers"] == [False, False], rank
        assert not rank["post_forward_graphs"], rank
    stock = ("attention", "ffn", "stock_fused_mhc_max_rows", "draft_attention")
    for key in stock:
        assert paths_on[0][key] == paths_off[0][key], key
    assert paths_on[0]["attention"] == ["DeepseekV4FlashInferMLAAttention"] * 5

    got_steps = torch.load(f"{tmp_path}/on")
    want_steps = torch.load(f"{tmp_path}/off")

    # The decode steps covered every per-step gate on the specialized-decode run.
    full = [s for s in got_steps if s["full"]]
    assert {s["requests"] for s in full} >= {1, 12}, full
    # 5 to 8 requests replay the 48-row graph, the largest whose next mHC
    # coefficients run on the side stream.
    assert any(5 <= s["requests"] <= 8 for s in full), "no 48-row FULL step"
    assert any(s["padded_rows"] > s["rows"] for s in full), "no padded FULL step"
    assert any(s["rows"] > 48 for s in full), "no FULL step above 48 rows"
    assert all(s["post_forward"] is not None for s in got_steps), "a step not sampled"
    assert any(s["post_forward"] == "replay" for s in full), "no post-forward replay"
    assert all(s["post_forward"] == "eager" for s in want_steps)

    def merged(steps):
        rows: dict = {}
        for step in steps:
            rows.update(step.get("tokens", {}))
        return rows

    # A decode token's inputs match when the greedy tokens before it agree;
    # a prefill token's (a key of prompt and position) always do.
    def same_context(key) -> bool:
        if len(key) == 2:
            return True
        prompt, position, step_ids = key
        generated = position - lens[prompt] - (len(step_ids) - 1)
        return on[prompt][: generated + 1] == off[prompt][: generated + 1]

    # Mixed steps run the stock layers on both runs.
    assert any(
        s["has_prefill"] and any(len(key) == 3 for key in s.get("tokens", {}))
        for s in got_steps
    ), "no decode token in a mixed step"
    got, want = merged(got_steps), merged(want_steps)
    keys = sorted(k for k in got.keys() & want.keys() if same_context(k))
    decode = [k for k in keys if len(k) == 3]
    prefill = [k for k in keys if len(k) == 2]
    assert len(decode) >= (DRAFTS + 1) * len(prompts), len(decode)
    assert len(prefill) > sum(lens) // (2 * PREFILL_SAMPLE), len(prefill)
    _check_close(got, want, decode, LAYERS, output=True)
    replay_start = paths_on[0]["decoder_replay_start"]
    assert replay_start == paths_off[0]["decoder_replay_start"]
    _check_close(got, want, prefill, range(replay_start), output=False)
