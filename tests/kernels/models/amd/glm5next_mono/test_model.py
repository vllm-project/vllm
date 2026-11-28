# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level precision of --monokernel on a mini GLM-5.3-Flash at TP2, TP4 and
TP8: the checkpoint's first six layers at their real shapes and quantization,
random weights (no checkpoint download beyond config.json), a small vocabulary.

- layers 0-2: dense KDA, stock;
- layer 3: sparse attention + MoE, a MonoKernel launch;
- layer 4: KDA + MoE, a MonoKernel launch;
- layer 5: KDA + MoE, stock (the last layer's mHC post contracts the streams).

Weights are drawn in the checkpoint's formats before post-load processing
(MXFP4 experts with E8M0 scales, FP8 block-128 projections, bf16 elsewhere), so
vLLM's own post-load path builds every copy the kernels read. The engine runs
teacher-forced decode (trace replay) with full-vocabulary logprobs, with the
MonoKernel eager, under FULL graphs and with an e4m3 KV cache, against vLLM's
layers in the same mode. Prompts mix short contexts with ones past the
indexer's 2048 tokens, so the sparse layer selects.

Eager also checks each kernel layer, token by token on every rank:

- the MoE of each run against an fp64 MoE (`reference.moe`) on that run's own
  input to it and its own routed experts, the kernels at most half vLLM's error;
- the routing: the kernels' expert ids are the fp64 router's on their input
  (`reference.route`), and the tokens where vLLM's and the kernels' ids differ
  are counted and left out of the later cross-run checks;
- the residual streams and layer 5's output against vLLM's, where both runs
  routed alike.

The attention halves are checked stage by stage against fp64 in test_ops.py;
the logits comparison stays as an end-to-end check of the whole step."""

import json
import zlib

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.models.glm5next.amd.specialized_decode.runtime.spec import (
    STEP_TOKENS,
    TP_MAX_WIDTH,
    TP_SIZES,
)
from vllm.utils.torch_utils import set_random_seed

from ._ffn import _ffn_input, _on_cdna4, _routed
from .reference import Bound, Numerics, dequant, moe, rel, route

MODEL = "amd/GLM-5.3-Flash-Quark-MXFP4"
LOAD_FORMAT = "glm5next_mono_random"
LAYERS = 6
KERNEL_LAYERS = (3, 4)
VOCAB = 4096
STEPS = 8
SEED = 0
# Per decode token (eager), each a row of the Numerics table at its batch size.
# Layers 0-2 run vLLM's code in both runs and, with every request added before
# the first step (the engine core runs in the test process), give identical
# outputs: each token's input to layer 3 is the same in both runs, which the
# comparison asserts.
BOUNDS = {
    # each run's MoE against the fp64 MoE on its own input and experts: vLLM
    # quantizes the expert inputs to MXFP4 (0.19-0.21 seen), the kernels to MXFP8
    # (0.054 seen, as test_ops' MoE row); a row is the worst kernel token against
    # the best vLLM one
    **{f"L{i} MoE": Bound(0.1, 0.5) for i in KERNEL_LAYERS},
    # the residual streams after layers 3 and 4, kernels against vLLM: 0.0044
    # seen at TP2, TP4 and TP8 (layer 4's carries layer 3's MoE difference)
    **{f"L{i} residual vs vLLM": Bound(1e-2) for i in KERNEL_LAYERS},
    # layer 5's output, vLLM's layer on each run's streams: 0.0065 seen
    "L5 output vs vLLM": Bound(2e-2),
}
# The kernels route in fp32 from bf16 inputs, so their 8 experts are the fp64
# router's except where its 8th and 9th scores are closer than fp32 can tell
# (rounding moves a score by ~1e-7 here)
ROUTE_TIE = 1e-5
ENV = {
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    # deterministic BF16 GEMMs for the stock layers: the split-K skinny and
    # AITER tuned GEMMs reduce in a run-dependent order
    "VLLM_ROCM_USE_SKINNY_GEMM": "0",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
    # the engine core in this process: every request is added before the first
    # step, so prefills group alike in every run
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}

pytestmark = pytest.mark.skipif(not _on_cdna4(), reason="needs CDNA4 GPUs")
TPS = [pytest.param(tp, marks=pytest.mark.distributed(num_gpus=tp)) for tp in TP_SIZES]


@torch.no_grad()
def _fill(name: str, t: torch.Tensor) -> None:
    """A random value for every weight in the checkpoint's format, seeded by its
    name (alike in every run, so both engines load the same model)."""
    if t.device.type == "meta" or not t.numel():
        return
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype == torch.uint8 and "scale" in name:
        # E8M0 2^-12 .. 2^-8, alike over each 32 rows, as the DeepSeek-V4.1 mono
        # test: larger random layers make the stack chaotic, and the order of
        # atomics alone then moves vLLM's own outputs run to run
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
    elif name.endswith("weight_scale_inv"):
        # FP8 block-128 scales of unit-normal FP8 values: weights of std 0.02
        t.fill_(0.02)
    elif "norm" in name:
        t.copy_(randn(0.1, 1.0))
    elif name.endswith("A_log"):
        t.copy_(torch.empty_like(t).uniform_(0.0, 2.7, generator=g))
    elif name.endswith(("dt_bias", "e_score_correction_bias")):
        t.copy_(randn(0.1))
    elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
        t.copy_(randn(0.5).abs() + 0.5)
    elif name.endswith(("hc_attn_base", "hc_ffn_base")):
        t.copy_(randn(0.1))
    elif "embed_tokens" in name:
        t.copy_(randn())
    else:
        t.copy_(randn(0.02))


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading with weights in the checkpoint's formats and ranges, before
    post-load processing, so every processed copy follows (plain dummy weights
    are near zero and leave the E8M0 scales zero)."""

    def load_weights(self, model, model_config) -> None:
        from vllm.models.glm5next.amd.specialized_decode.mono.config import N_EXPERTS
        from vllm.models.glm5next.common.model import Glm5NextModel

        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            _fill(name, t)
        # the kernel layers' raw expert shards, for the fp64 MoE: post-load
        # processing shuffles them for AITER
        backbone = next(m for m in model.modules() if isinstance(m, Glm5NextModel))
        raw = {}
        for idx in KERNEL_LAYERS:
            moe = backbone.layers[idx].mlp
            e = moe.experts.routed_experts
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            shards = [getattr(e, n).data.view(torch.uint8).clone() for n in names]
            shared = N_EXPERTS
            if moe.shared_experts is not None:
                sh = moe.shared_experts
                shared = tuple(
                    t.data.view(torch.uint8).clone()
                    for lin in (sh.gate_up_proj, sh.down_proj)
                    for t in (lin.weight, lin.weight_scale)
                )
            raw[idx] = (*shards, shared)
        backbone.raw_moe = raw


def _fp64_moe(x, gate, raw, ids) -> torch.Tensor:
    """This rank's intermediate shard of the fp64 MoE on x [N, H], running the
    experts `ids` [N, 8]; `raw`: the layer's shards as the loader kept them."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import N_EXPERTS

    w13, s13, w2, s2, shared = raw

    def expert(e):
        if e == N_EXPERTS and isinstance(shared, tuple):
            a13, b13, a2, b2 = shared
        else:
            a13, b13, a2, b2 = w13[e], s13[e], w2[e], s2[e]
        return dequant(a13, b13), dequant(a2, b2)

    bias = gate.e_score_correction_bias
    return moe(x, gate.weight, bias, expert, N_EXPERTS, order=ids)[0]


class Probe:
    """Worker extension (`worker_extension_cls`): it also brings this module,
    and so the loader, into the workers."""

    def mono_steps(self) -> int:
        from vllm.models.glm5next.amd.specialized_decode.runtime import failstop

        return failstop.counters["mono_steps"]

    def start_capture(self, lens: list[int]) -> None:
        """Keep every decoder layer's outputs (hidden states, residual) for each
        decode token of a decode-only step, keyed by (layer, prompt length,
        position), as the DeepSeek-V4.1 mono test does: a token's step and row
        can differ between engines, its prompt and position cannot (`lens`:
        the batch's prompt lengths, more than STEPS apart). vLLM's layers are
        read through forward hooks, the kernel layers by wrapping the launch
        `plan()` returns. For the kernel layers' MoE each run also keeps its
        input, its expert ids (vLLM's from its router logits, the kernels' from
        their mailbox) and this rank's share of the fp64 MoE on both. Eager
        only (hooks do not run in graph replay)."""
        from vllm.forward_context import get_forward_context
        from vllm.models.glm5next.common.model import (
            Glm5NextDecoderLayer,
            Glm5NextModel,
            Glm5NextMoE,
        )
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        self._captured: dict = {}
        self._moe: dict = {}
        self._kernel_layers: set = set()
        self._lens = sorted(lens)
        if getattr(self, "_hooked", False):
            return
        self._hooked = True

        def decode_only() -> bool:
            md = get_forward_context().attn_metadata
            return isinstance(md, dict) and all(
                getattr(m, "num_prefills", 0) == 0 for m in md.values()
            )

        def keys(idx, positions):
            for row, p in enumerate(positions.tolist()):
                yield row, (idx, max(n for n in self._lens if n <= p), p)

        # The hooks copy to the host mid-step on purpose: exempt them from the
        # engine's VLLM_GPU_SYNC_CHECK (set to "error" in the ROCm CI image).
        def record(idx, positions, out, kernel=False):
            if not decode_only():
                return
            if kernel:
                self._kernel_layers.add(idx)
            # the last layer applies its own hc_post: no residual
            with gpu_sync_allowed():
                for row, key in keys(idx, positions):
                    self._captured.setdefault(
                        key,
                        [None if t is None else t[row].float().cpu() for t in out[:2]],
                    )

        def moe_record(idx, positions, x, ids):
            """A run's MoE input x [N, H] and expert ids [N, 8] at layer idx:
            the ids sorted, the fp64 router's sorted ids on x and its margin
            between the 8th and 9th score, and this rank's share of the fp64
            MoE on x running `ids`."""
            if idx not in backbone.raw_moe or not decode_only():
                return
            gate = backbone.layers[idx].mlp.gate
            with gpu_sync_allowed():
                ids64, gap64 = route(x, gate.weight, gate.e_score_correction_bias)
                ref = _fp64_moe(x, gate, backbone.raw_moe[idx], ids).cpu()
                ids = ids.int().sort(-1).values.cpu()
                ids64, gap64 = ids64.int().sort(-1).values.cpu(), gap64.tolist()
                for row, key in keys(idx, positions):
                    self._moe.setdefault(
                        key, (ids[row], ids64[row], gap64[row], ref[row])
                    )

        positions: list = [None]  # the running decoder layer's
        moe_in: dict = {}  # the running MoE's input and vLLM's router logits

        def vllm_moe(idx, moe):
            gate = moe.gate
            scores = torch.sigmoid(moe_in["logits"].float())
            if gate.e_score_correction_bias is not None:
                scores = scores + gate.e_score_correction_bias.float()
            ids = scores.topk(top_k, dim=-1).indices
            moe_record(idx, positions[0], moe_in["x"], ids)

        model = self.model_runner.model  # type: ignore[attr-defined]
        backbone = next(m for m in model.modules() if isinstance(m, Glm5NextModel))
        top_k = backbone.config.num_experts_per_tok
        for idx, layer in enumerate(backbone.layers):
            if not isinstance(layer, Glm5NextDecoderLayer):
                continue
            layer.register_forward_pre_hook(
                lambda m, args: positions.__setitem__(0, args[0])
            )
            layer.register_forward_hook(
                lambda m, args, out, idx=idx: record(idx, args[0], out)
            )
            if isinstance(layer.mlp, Glm5NextMoE) and idx in backbone.raw_moe:
                layer.mlp.register_forward_pre_hook(
                    lambda m, args: moe_in.__setitem__("x", args[0])
                )
                # the MoE runner calls the gate module for vLLM's router logits
                layer.mlp.gate.register_forward_hook(
                    lambda m, args, out: moe_in.__setitem__(
                        "logits", out[0] if isinstance(out, tuple) else out
                    )
                )
                layer.mlp.register_forward_hook(
                    lambda m, args, out, idx=idx: vllm_moe(idx, m)
                )
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        decode = model.specialized_decode
        if decode is None:
            return
        plan = decode.plan

        class _Step:
            def __init__(self, step):
                self.step = step

            @property
            def positions(self):
                return self.step.positions

            def layer(self, idx):
                kernel = self.step.layer(idx)
                if kernel is None:
                    return None

                def run(positions, *args):
                    out = kernel(positions, *args)
                    record(idx, positions, out, kernel=True)
                    # the kernel's FFN input and routing, from its mailbox
                    rt, n = self.step.kernels[idx].w.rt, len(positions)
                    x = _ffn_input(rt, rt.width)[:n]
                    moe_record(idx, positions, x, _routed(rt, rt.width)[:n])
                    return out

                return run

        def capturing_plan(positions, intermediate_tensors):
            step = plan(positions, intermediate_tensors)
            return None if step is None else _Step(step)

        decode.plan = capturing_plan

    def save_capture(self, path: str) -> None:
        captured = (self._captured, sorted(self._kernel_layers), self._moe)
        torch.save(captured, f"{path}.rank{self.rank}")  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """The checkpoint's config.json, cut to its first six layers and a small
    vocabulary; its quantization config is the checkpoint's own."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    config = get_hf_file_to_dict("config.json", MODEL)
    assert config is not None, f"no config.json for {MODEL}"
    text = config["text_config"]
    assert text["mlp_layer_types"][:LAYERS] == ["dense"] * 3 + ["sparse"] * 3
    assert text["layer_types"][3] == "deepseek_sparse_attention"
    assert text["layer_types"][4:LAYERS] == ["linear_attention"] * 2
    text.update(
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        layer_types=text["layer_types"][:LAYERS],
        mlp_layer_types=text["mlp_layer_types"][:LAYERS],
    )
    path = tmp_path_factory.mktemp("glm53_flash_mini")
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _batches(tp: int) -> tuple[int, ...]:
    """Every kernel step size at this TP size (TP2: up to 4 rows)."""
    return tuple(n for n in STEP_TOKENS if n <= TP_MAX_WIDTH.get(tp, max(STEP_TOKENS)))


def _requests(tp: int):
    set_random_seed(SEED)
    requests = []
    for n in _batches(tp):
        # short contexts attend every token; past 2048 the indexer selects
        sizes = [64, 2600, 3500, 1200, 4100, 300, 2900, 5000][:n]
        prompts = [
            {"prompt_token_ids": torch.randint(VOCAB, (size,)).tolist()}
            for size in sizes
        ]
        traces = [torch.randint(VOCAB, (STEPS,)).tolist() for _ in range(n)]
        requests.append((prompts, traces))
    return requests


def _decode(llm, tp: int, capture: str | None = None):
    """Full-vocabulary logprobs of every decode step (the first output token
    comes from prefill, which runs stock), every token forced; with `capture`
    each rank also saves each batch's decoder-layer outputs, the layers the
    kernels ran and their MoE references to `{capture}.b{batch}.rank{rank}`."""
    from vllm import SamplingParams

    rows = []
    for b, (prompts, traces) in enumerate(_requests(tp)):
        params = [
            SamplingParams(
                max_tokens=STEPS,
                temperature=0,
                logprobs=-1,
                trace_decode_token_ids=trace,
                detokenize=False,
                ignore_eos=True,
            )
            for trace in traces
        ]
        if capture:
            lens = [len(p["prompt_token_ids"]) for p in prompts]
            llm.collective_rpc("start_capture", args=(lens,))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if capture:
            llm.collective_rpc("save_capture", args=(f"{capture}.b{b}",))
        for out, trace in zip(outputs, traces):
            assert list(out.outputs[0].token_ids) == trace
            for step in out.outputs[0].logprobs[1:]:
                rows.append([step[v].logprob for v in range(VOCAB)])
    return torch.tensor(rows)


def _logit_errors(got, want):
    """Cosine over all rows, each row's relative L2 error of the logits
    (logprobs less their row mean) and the share of rows with the same top-1."""
    top1 = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    got, want = (x.float() - x.float().mean(-1, keepdim=True) for x in (got, want))
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    rows = (got - want).norm(dim=-1) / want.norm(dim=-1)
    print(
        f"cos {cos.item():.5f}, row median {rows.median():.2e} p90 "
        f"{rows.quantile(0.9):.2e} max {rows.max():.2e}, top-1 {top1:.3f}"
    )
    return cos.item(), rows, top1


def _assert_close(got, want):
    """The kernels quantize expert inputs to MXFP8 where vLLM's MoE uses MXFP4,
    so the logits are close but not equal. Seen on these weights, against vLLM
    in the same mode at TP2, TP4 and TP8: eager cos 0.9997, top-1 95-98%; FULL
    graphs (bf16 or e4m3 KV) cos 0.9995-1.0000, top-1 96-98%; the worst row at
    most 9.4%. A kernel layer that breaks moves every row."""
    cos, rows, top1 = _logit_errors(got, want)
    assert cos > 0.999 and top1 > 0.9 and rows.max() < 0.2


def _engine(vllm_runner, model_dir, tp, mono, graphs=False, kv="auto"):
    return vllm_runner(
        model_dir,
        load_format=LOAD_FORMAT,
        worker_extension_cls=f"{__name__}.Probe",
        skip_tokenizer_init=True,
        language_model_only=True,
        tensor_parallel_size=tp,
        max_num_seqs=max(_batches(tp)),
        max_model_len=8192,
        max_num_batched_tokens=32768,
        kv_cache_dtype=kv,
        kernel_config={"monokernel": mono},
        enforce_eager=not graphs,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY" if graphs else "NONE"},
        enable_prefix_caching=False,
        enable_trace_replay=True,
        max_logprobs=-1,
        gpu_memory_utilization=0.3,
        # ROCm's default, pinned: graphs capture on the null-block cache views and
        # replay after the cache grows to its full size (as fixed for Kimi-K3 in
        # #66370), which the kernels' cache binding must follow
        enable_extensible_kv_cache=True,
    )


@pytest.fixture(scope="module", autouse=True)
def _env():
    from vllm._aiter_ops import rocm_aiter_ops

    with pytest.MonkeyPatch.context() as mp:
        for name, value in ENV.items():
            mp.setenv(name, value)
        # rocm_aiter_ops read its switches at import, before ENV: without this
        # the first engine alone still sees AITER linear on (+quant_fp8)
        rocm_aiter_ops.refresh_env_variables()
        yield
    rocm_aiter_ops.refresh_env_variables()


_NATIVE: dict = {}


def _native(vllm_runner, model_dir, tp, graphs, kv="auto", capture=None):
    """The reference: vLLM's layers in the same mode, decoded once a module."""
    key = (tp, graphs, kv)
    if key not in _NATIVE:
        with _engine(vllm_runner, model_dir, tp, mono=False, graphs=graphs, kv=kv) as r:
            _NATIVE[key] = _decode(r.llm, tp, capture)
            assert r.llm.collective_rpc("mono_steps") == [0] * tp
    return _NATIVE[key]


def _mono(vllm_runner, model_dir, tp, graphs, kv="auto", capture=None):
    with _engine(vllm_runner, model_dir, tp, mono=True, graphs=graphs, kv=kv) as r:
        logprobs = _decode(r.llm, tp, capture)
        steps = r.llm.collective_rpc("mono_steps")
    print(f"MonoKernel steps per rank: {steps}")
    # eager plans every decode step in Python (all of a batch's but the first,
    # which may share a step with another request's prefill); FULL graphs plan
    # once per captured size and replay the launches, at least one per batch size
    least = len(_batches(tp)) if graphs else len(_batches(tp)) * (STEPS - 2)
    assert len(set(steps)) == 1 and steps[0] >= least, steps
    return logprobs


def _moe_refs(path: str, b: int, tp: int) -> dict:
    """A run's MoE references of batch b, per (layer, prompt length, position):
    (its expert ids, the fp64 router's ids, its margin, the fp64 MoE summed
    over the ranks' intermediate shards). Every rank routes alike."""
    parts = [torch.load(f"{path}.b{b}.rank{r}")[2] for r in range(tp)]
    refs = {}
    for key, (ids, ids64, gap, _) in parts[0].items():
        assert all(torch.equal(p[key][0], ids) for p in parts), (path, key)
        refs[key] = (ids, ids64, gap, sum(p[key][3] for p in parts))
    return refs


def _assert_layers_close(got: str, want: str, tp: int) -> None:
    """Every kernel layer on every rank, token by token, kernels (`got`) and
    vLLM's layers (`want`), as one Numerics table a rank (see BOUNDS): each
    run's MoE against fp64 on its own input and experts, the kernels' routing
    against the fp64 router, and the residual streams and last layer against
    vLLM's while both runs have routed alike. Each token's input to layer 3 is
    the same in both runs."""
    batches = _batches(tp)
    refs = [
        (_moe_refs(got, b, tp), _moe_refs(want, b, tp)) for b in range(len(batches))
    ]
    for rank in range(tp):
        num = Numerics(f"rank {rank}: kernel layers per decode token", BOUNDS, rank)
        same, flips, vllm_off = 0, [], 0
        requests = set()  # (batch, prompt length): a batch's lengths all differ
        for b, width in enumerate(batches):
            g, kernel_layers, _ = torch.load(f"{got}.b{b}.rank{rank}")
            w, _, _ = torch.load(f"{want}.b{b}.rank{rank}")
            assert kernel_layers == list(KERNEL_LAYERS), (rank, kernel_layers)
            g_moe, w_moe = refs[b]
            errs: dict[str, list[float]] = {}
            tokens = sorted({k[1:] for k in g} & {k[1:] for k in w})
            for k in tokens:
                assert all((i, *k) in g and (i, *k) in w for i in range(LAYERS)), k
                assert all(
                    torch.equal(x, y)
                    for i in range(KERNEL_LAYERS[0])
                    for x, y in zip(g[(i, *k)], w[(i, *k)])
                ), (rank, k, "input to layer 3 differs between the runs")
                same += 1
                requests.add((b, k[0]))
                alike = True  # both runs routed alike in every layer so far
                for i in KERNEL_LAYERS:
                    (x, res), (x_v, res_v) = g[(i, *k)], w[(i, *k)]
                    if alike:
                        errs.setdefault(f"L{i} residual vs vLLM", []).append(
                            rel(res, res_v)
                        )
                    ids, ids64, gap, ref = g_moe[(i, *k)]
                    assert torch.equal(ids, ids64) or gap < ROUTE_TIE, (
                        rank, i, k, "the kernels route other experts than fp64",
                        ids, ids64, gap,
                    )  # fmt: skip
                    v_ids, v_ids64, _, v_ref = w_moe[(i, *k)]
                    vllm_off += not torch.equal(v_ids, v_ids64)
                    errs.setdefault(f"L{i} MoE", []).append(rel(x, ref))
                    errs.setdefault(f"L{i} MoE vLLM", []).append(rel(x_v, v_ref))
                    if not torch.equal(ids, v_ids):
                        flips.append((i, *k, round(gap, 6)))
                        alike = False
                if alike:
                    last = LAYERS - 1
                    errs.setdefault(f"L{last} output vs vLLM", []).append(
                        rel(g[(last, *k)][0], w[(last, *k)][0])
                    )
            for stage in BOUNDS:
                if stage not in errs:
                    continue
                e_v = errs.get(f"{stage} vLLM")
                num.add_errors(stage, width, max(errs[stage]), e_v and min(e_v))
        print(
            f"rank {rank}: {same} tokens with equal inputs; experts the kernels "
            f"and vLLM route differently (layer, prompt, position, fp64 margin) "
            f"{flips}; vLLM off the fp64 router in {vllm_off} token-layers"
        )
        num.check()
        # every request's tokens compared, and few cross-run routing differences
        assert len(requests) == sum(batches), (rank, sorted(requests))
        assert len(flips) <= same // 4, (rank, flips)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
@pytest.mark.parametrize("tp", TPS)
def test_model_matches_native(vllm_runner, model_dir, tmp_path_factory, tp, graphs):
    """Teacher-forced decode with --monokernel against vLLM's layers: the
    decode steps take the kernels on every rank and the logits agree; eager
    also compares every decoder layer's outputs on every rank, token by token,
    as the DeepSeek-V4/V4.1 mono tests do (test_mono_decode_matches_the_model)."""
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    want = got = None
    if not graphs:
        base = tmp_path_factory.getbasetemp()
        want, got = f"{base}/native_tp{tp}", f"{base}/mono_tp{tp}"
    want_logprobs = _native(vllm_runner, model_dir, tp, graphs, capture=want)
    got_logprobs = _mono(vllm_runner, model_dir, tp, graphs, capture=got)
    _assert_close(got_logprobs, want_logprobs)
    if not graphs:
        _assert_layers_close(got, want, tp)


@pytest.mark.distributed(num_gpus=4)
def test_e4m3_kv_cache_matches_native(vllm_runner, model_dir, monkeypatch):
    """The same with an e4m3 KV cache (the kernel reads it back to bf16) under
    FULL graphs at TP4, against vLLM's layers on the same cache dtype."""
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    # vLLM's own fp8 sparse-MLA decode needs the Gluon path on ROCm
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", "1")
    want = _native(vllm_runner, model_dir, 4, True, kv="fp8")
    got = _mono(vllm_runner, model_dir, 4, True, kv="fp8")
    _assert_close(got, want)
