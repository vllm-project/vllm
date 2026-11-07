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
layers in the same mode; eager also compares every decoder layer's outputs on
every rank for each decode token, as the DeepSeek-V4/V4.1 mono tests do
(``test_mono_decode_matches_the_model``), with their weight ranges. Prompts mix
short contexts with ones past the indexer's 2048 tokens, so the sparse layer
selects."""

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

from ._ffn import _on_cdna4

MODEL = "amd/GLM-5.3-Flash-Quark-MXFP4"
LOAD_FORMAT = "glm5next_mono_random"
LAYERS = 6
KERNEL_LAYERS = (3, 4)
VOCAB = 4096
STEPS = 8
SEED = 0
# Per decode token, kernels against vLLM's layers (eager), for the tokens whose
# input to layer 3 is the same in both runs: layers 0-2 run vLLM's code on both
# sides and gave identical outputs. The others are all of one request's tokens
# in one run: its prefill ran otherwise there (requests reach the engine core
# while it runs), which moves its KDA state and so every later token a little.
# - residual stream after layers 3 and 4: 0.0044 seen at TP2, TP4 and TP8;
RESIDUAL_BOUND = 0.01
# - layer 5's output, vLLM's MoE on the kernels' streams: 0.0065 seen;
LAST_BOUND = 0.02
# - the MoE output of layers 3 and 4: vLLM quantizes expert inputs to MXFP4, the
#   kernels to MXFP8, which puts vLLM 0.21 and the kernels 0.047 from an fp32
#   MoE (test_ffn_matches_glm5next_moe), so the two differ by 0.22-0.28 a token;
MOE_BOUND = 0.3
# - past MOE_BOUND only a token whose 8th and 9th routing scores on vLLM's path
#   are within TIE: the residual streams' 0.1-0.4% difference moves scores by
#   about 1e-3, so it may pick another expert (seen: 0.33-0.43 at gaps up to
#   1.1e-3, at most 5 of 89 tokens), and later layers are exempt for it.
TIE = 2e-3
ENV = {
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    # deterministic BF16 GEMMs for the stock layers: the split-K skinny and
    # AITER tuned GEMMs reduce in a run-dependent order
    "VLLM_ROCM_USE_SKINNY_GEMM": "0",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
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
        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            _fill(name, t)


class Probe:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers."""

    def mono_steps(self) -> int:
        from vllm.models.glm5next.amd.specialized_decode.runtime import failstop

        return failstop.counters["mono_steps"]

    def start_capture(self, lens: list[int]) -> None:
        """Keep every decoder layer's outputs (hidden states, residual) for each
        decode token of a decode-only step, keyed by (layer, prompt length,
        position), as the DeepSeek-V4.1 mono test does: a token's step and row
        can differ between engines, its prompt and position cannot (``lens``:
        the batch's prompt lengths, more than STEPS apart). vLLM's layers are
        read through forward hooks, the kernel layers by wrapping the launch
        ``plan()`` returns; vLLM's MoE also keeps each token's margin between
        its last chosen and first unchosen routing score. Eager only (hooks do
        not run in graph replay)."""
        from vllm.forward_context import get_forward_context
        from vllm.models.glm5next.common.model import (
            Glm5NextDecoderLayer,
            Glm5NextModel,
            Glm5NextMoE,
        )

        self._captured: dict = {}
        self._gaps: dict = {}
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

        def record(idx, positions, out, kernel=False):
            if not decode_only():
                return
            if kernel:
                self._kernel_layers.add(idx)
            # the last layer applies its own hc_post: no residual
            for row, key in keys(idx, positions):
                self._captured.setdefault(
                    key, [None if t is None else t[row].float().cpu() for t in out[:2]]
                )

        positions: list = [None]  # the running decoder layer's

        def margin(idx, moe, h):
            if not decode_only():
                return
            gate = moe.gate
            scores = torch.sigmoid(h.float() @ gate.weight.float().T)
            if gate.e_score_correction_bias is not None:
                scores = scores + gate.e_score_correction_bias.float()
            top = scores.topk(top_k + 1, dim=-1).values
            gap = (top[:, top_k - 1] - top[:, top_k]).tolist()
            for row, key in keys(idx, positions[0]):
                self._gaps.setdefault(key, gap[row])

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
            if isinstance(layer.mlp, Glm5NextMoE):
                layer.mlp.register_forward_pre_hook(
                    lambda m, args, idx=idx: margin(idx, m, args[0])
                )
        decode = backbone.specialized_decode
        if decode is None:
            return
        plan = decode.plan

        class _Step:
            def __init__(self, step):
                self.step = step

            def layer(self, idx):
                kernel = self.step.layer(idx)
                if kernel is None:
                    return None

                def run(positions, *args):
                    out = kernel(positions, *args)
                    record(idx, positions, out, kernel=True)
                    return out

                return run

        def capturing_plan(positions, intermediate_tensors):
            step = plan(positions, intermediate_tensors)
            return None if step is None else _Step(step)

        decode.plan = capturing_plan

    def save_capture(self, path: str) -> None:
        captured = (self._captured, self._gaps, sorted(self._kernel_layers))
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
    comes from prefill, which runs stock), every token forced; with ``capture``
    each rank also saves each batch's decoder-layer outputs, routing margins and
    the layers the kernels ran to ``{capture}.b{batch}.rank{rank}``."""
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
    )


@pytest.fixture(scope="module", autouse=True)
def _env():
    with pytest.MonkeyPatch.context() as mp:
        for name, value in ENV.items():
            mp.setenv(name, value)
        yield


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
    # at capture and replay the captured launches
    least = 1 if graphs else len(_batches(tp)) * (STEPS - 2)
    assert len(set(steps)) == 1 and steps[0] >= least, steps
    return logprobs


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _assert_layers_close(got: str, want: str, tp: int) -> None:
    """Every decoder layer's outputs on every rank, token by token, kernels
    (``got``) against vLLM's layers (``want``), for the decode tokens whose
    input to layer 3 is the same in both runs (see RESIDUAL_BOUND)."""
    batches = _batches(tp)
    for rank in range(tp):
        same, flipped, seen = 0, [], {"residual": 0.0, "moe": 0.0, "last": 0.0}
        for b in range(len(batches)):
            g, _, kernel_layers = torch.load(f"{got}.b{b}.rank{rank}")
            w, gaps, _ = torch.load(f"{want}.b{b}.rank{rank}")
            assert kernel_layers == list(KERNEL_LAYERS), (rank, kernel_layers)
            tokens = sorted({k[1:] for k in g} & {k[1:] for k in w})
            for k in tokens:
                assert all((i, *k) in g and (i, *k) in w for i in range(LAYERS)), k
                if not all(
                    torch.equal(x, y)
                    for i in range(KERNEL_LAYERS[0])
                    for x, y in zip(g[(i, *k)], w[(i, *k)])
                ):
                    continue
                same += 1
                for i in KERNEL_LAYERS:
                    (x, res), (x_ref, res_ref) = g[(i, *k)], w[(i, *k)]
                    err = _rel(res, res_ref)
                    seen["residual"] = max(seen["residual"], err)
                    assert err < RESIDUAL_BOUND, (rank, i, k, "residual", err)
                    err = _rel(x, x_ref)
                    if err >= MOE_BOUND:
                        assert gaps[(i, *k)] < TIE, (rank, i, k, "moe", err, gaps)
                        flipped.append((i, *k, round(err, 3), gaps[(i, *k)]))
                        break  # its residual stream differs from here on
                    seen["moe"] = max(seen["moe"], err)
                else:
                    err = _rel(g[(LAYERS - 1, *k)][0], w[(LAYERS - 1, *k)][0])
                    seen["last"] = max(seen["last"], err)
                    assert err < LAST_BOUND, (rank, k, "last layer", err)
        print(
            f"rank {rank}: {same} tokens with equal inputs, worst {seen}, "
            f"near-tie flips {flipped}"
        )
        # at least one equal-input token a request, and few flips
        assert same >= sum(batches), (rank, same)
        assert len(flipped) <= same // 4, (rank, flipped)


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
