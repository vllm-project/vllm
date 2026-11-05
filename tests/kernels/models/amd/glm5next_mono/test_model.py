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
layers in the same mode; eager also compares every decoder layer's outputs for
each decode token, as the DeepSeek-V4.1 mono test does. Prompts mix short
contexts with ones past the indexer's 2048 tokens, so the sparse layer
selects."""

import io
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
# per-layer bounds on the relative error over all decode tokens, kernels against
# vLLM's layers (eager). Two runs of vLLM's layers already differ by up to
# (TP2/TP4): hidden 0.019, 0.052, 0.081, 0.29, 0.29, 0.055 and residual 0.001,
# 0.019, 0.038, 0.055, 0.055 for layers 0-5; with equal inputs to layer 3 the
# kernels' residual is within 0.001 of vLLM's, their MoE output within 0.22
# (MXFP8 against MXFP4 inputs). Layer 5 has no residual.
LAYER_BOUNDS = {
    0: {"hidden": 0.05, "residual": 0.01},
    1: {"hidden": 0.1, "residual": 0.04},
    2: {"hidden": 0.15, "residual": 0.08},
    3: {"hidden": 0.45, "residual": 0.1},
    4: {"hidden": 0.45, "residual": 0.1},
    5: {"hidden": 0.1},
}
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
def _fill(name: str, t: torch.Tensor, fan_in: int) -> None:
    """A random value for every weight in the checkpoint's format, seeded by its
    name (alike in every run, so both engines load the same model)."""
    if t.device.type == "meta" or not t.numel():
        return
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype == torch.uint8 and "scale" in name:
        # E8M0 2^-9 .. 2^-7: unit gate / up pre-activations
        t.copy_(torch.randint(118, 121, t.shape, generator=g, device=t.device))
    elif t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(randn().clamp(-6, 6))
    elif not torch.is_floating_point(t):
        return
    elif name.endswith("weight_scale_inv"):
        # FP8 block-128 scales of unit-normal FP8 values
        t.fill_(fan_in**-0.5)
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
    elif t.dim() == 1:
        t.copy_(randn(0.02))
    else:
        t.copy_(randn(fan_in**-0.5))


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading with weights in the checkpoint's formats and ranges, before
    post-load processing, so every processed copy follows (plain dummy weights
    are near zero and leave the E8M0 scales zero)."""

    def load_weights(self, model, model_config) -> None:
        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            _fill(name, t, t.shape[-1] if t.dim() > 1 else 1)


class Probe:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers."""

    def mono_steps(self) -> int:
        from vllm.models.glm5next.amd.specialized_decode.runtime import failstop

        return failstop.counters["mono_steps"]

    def start_capture(self) -> None:
        """Keep every decoder layer's outputs (hidden states, residual) for each
        decode token of a decode-only step, keyed by (layer, position): vLLM's
        layers through forward hooks, the kernel layers by wrapping the launch
        ``plan()`` returns. Eager only (hooks do not run in graph replay)."""
        from vllm.forward_context import get_forward_context
        from vllm.models.glm5next.common.model import (
            Glm5NextDecoderLayer,
            Glm5NextModel,
        )

        self._captured: dict = {}
        self._kernel_layers: set = set()
        if getattr(self, "_hooked", False):
            return
        self._hooked = True

        def decode_only() -> bool:
            md = get_forward_context().attn_metadata
            return isinstance(md, dict) and all(
                getattr(m, "num_prefills", 0) == 0 for m in md.values()
            )

        def record(idx, positions, out, kernel=False):
            if not decode_only():
                return
            if kernel:
                self._kernel_layers.add(idx)
            # the last layer applies its own hc_post: no residual
            for row, p in enumerate(positions.tolist()):
                self._captured.setdefault(
                    (idx, p),
                    [None if t is None else t[row].float().cpu() for t in out[:2]],
                )

        model = self.model_runner.model  # type: ignore[attr-defined]
        backbone = next(m for m in model.modules() if isinstance(m, Glm5NextModel))
        for idx, layer in enumerate(backbone.layers):
            if isinstance(layer, Glm5NextDecoderLayer):
                layer.register_forward_hook(
                    lambda m, args, out, idx=idx: record(idx, args[0], out)
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

    def take_capture(self) -> bytes:
        # one torch.save blob: the RPC encoder sends nested tensors as stubs
        buf = io.BytesIO()
        torch.save((self._captured, sorted(self._kernel_layers)), buf)
        return buf.getvalue()


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


def _decode(llm, tp: int, capture: bool = False):
    """Full-vocabulary logprobs of every decode step (the first output token
    comes from prefill, which runs stock), every token forced; with ``capture``
    also each batch's decoder-layer outputs and the layers the kernels ran."""
    from vllm import SamplingParams

    rows, layers = [], []
    for prompts, traces in _requests(tp):
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
            llm.collective_rpc("start_capture")
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if capture:
            blob = llm.collective_rpc("take_capture")[0]
            layers.append(torch.load(io.BytesIO(blob)))
        for out, trace in zip(outputs, traces):
            assert list(out.outputs[0].token_ids) == trace
            for step in out.outputs[0].logprobs[1:]:
                rows.append([step[v].logprob for v in range(VOCAB)])
    return torch.tensor(rows), layers


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
    so the logits are close but not equal. Seen on these weights over four runs
    on three nodes, against vLLM in the same mode: eager cos 0.9981-0.9993,
    top-1 89-98%; FULL graphs (bf16 or e4m3 KV) cos 0.9984-0.9990, top-1 88-95%;
    the worst row at most 12% in every mode.
    vLLM's own FULL graphs differ from its eager path by more (cos 0.9977,
    top-1 83%). A kernel layer that breaks moves every row."""
    cos, rows, top1 = _logit_errors(got, want)
    assert cos > 0.997 and top1 > 0.8 and rows.max() < 0.2


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


def _native(vllm_runner, model_dir, tp, graphs, kv="auto"):
    """The reference: vLLM's layers in the same mode, decoded once a module."""
    key = (tp, graphs, kv)
    if key not in _NATIVE:
        with _engine(vllm_runner, model_dir, tp, mono=False, graphs=graphs, kv=kv) as r:
            _NATIVE[key] = _decode(r.llm, tp, capture=not graphs)
            assert r.llm.collective_rpc("mono_steps") == [0] * tp
    return _NATIVE[key]


def _mono(vllm_runner, model_dir, tp, graphs, kv="auto"):
    with _engine(vllm_runner, model_dir, tp, mono=True, graphs=graphs, kv=kv) as r:
        out = _decode(r.llm, tp, capture=not graphs)
        steps = r.llm.collective_rpc("mono_steps")
    print(f"MonoKernel steps per rank: {steps}")
    # eager plans every decode step in Python (all of a batch's but the first,
    # which may share a step with another request's prefill); FULL graphs plan
    # at capture and replay the captured launches
    least = 1 if graphs else len(_batches(tp)) * (STEPS - 2)
    assert len(set(steps)) == 1 and steps[0] >= least, steps
    return out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _assert_layers_close(got, want):
    """Each decoder layer's outputs (hidden states, residual) for every decode
    token both engines ran in a decode-only step, kernels against vLLM's
    layers. Layers 0-2 run vLLM's layers on both sides; 3 and 4 are the kernel
    launches; 5 takes their output. The error is over all tokens at once: one
    row alone moves by as much between two runs of vLLM's layers."""
    rows: dict = {}
    for (g, kernel_layers), (w, _) in zip(got, want, strict=True):
        assert kernel_layers == list(KERNEL_LAYERS), kernel_layers
        keys = g.keys() & w.keys()
        assert {layer for layer, _ in keys} == set(range(LAYERS)), sorted(keys)
        for key in keys:
            for name, a, b in zip(("hidden", "residual"), g[key], w[key]):
                assert (a is None) == (b is None), (key, name)
                if a is not None:
                    rows.setdefault((key[0], name), []).append((a, b))
    errors = {
        key: _rel(torch.stack([a for a, _ in ab]), torch.stack([b for _, b in ab]))
        for key, ab in sorted(rows.items())
    }
    print("relative error per layer:", errors)
    assert set(errors) == {(i, n) for i, b in LAYER_BOUNDS.items() for n in b}
    for (layer, name), err in errors.items():
        assert err < LAYER_BOUNDS[layer][name], (layer, name, err)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
@pytest.mark.parametrize("tp", TPS)
def test_model_matches_native(vllm_runner, model_dir, tp, graphs):
    """Teacher-forced decode with --monokernel against vLLM's layers: the
    decode steps take the kernels on every rank and the logits agree; eager
    also compares every decoder layer's outputs, token by token."""
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    want, want_layers = _native(vllm_runner, model_dir, tp, graphs)
    got, got_layers = _mono(vllm_runner, model_dir, tp, graphs)
    _assert_close(got, want)
    if not graphs:
        _assert_layers_close(got_layers, want_layers)


@pytest.mark.distributed(num_gpus=4)
def test_e4m3_kv_cache_matches_native(vllm_runner, model_dir, monkeypatch):
    """The same with an e4m3 KV cache (the kernel reads it back to bf16) under
    FULL graphs at TP4, against vLLM's layers on the same cache dtype."""
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    # vLLM's own fp8 sparse-MLA decode needs the Gluon path on ROCm
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", "1")
    want, _ = _native(vllm_runner, model_dir, 4, True, kv="fp8")
    got, _ = _mono(vllm_runner, model_dir, 4, True, kv="fp8")
    _assert_close(got, want)
