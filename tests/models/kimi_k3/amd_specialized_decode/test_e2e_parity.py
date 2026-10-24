# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode in the engine, on the 8-layer dummy Kimi-K3
(KDA layers 1, 2 and 4-6 run K1 + K2, MLA layers 3 and 7 vLLM's MLA then K2)
with its DSpark draft at TP8 and DCP8.

- layers: eager verify steps, every kernel layer against the decoder layer with
  vLLM's ops on the same inputs, KDA state slots and loaded weights;
- graphs: under FULL graphs every pure verify step runs the kernels on every
  rank as a graph replay, whether a request fills the 8-row graph (7
  speculative tokens) or one request of two replays it padded (3).
"""

import pytest

from tests.models.kimi_k3.amd_specialized_decode.test_numerics import needs_gfx950
from tests.utils import multi_gpu_test
from vllm import SamplingParams

MODEL = "riverclouds/Kimi-K3-8L-dummy"
DRAFT = "riverclouds/Kimi-K3-DSpark-for-8L-dummy"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "1, 2, 3, 4, 5,",
]
PARAMS = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
LAYER_LAUNCHES = 28

# For the layer test's relative errors: scales every float dummy weight as the
# dummy loader draws it, 1-D norm weights to U(0.9, 1.1), matrices to
# U(+-sqrt(3 / fan_in)), before vLLM's post-load processing.
_SITECUSTOMIZE = """\
import importlib.abc
import importlib.machinery
import sys

_MODULE = "vllm.model_executor.model_loader.dummy_loader"


def _patch(module):
    init = module.initialize_single_dummy_weight

    def scaled(param, *args, name="", **kwargs):
        init(param, *args, name=name, **kwargs)
        if (
            not param.is_floating_point()
            or param.element_size() == 1
            or getattr(param, "dummy_weight_value", None) is not None
        ):
            return
        if param.dim() == 1 and name == "weight":
            param.data.mul_(100).add_(1)
        elif param.dim() > 1:
            param.data.mul_(1e3 * (3 / param[0].numel()) ** 0.5)

    module.initialize_single_dummy_weight = scaled


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = patched_exec
        return spec


sys.meta_path.insert(0, _Finder())
"""


@pytest.fixture(scope="module", autouse=True)
def _env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        mp.setenv("VLLM_ROCM_USE_AITER", "1")
        mp.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "asm")
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        yield


def _decode(worker):
    return worker.model_runner.model_state.language_model.specialized_decode


def _scaled_weights(worker) -> bool:
    from vllm.model_executor.model_loader import dummy_loader

    return dummy_loader.initialize_single_dummy_weight.__name__ == "scaled"


def _count_steps(worker, q: int) -> None:
    """Counts the scheduled steps that are pure verify steps (every request
    scheduled with its ``q - 1`` draft tokens), and records the width of every
    step plan() hands the kernels: under FULL graphs plan() runs only for steps
    that are not replayed."""
    decode, runner = _decode(worker), worker.model_runner
    plan, decode.eager_widths = decode.plan, []
    execute, decode.verify_steps = runner.execute_model, 0

    def counted_plan(*args):
        step = plan(*args)
        if step is not None:
            decode.eager_widths.append(step.width)
        return step

    def counted_execute(scheduler_output, *args, **kwargs):
        spec = scheduler_output.scheduled_spec_decode_tokens
        if not kwargs.get("dummy_run") and all(
            n == q and len(spec.get(r, ())) == q - 1
            for r, n in scheduler_output.num_scheduled_tokens.items()
        ):
            decode.verify_steps += scheduler_output.total_num_scheduled_tokens > 0
        return execute(scheduler_output, *args, **kwargs)

    decode.plan, runner.execute_model = counted_plan, counted_execute


def _steps(worker) -> tuple[int, int, int]:
    """Forwards that ran the kernels (each bumps the device epoch once), those of
    them that were not replayed, and the pure verify steps."""
    decode = _decode(worker)
    return int(decode.runner.epoch[0]), len(decode.eager_widths), decode.verify_steps


def _engine(vllm_runner, num_spec: int, max_num_seqs: int, **kwargs):
    return vllm_runner(
        MODEL,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        dcp_comm_backend="a2a",
        language_model_only=True,
        load_format="dummy",
        kv_cache_dtype="fp8",
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.15,
        attention_config={"mla_prefill_backend": "ROCM_AITER_FA"},
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": num_spec,
            "attention_backend": "ROCM_AITER_MLA",
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        },
        additional_config={"decode_backend": "latency_specialized"},
        **kwargs,
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "num_spec, max_num_seqs, batches",
    [(7, 1, [PROMPTS]), (3, 2, [PROMPTS[:1], PROMPTS[1:]])],
    ids=["8_rows", "padded"],
)
def test_verify_steps_replay_the_kernels(vllm_runner, num_spec, max_num_seqs, batches):
    """Only 8-row FULL graphs are captured: every pure verify step, including
    a single request of 4 rows (``padded``, first batch), runs the kernels on
    every rank, each step a replay."""
    with _engine(
        vllm_runner,
        num_spec,
        max_num_seqs,
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [8],
        },
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_count_steps, args=(num_spec + 1,))
        for prompts in batches:
            before = llm.collective_rpc(_steps)
            llm.generate(prompts, PARAMS, use_tqdm=False)
            after = llm.collective_rpc(_steps)
            deltas = {tuple(a - b for a, b in zip(*ab)) for ab in zip(after, before)}
            assert len(deltas) == 1, deltas
            kernel_steps, eager, verify_steps = deltas.pop()
            assert kernel_steps == verify_steps > 0 and eager == 0


def _check_layers(worker, launches: int) -> None:
    """For the first ``launches`` kernel launches, runs the decoder layer with
    vLLM's ops first, on copies of the kernel's inputs and of the KDA state slots
    the step reads and writes, then the kernel on the originals, and records how
    they differ. ``x``: the MoE input K2 left in scratch against vLLM's; ``moe``:
    the kernel's MoE output against vLLM's MoE on that input one row at a time.
    vLLM's MoE is not batch invariant: at 8 rows its routed down projection
    takes a less exact GEMM than at 1-5, which the fp4 activations of its a4w4
    experts amplify to ~2.4% of the MoE output."""
    import torch

    from tests.models.kimi_k3.amd_specialized_decode.test_numerics import _k2_x
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import _k2_key

    decode = _decode(worker)
    decode.layer_errors = []

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def wrap(kernel):
        launch = kernel.launch

        def checked(step, positions, prefix_sum, block_residual, prefix_delta):
            if len(decode.layer_errors) >= launches:
                return launch(step, positions, prefix_sum, block_residual, prefix_delta)
            layer = kernel.layer
            kda = hasattr(kernel, "caches")
            if kda:
                m = step.attn_metadata[layer.self_attn.prefix]
                slots = m.spec_state_indices_tensor.flatten().long()
                caches = kernel.caches()
                before = [c[slots] for c in caches]
            hook = layer.mlp.register_forward_pre_hook(
                lambda _, args: setattr(decode, "moe_input", args[0].clone())
            )
            want = layer(
                positions,
                prefix_sum.clone(),
                block_residual.clone(),
                None if prefix_delta is None else prefix_delta.clone(),
            )
            hook.remove()
            if kda:
                want_states = [c[slots] for c in caches]
                for c, b in zip(caches, before):
                    c[slots] = b
            got = launch(step, positions, prefix_sum, block_residual, prefix_delta)
            s = got[2].size(0)
            key = _k2_key(layer, kernel.runner, s, layer.is_block_write_layer)
            x = _k2_x(kernel.runner, key).clone()
            moe = torch.cat([layer.mlp(x[r : r + 1]) for r in range(s)])
            errs = {
                "prefix": rel(got[0], want[0]),
                "blocks": torch.equal(got[1], want[1]),
                "x": rel(x, decode.moe_input),
                "moe": rel(got[2], moe),
            }
            if kda:
                errs["conv"], errs["rstate"] = (
                    rel(c[slots], w) for c, w in zip(caches, want_states)
                )
            decode.layer_errors.append((layer.layer_idx, errs))
            return got

        kernel.launch = checked

    for kernels in decode.kernels.values():
        for kernel in kernels.values():
            wrap(kernel)


def _layer_errors(worker):
    return _decode(worker).layer_errors


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_layer_kernels_match_vllm_layers(vllm_runner, tmp_path_factory):
    """Eager verify steps of two requests at 3 speculative tokens, on scaled
    dummy weights: the prefix, KDA states, untouched AttnRes blocks (layers 1-7
    sit in the first block) and MoE input and output of every kernel layer
    against vLLM's layer on every rank."""
    import os

    site = tmp_path_factory.mktemp("kimi_k3_scaled_dummy")
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    path = os.pathsep.join(filter(None, [str(site), os.getenv("PYTHONPATH")]))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", path)
        with _engine(vllm_runner, 3, 2, enforce_eager=True) as runner:
            llm = runner.llm
            assert all(llm.collective_rpc(_scaled_weights))
            llm.collective_rpc(_check_layers, args=(LAYER_LAUNCHES,))
            llm.generate(PROMPTS[:2], PARAMS, use_tqdm=False)
            ranks = llm.collective_rpc(_layer_errors)
    for errors in ranks:
        assert len(errors) == LAYER_LAUNCHES
        assert {i for i, _ in errors} == set(range(1, 8))
        for i, errs in errors:
            assert errs["blocks"], i
            assert errs["moe"] < 6e-3, (i, errs)
            if i in (3, 7):
                assert errs["prefix"] < 2e-4 and errs["x"] < 2e-4, (i, errs)
            else:
                assert errs["prefix"] < 6e-3 and errs["x"] < 6e-3, (i, errs)
                assert errs["conv"] < 6e-4 and errs["rstate"] < 5e-3, (i, errs)
