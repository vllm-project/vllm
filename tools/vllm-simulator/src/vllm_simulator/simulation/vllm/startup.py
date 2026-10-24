"""
Installs all vLLM simulation hooks.

init_hook() must be called BEFORE any `import vllm.*` statement.
"""

import logging
import os


def _patch_triton_for_cpu():
    """Keep the real triton module on CPU-only hosts.

    Without a GPU driver some vLLM versions use TritonLanguagePlaceholder,
    which lacks ``extra.cuda``.  An empty CUDA_VISIBLE_DEVICES keeps
    HAS_TRITON=True; pre-importing triton.language.extra.cuda is a safety net.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import triton.language.extra.cuda  # noqa: F401
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass


# Guards against a double install (several simulator entrypoints imported).
_HOOKS_INSTALLED = False


def init_hook():
    """Install all vLLM hooks; must run before importing vllm.  Idempotent."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    import vllm_simulator.hook as vllm_simulator_hook
    from vllm_simulator.simulation.vllm import (
        decode_bench,
        engine_args,
        engine_core_pipeline,
        kv_offload,
        platform,
        worker,
    )

    # Fixed NONE_HASH seed so block hashes match across nodes.
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '42'
        logging.getLogger('vllm_simulator').warning(
            '[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency')

    _patch_triton_for_cpu()

    # The stub stands in for Model Runner V2; also selects the V2 scheduler
    # output layout (resumed requests are sent as new requests).
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    hooks = [
        platform.C_VLLMPlatformHook,
        engine_args.C_VLLMEngineArgsHook,
        worker.C_VLLMWorkerHook,
        engine_core_pipeline.C_VLLMEngineCoreHook,
        engine_core_pipeline.C_VLLMSchedulerHook,
        engine_core_pipeline.C_VLLMExecutorHook,
        kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,
        kv_offload.C_VLLMOffloadingConnectorWorkerHook,
        decode_bench.C_VLLMDecodeBenchConnectorWorkerHook,
    ]

    vllm_simulator_hook.install_class_hooks(hooks)

    _HOOKS_INSTALLED = True
