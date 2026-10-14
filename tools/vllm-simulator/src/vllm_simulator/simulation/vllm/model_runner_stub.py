"""
CPU stand-in for GPUModelRunnerV2 (vllm/v1/worker/gpu/model_runner.py),
driven by the native Worker.

KV tensors are one page each; page sizes stay at real model values since
they are accounting-only.
"""

import functools
import os

import torch
from torch.overrides import TorchFunctionMode

from vllm_simulator.simulation.manager import ConfigManager
from vllm_simulator.utils import get_logger

logger = get_logger()


def _noop(*args, **kwargs):
    """Module-level so it is picklable."""
    return None


class _ModelRunnerStub:
    """Picklable model_runner stub; other attributes resolve to no-ops."""

    # Read by Worker.execute_model under the V2 runner.
    is_pooling_model = False

    def __init__(self, vllm_config, kv_spec: dict):
        from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

        self.vllm_config = vllm_config
        self._kv_spec = kv_spec
        self.kv_connector = NO_OP_KV_CONNECTOR
        self.model_memory_usage = 0
        # Explicit None (not the __getattr__ no-op) so LLMEngine skips its
        # model cleanup finalizer.
        self.model = None

    def get_kv_cache_spec(self):
        return self._kv_spec

    def get_supported_tasks(self):
        return ("generate",)

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        """Allocate one-page CPU KV tensors and hand them to the V2 KV
        connector wrapper."""
        from vllm.v1.worker.gpu.kv_connector import get_kv_connector

        kv_caches: dict = {}
        for kv_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                min(kv_tensor.size, 4096), dtype=torch.int8, device="cpu"
            )
            for layer_name in kv_tensor.layers:
                kv_caches[layer_name] = tensor
        logger.info(
            "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
            "(num_blocks=%d, simulated_bytes=%d, actual_bytes=4k)",
            len(kv_cache_config.kv_cache_tensors),
            kv_cache_config.num_blocks,
            sum(t.size for t in kv_cache_config.kv_cache_tensors),
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches)

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        """Mock forward inside the V2 KV connector lifecycle.

        Returns the output directly (no sample_tokens RPC); finish_forward
        is skipped since only HiSparse connectors implement it.
        """
        from vllm.v1.outputs import ModelRunnerOutput

        if not scheduler_output.num_scheduled_tokens:
            return self.kv_connector.no_forward(scheduler_output)

        self.kv_connector.pre_forward(scheduler_output)
        kv_connector_output = self.kv_connector.post_forward(
            scheduler_output.finished_req_ids
        )

        # Whether each request samples a token this step comes from
        # C_VLLMExecutorHook (_sim_token_emitted).
        req_ids = list(scheduler_output.num_scheduled_tokens)
        token_emitted = scheduler_output._sim_token_emitted
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[
                [1] if token_emitted[rid] else [] for rid in req_ids
            ],
            kv_connector_output=kv_connector_output,
        )

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _noop


class _ForceMetaDevice(TorchFunctionMode):
    """Redirect explicit ``device=`` arguments to meta; the plain
    ``torch.device("meta")`` context only fills in a missing device."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("device") is not None:
            kwargs["device"] = "meta"
        return func(*args, **kwargs)


class _NullCudaStream:
    """Stands in for torch.cuda.Stream; aux streams are never used on
    the meta build."""

    def __init__(self, *args, **kwargs):
        pass


@functools.cache
def _simulated_device_capability():
    """Capability of the simulated accelerator, read from its
    AIConfigurator system spec; A100 when the system is unknown."""
    import yaml
    from aiconfigurator.sdk.perf_database import get_systems_paths

    from vllm.platforms.interface import DeviceCapability

    name = ConfigManager.get_accelerator_info().name
    for path in get_systems_paths():
        spec_path = os.path.join(path, f"{name}.yaml")
        if os.path.exists(spec_path):
            with open(spec_path) as f:
                sm_version = yaml.safe_load(f)["gpu"]["sm_version"]
            return DeviceCapability(*divmod(sm_version, 10))
    return DeviceCapability(8, 0)


def build_kv_cache_spec(vllm_config) -> dict:
    """Derive the KV cache spec like GPUModelRunnerV2: build the model on
    the meta device and collect its attention layers' specs.

    Built under the simulated TP size so specs describe one rank of the
    target deployment.  Requires the distributed env (set up in
    init_device).
    """
    import vllm.distributed.parallel_state as ps
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    # The engine runs at tp=1 (EngineArgs hook); the simulated tp lives in
    # the simulator scheduler config.
    sim_tp = ConfigManager.get_scheduler_config().tp_size

    tp_group = ps.get_tp_group()
    orig_world = tp_group.world_size
    orig_cfg_tp = vllm_config.parallel_config.tensor_parallel_size
    orig_oot = current_platform.is_out_of_tree
    orig_ep = getattr(ps, "_EP", None)

    # Scope-patches for the meta build:
    # 1. Shard sizes come from get_tp_group().world_size, not
    #    parallel_config; no communication happens on the meta device.
    # 2. The unquantized-MoE backend oracle raises UnboundLocalError on
    #    a platform that is none of cuda/rocm/xpu/cpu.  Out-of-tree
    #    makes it return (OOT, None); experts_cls is unused here.
    # 3. _EP exists only if the first engine in the process was MoE.
    #    Lend it the TP group: MoE layers contribute no KV spec.
    # 4. Sparse-attention models (DSA, DeepSeek-V4, Kimi-K3) allocate
    #    buffers with an explicit CUDA device and create aux CUDA
    #    streams; no CUDA device is visible to the simulator.
    # 5. Attention backends and quantized kernels are chosen by device
    #    capability and platform and decide the KV cache layout: build
    #    (and collect the specs) as CUDA on the simulated accelerator.
    #    The compute-unit count only sizes kernel workspaces.
    platform_cls = type(current_platform)
    platform_patches = {
        "get_device_capability": classmethod(
            lambda cls, device_id=0: _simulated_device_capability()
        ),
        "is_cuda": lambda self: True,
        "get_cuda_runtime_major": classmethod(
            lambda cls: int((torch.version.cuda or "0").split(".")[0])
        ),
        "num_compute_units": classmethod(lambda cls, device_id=0: 132),
    }
    orig_platform = {
        name: vars(platform_cls).get(name) for name in platform_patches
    }
    orig_stream = torch.cuda.Stream
    current_platform.is_out_of_tree = lambda: True
    torch.cuda.Stream = _NullCudaStream
    for name, attr in platform_patches.items():
        setattr(platform_cls, name, attr)
    if ps._EP is None:
        ps._EP = tp_group
    if sim_tp > 1:
        tp_group.world_size = sim_tp
        vllm_config.parallel_config.tensor_parallel_size = sim_tp
    try:
        try:
            with (
                set_default_torch_dtype(vllm_config.model_config.dtype),
                torch.device("meta"),
                _ForceMetaDevice(),
            ):
                initialize_model(vllm_config)
        finally:
            tp_group.world_size = orig_world
            vllm_config.parallel_config.tensor_parallel_size = orig_cfg_tp
            current_platform.is_out_of_tree = orig_oot
            torch.cuda.Stream = orig_stream
            ps._EP = orig_ep
        # The executor resizes blocks for the backends only after
        # init_device; the specs are collected here, so resize first.
        current_platform.update_block_size_for_backend(vllm_config)
        kv_cache_spec = get_kv_cache_spec(vllm_config)
    finally:
        for name, attr in orig_platform.items():
            if attr is None:
                delattr(platform_cls, name)
            else:
                setattr(platform_cls, name, attr)

    logger.info(
        "[vLLM Hijack] Built KV cache spec via meta-device model walk: "
        "%d layers (sim tp=%d)", len(kv_cache_spec), sim_tp,
    )
    return kv_cache_spec
