# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import math
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional

import torch
import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms.tt import (
    TTPlatform,
    _should_pre_register_tt_test_models_from_cli,
    register_tt_models,
)
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.tt_model_runner import TTModelInput, TTModelRunner
from vllm.v1.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import LogprobsLists

logger = init_logger(__name__)

# Ensure TT model architectures are registered in this process as early as
# possible. `WorkerWrapperBase.init_worker` imports the worker class module
# before initializing multimodal caches; without this, early architecture
# inspection may fail for TT-prefixed architectures.
register_tt_models(register_test_models=_should_pre_register_tt_test_models_from_cli())


class TTWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        # Initialized by init_device
        self.mesh_device = None

        # Whether to use ttnn tracing for model execution
        override_tt_config = self.model_config.override_tt_config
        trace_key = "trace_mode"
        self.trace_mode = "all"
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in ["decode_only", "all", "none"], (
                f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            )
            self.trace_mode = override_tt_config[trace_key]

        enable_model_warmup_key = "enable_model_warmup"
        self.enable_model_warmup = True
        if override_tt_config and enable_model_warmup_key in override_tt_config:
            assert override_tt_config[enable_model_warmup_key] in [True, False], (
                f"Invalid {enable_model_warmup_key}: \
                {override_tt_config[enable_model_warmup_key]}"
            )

            self.enable_model_warmup = override_tt_config[enable_model_warmup_key]

    def init_device(self) -> None:
        # Validate/apply TT config in this worker process (multiprocessing
        # means platform class attrs + config mutations must be applied per
        # subprocess) before runner init.
        TTPlatform.check_and_update_config(self.vllm_config)

        local_dp_rank = self.parallel_config.data_parallel_rank_local
        # Open mesh only on local DP rank 0 (device ranks).
        if local_dp_rank == 0:
            self.mesh_device = open_mesh_device(
                self.model_config.override_tt_config, self.trace_mode, local_dp_rank
            )
            self.device_config.device = self.mesh_device
            assert self.mesh_device is not None
            self.device_config.num_devices = self.mesh_device.get_num_devices()
        else:
            mesh_grid = get_mesh_grid(local_dp_rank)
            self.mesh_device = None
            # Num devices is required for determining num blocks in KV cache.
            self.device_config.num_devices = mesh_grid[0] * mesh_grid[1]
        # Init ModelRunner here, so that we have access to self.mesh_device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            mesh_device=self.mesh_device,
            trace_mode=self.trace_mode,
            enable_model_warmup=self.enable_model_warmup,
        )

    def load_model(self):
        # Only local DP rank 0 (device rank) loads the model
        if self.parallel_config.data_parallel_rank_local == 0:
            self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        For the GPU/TPU backends, this method generates the KVCacheSpec by
        parsing the kv cache format from each Attention module in the static
        forward context (compilation_config.static_forward_context).
        core/kv_cache_utils.py uses the KVCacheSpec along with available
        memory info from a profiling run to determine num blocks.

        For the TT backend, the static forward context is not populated since
        the modelling code is independent so we currently skip creating a
        kv cache spec for each layer, similar to the Spyre/Neuron backends.
        Currently we also don't run profiling to determine available memory.

        Return a dummy single layer KVCacheSpec and in the
        determine_available_memory function override num blocks using
        self.cache_config.num_gpu_blocks_override.
        """

        # TODO: Once we're able to populate a static forward context,
        # generate separate specs per layer (e.g. also sliding window, local
        # attention).

        model_config = self.model_config
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        # Excludes TP factor since that is handled on the model side for TT.
        total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )

        use_mla = model_config.use_mla
        sliding_window = model_config.get_sliding_window()
        if use_mla:
            assert not sliding_window, "MLA not supported for sliding window"
            attn_spec = MLAAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
            )
        else:
            attn_spec = FullAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                sliding_window=sliding_window,
            )
        kv_cache_spec: dict[str, KVCacheSpec] = {"foo": attn_spec}
        return kv_cache_spec

    def determine_available_memory(self) -> int:
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).

        Currenly we just return a large dummy number of bytes similar to the
        Spyre/Neuron backends and override the number of kv cache blocks.
        """

        # TODO: Once we can run profiling, return real available memory
        # instead of overriding the number of blocks.
        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 1 << 64

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache (only DP rank 0) and initialize persistent
        input batch (all DP ranks) with the specified kv_cache_config.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Cache is already initialized in initialize_from_config.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        if not self.enable_model_warmup:
            logger.warning("Skipping model warmup")
            return
        local_rank = self.parallel_config.data_parallel_rank_local
        if local_rank == 0:
            self.model_runner.warmup_model()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        assert self.is_driver_worker, "There should only be one Worker for TT"
        output = self.model_runner.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    # ---- DP gather hooks called by DPEngineCoreProc in core.py ----

    def build_dp_model_input(
        self, scheduler_output: Optional["SchedulerOutput"]
    ) -> tuple[TTModelInput | None, int, int, int, int, int, int]:
        """Called by each DP rank to build model input from scheduler output.
        Returns: (model_input, max_blocks, has_structured_input, has_penalties,
        reset_batch, can_sample_device, needs_logprobs)
        """
        model_input = None
        has_penalties = 0
        reset_batch = 0
        can_sample_device = 1
        needs_logprobs = 0
        if scheduler_output is not None:
            model_input = self.model_runner.build_model_input(scheduler_output)
            if model_input is not None:
                has_penalties = int(not self.model_runner.input_batch.no_penalties)
                reset_batch = int(model_input.reset_batch)
                can_sample_device = int(model_input.perform_device_sampling)
                needs_logprobs = int(model_input.max_num_logprobs is not None)
        max_blocks = model_input.block_tables.shape[1] if model_input else 0
        has_structured_input = (
            int(model_input.grammar_bitmask[0] is not None) if model_input else 0
        )
        return (
            model_input,
            max_blocks,
            has_structured_input,
            has_penalties,
            reset_batch,
            can_sample_device,
            needs_logprobs,
        )

    def build_dp_decode_gather_input(
        self,
        model_input: TTModelInput | None,
        max_blocks_decode_batch: int,
        any_structured_inputs: bool,
        any_penalties_inputs: bool,
    ) -> dict[str, Any]:
        return self.model_runner.build_dp_decode_gather_input(
            model_input,
            max_blocks_decode_batch,
            any_structured_inputs,
            any_penalties_inputs,
        )

    def concat_and_execute_dp(
        self,
        inputs: list[TTModelInput | None] | dict[str, Any],
        is_decode: bool,
        max_blocks_decode_batch: int | None,
        any_structured_inputs: bool,
    ) -> tuple[torch.Tensor, list]:
        """Called by TT device ranks (local DP rank 0) to concatenate DP-sized
        inputs and execute. Returns a tuple of:
        - stacked tensor [world, max_num_seqs, 1] of sampled ids
        - list of logprobs per DP rank (as picklable lists, or None)
        Each DP slice is right-padded with zeros to max_num_seqs; empty entries
        are zeros. Same behavior for both prefill and decode."""

        assert self.parallel_config.data_parallel_rank_local == 0, (
            "concat_and_execute_dp must run on local DP rank 0 (device rank)"
        )
        assert self.is_driver_worker, "concat_and_execute_dp must run on driver"
        merged = self.model_runner.concat_dp_model_inputs(
            inputs, is_decode, max_blocks_decode_batch, any_structured_inputs
        )
        sampled_token_ids_per_dp, logprobs_per_dp = (
            self.model_runner.execute_with_model_input(merged)
        )

        # Convert LogprobsTensors to picklable lists for scatter
        logprobs_lists_per_dp = [
            lp.tolists() if lp is not None else None for lp in logprobs_per_dp
        ]

        # Pad each DP result to uniform shape for tensor all_gather.
        world = self.parallel_config.data_parallel_size
        assert len(sampled_token_ids_per_dp) == world
        B = int(self.model_runner.scheduler_config.max_num_seqs)
        for dp_rank in range(world):
            token_ids = sampled_token_ids_per_dp[dp_rank].to(torch.int32)
            if token_ids.numel() == 0:
                token_ids = torch.zeros((B, 1), dtype=torch.int32)
            else:
                assert token_ids.dim() == 2 and token_ids.shape[1] == 1, (
                    "Currently only supporting 1 output token per request"
                )
                pad_rows = B - token_ids.shape[0]
                if pad_rows > 0:
                    token_ids = torch.cat(
                        [
                            token_ids,
                            torch.zeros(
                                (pad_rows, token_ids.shape[1]), dtype=torch.int32
                            ),
                        ],
                        dim=0,
                    )
            sampled_token_ids_per_dp[dp_rank] = token_ids
        return torch.stack(
            sampled_token_ids_per_dp
        ), logprobs_lists_per_dp  # [world, B, 1], [world]

    def apply_dp_execution_result(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs_lists: Optional["LogprobsLists"] = None,
    ) -> ModelRunnerOutput:
        """Called by each DP rank to apply sampled tokens to internal caches.

        Args:
            sampled_token_ids: Sampled token IDs for this DP rank.
            logprobs_lists: Logprobs as lists (already converted from tensors)
                for this DP rank, or None if not requested.
        """
        # Trim to active local batch size to drop padding rows.
        num_reqs = self.model_runner.input_batch.num_reqs
        sampled_token_ids = sampled_token_ids[:num_reqs]
        return self.model_runner.generate_runner_output(
            sampled_token_ids, logprobs_lists
        )

    # ---- Destructor (used to close devices) ----

    def __del__(self):
        # Delete model runner first in case there are model artifacts
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

            if self.mesh_device:
                close_mesh_device(
                    self.mesh_device, self.model_config.override_tt_config
                )
                del self.mesh_device

        if hasattr(super(), "__del__"):
            super().__del__()  # type: ignore


def get_num_available_blocks_tt(vllm_config: VllmConfig) -> int:
    """
    Used to set the number of available blocks for the TT KV cache as we
    currently do not run profiling to determine available memory.
    """

    model_config = vllm_config.model_config
    device_config = vllm_config.device_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config

    data_parallel = vllm_config.parallel_config.data_parallel_size

    is_wormhole = "wormhole_b0" in ttnn.get_arch_name()
    devices_per_dp_cache = device_config.num_devices // data_parallel

    if (
        "Llama-3.1-8B" in model_config.model
        and devices_per_dp_cache == 1
        and is_wormhole
    ):
        # Llama8B on N150
        max_tokens_all_users = 32768
    elif "Qwen3-8B" in model_config.model and devices_per_dp_cache == 1 and is_wormhole:
        # Qwen3-8B on N150 (same constraint as Llama8B-N150)
        max_tokens_all_users = 32768
    elif (
        ("Mistral-7B" in model_config.model or "gemma-3-4b" in model_config.model)
        and devices_per_dp_cache == 1
        and is_wormhole
    ):
        # Mistral7B, and gemma3-4b on N150
        max_tokens_all_users = 65536
    elif (
        (
            "DeepSeek-R1-Distill-Qwen-14B" in model_config.model
            or "Qwen2.5-14B" in model_config.model
            or "gemma-3-4b" in model_config.model
        )
        and devices_per_dp_cache == 2
        and is_wormhole
    ):
        # Qwen2.5-14B and gemma3-4b on N300
        max_tokens_all_users = 65536
    elif (
        "Llama-3.2-90B" in model_config.model
        and devices_per_dp_cache == 8
        and is_wormhole
    ):
        # Llama90B on WH T3K
        max_tokens_all_users = 65536
    elif (
        "Qwen2.5-VL-72B" in model_config.model
        and devices_per_dp_cache == 8
        and is_wormhole
    ):
        # Qwen2.5-VL-72B on WH T3K
        max_tokens_all_users = 65536
    elif "DeepSeek-R1-0528" in model_config.model and is_wormhole:
        max_tokens_all_users = 32768
    else:
        # Note: includes num vision tokens for multi-modal
        max_tokens_all_users = 131072

    # To fit a max batch with (max_tokens_all_users / max batch) per user,
    # allocate an extra block_size per user since vLLM uses a worst-case
    # heuristic and assumes each touched block will require a new
    # allocation. E.g. batch 32, block 64 needs an extra 2048 tokens.
    max_batch = scheduler_config.max_num_seqs
    max_tokens_all_users += cache_config.block_size * max_batch

    num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)

    return num_tt_blocks


# TT-NN utilities


def get_dispatch_core_config(override_tt_config):
    dispatch_core_axis: ttnn.DispatchCoreAxis = None
    if override_tt_config is not None and "dispatch_core_axis" in override_tt_config:
        assert override_tt_config["dispatch_core_axis"] in ["row", "col"], (
            "Invalid dispatch_core_axis:"
            f"{override_tt_config['dispatch_core_axis']}. "
            "Expected: row, col."
        )
        dispatch_core_axis = (
            ttnn.DispatchCoreAxis.COL
            if override_tt_config["dispatch_core_axis"] == "col"
            else ttnn.DispatchCoreAxis.ROW
        )

    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(override_tt_config, num_devices):
    if num_devices == 1:
        # No fabric config for single device
        fabric_config = None
    else:
        # Set the most common value as default
        is_6u = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        fabric_config = (
            ttnn.FabricConfig.FABRIC_1D_RING if is_6u else ttnn.FabricConfig.FABRIC_1D
        )

    # Override fabric_config if specified in override_tt_config
    if override_tt_config is not None and "fabric_config" in override_tt_config:
        fabric_config_str = override_tt_config["fabric_config"]
        fabric_config_map = {
            "DISABLED": ttnn.FabricConfig.DISABLED,
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
            "CUSTOM": ttnn.FabricConfig.CUSTOM,
        }
        fabric_config = fabric_config_map.get(fabric_config_str)
        assert fabric_config is not None, (
            f"Invalid fabric_config: {fabric_config_str}. "
            f"Expected one of {list(fabric_config_map.keys())}."
        )
    return fabric_config


def get_reliability_mode(override_tt_config):
    # Default to strict init and override if specified in override_tt_config.
    reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if (
        override_tt_config is not None
        and "fabric_reliability_mode" in override_tt_config
    ):
        reliability_mode_str = override_tt_config["fabric_reliability_mode"]
        reliability_mode_map = {
            "STRICT_INIT": ttnn.FabricReliabilityMode.STRICT_INIT,
            "RELAXED_INIT": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
        reliability_mode = reliability_mode_map.get(reliability_mode_str)
        assert reliability_mode is not None, (
            f"Invalid fabric_reliability_mode: {reliability_mode_str}. "
            f"Expected one of {list(reliability_mode_map.keys())}."
        )
    return reliability_mode


# From tt-metal/conftest.py:
# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        reliability_mode = get_reliability_mode(override_tt_config)
        logger.info(
            "Setting fabric config: %s, reliability mode: %s",
            fabric_config,
            reliability_mode,
        )
        ttnn.set_fabric_config(fabric_config, reliability_mode)


# From tt-metal/conftest.py:
# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed
# in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate
# the instability to the rest of CI
def reset_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_override_tt_config(override_tt_config, trace_mode):
    device_params = {}

    if trace_mode in ["all", "decode_only"]:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 50000000
        if override_tt_config and "trace_region_size" in override_tt_config:
            device_params["trace_region_size"] = override_tt_config["trace_region_size"]

    if override_tt_config and "worker_l1_size" in override_tt_config:
        device_params["worker_l1_size"] = override_tt_config["worker_l1_size"]

    if override_tt_config and "l1_small_size" in override_tt_config:
        device_params["l1_small_size"] = override_tt_config["l1_small_size"]

    return device_params


def get_mesh_grid(local_dp_rank=0):
    if local_dp_rank == 0:
        # Only DP rank 0 should query devices.
        num_devices_available = ttnn.get_num_devices()
    mesh_grid_dict = {
        "N150": (1, 1),
        "P100": (1, 1),
        "P150": (1, 1),
        "P150x2": (1, 2),
        "N300": (1, 2),
        "P300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "P150x8": (1, 8),
        "TG": (8, 4),
    }
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env is not None:
        try:
            # Try to parse as a literal tuple first
            parsed_value = ast.literal_eval(mesh_device_env)
            if isinstance(parsed_value, tuple) and len(parsed_value) == 2:
                mesh_grid = parsed_value
            else:
                raise ValueError("Not a valid tuple")
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a string key for mesh_grid_dict
            assert mesh_device_env in mesh_grid_dict, (
                f"Invalid MESH_DEVICE: {mesh_device_env}"
            )
            mesh_grid = mesh_grid_dict[mesh_device_env]
    else:
        assert local_dp_rank == 0, (
            "MESH_DEVICE must be set when running with data_parallel_size > 1"
        )
        mesh_grid = (1, num_devices_available)

    assert (
        local_dp_rank != 0
        or ttnn.using_distributed_env()
        or (mesh_grid[0] * mesh_grid[1] <= num_devices_available)
    ), (
        f"Requested mesh grid shape {mesh_grid} is larger than "
        f"number of available devices {num_devices_available}"
    )

    return mesh_grid


def open_mesh_device(override_tt_config, trace_mode, local_dp_rank=0):
    assert local_dp_rank == 0, "open_mesh_device must run on local DP rank 0"
    mesh_grid = get_mesh_grid(local_dp_rank)
    logger.info("Attempting to open mesh device with grid shape %s", mesh_grid)

    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode
    )

    # Set fabric before opening the device
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(override_tt_config, num_devices_requested)

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(override_tt_config),
        **device_params,
    )
    logger.info(
        "multidevice with %d devices and grid %s is created",
        mesh_device.get_num_devices(),
        mesh_grid,
    )
    return mesh_device


def close_mesh_device(mesh_device, override_tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(override_tt_config, num_devices)
