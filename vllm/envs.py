# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Private helpers (module-level)
# ----------------------------------------------------------------------------


def _xdg_cache_home() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def _xdg_config_home() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def _default_dbo_comm_sms() -> int:
    try:
        import torch

        if getattr(torch.version, "hip", None) is not None:
            return 64
    except ImportError:
        pass
    return 20


def _tpu_pathways_default() -> bool:
    return "proxy" in os.getenv("JAX_PLATFORMS", "").lower()


_MCP_LABEL_CHOICES = {"container", "code_interpreter", "web_search_preview"}


def maybe_convert_bool(value: str | None) -> bool | None:
    """Back-compat shim: used by tests/ci_envs.py."""
    if value is None:
        return None
    return bool(int(value))


# ----------------------------------------------------------------------------
# Shared config for sub-models
# ----------------------------------------------------------------------------

_SUB_CONFIG = SettingsConfigDict(
    env_prefix="VLLM_",
    extra="ignore",
    case_sensitive=False,
    populate_by_name=True,
)


# ----------------------------------------------------------------------------
# Sub-models
# ----------------------------------------------------------------------------


class BuildSettings(BaseSettings):
    model_config = _SUB_CONFIG

    target_device: str = "cuda"
    main_cuda_version: str = "13.0"
    max_jobs: str | None = Field(default=None, alias="MAX_JOBS")
    nvcc_threads: str | None = Field(default=None, alias="NVCC_THREADS")
    use_precompiled: bool = Field(
        default=False,
        description=(
            "Use precompiled binaries (.so) instead of building from source. "
            "Implicitly enabled when VLLM_PRECOMPILED_WHEEL_LOCATION is set."
        ),
    )
    skip_precompiled_version_suffix: bool = False
    docker_build_context: bool = False
    cmake_build_type: Literal["Debug", "Release", "RelWithDebInfo"] | None = Field(
        default=None, alias="CMAKE_BUILD_TYPE"
    )
    verbose: bool = Field(default=False, alias="VERBOSE")

    @field_validator("target_device", mode="before")
    @classmethod
    def _lower_target_device(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("main_cuda_version", mode="before")
    @classmethod
    def _normalize_main_cuda_version(cls, v: Any) -> Any:
        if v is None:
            return "13.0"
        if isinstance(v, str):
            lowered = v.lower()
            return lowered or "13.0"
        return v


class PathSettings(BaseSettings):
    model_config = _SUB_CONFIG

    config_root: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_config_home(), "vllm")
        )
    )
    cache_root: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm")
        )
    )
    assets_cache: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm", "assets")
        )
    )
    xla_cache_path: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm", "xla_cache")
        )
    )
    rpc_base_path: str = Field(default_factory=tempfile.gettempdir)
    tuned_config_folder: str | None = None
    model_redirect_path: str | None = None
    lora_resolver_cache_dir: str | None = None
    lora_resolver_hf_repo_list: str | None = None
    cudart_so_path: str | None = None
    nccl_so_path: str | None = None
    nccl_include_path: str | None = None
    ld_library_path: str | None = Field(default=None, alias="LD_LIBRARY_PATH")
    cuda_home: str | None = Field(default=None, alias="CUDA_HOME")
    cuda_compatibility_path: str | None = None
    enable_cuda_compatibility: bool = False
    logging_config_path: str | None = None
    debug_dump_path: str | None = None
    pattern_match_debug: str | None = None
    gc_debug: str = ""
    system_start_date: str | None = None

    @field_validator(
        "config_root", "cache_root", "assets_cache", "xla_cache_path", mode="after"
    )
    @classmethod
    def _expanduser(cls, v: str) -> str:
        return os.path.expanduser(v)


class ServerSettings(BaseSettings):
    model_config = _SUB_CONFIG

    host_ip: str = ""
    port: int | None = None
    api_key: str | None = None
    debug_log_api_server_response: bool = False
    rpc_timeout: int = 10000
    http_timeout_keep_alive: int = 5
    max_n_sequences: int = 16384
    engine_iteration_timeout_s: int = 60
    engine_ready_timeout_s: int = 600
    execute_model_timeout_seconds: int = 300
    keep_alive_on_engine_death: bool = False
    server_dev_mode: bool = False
    allow_long_max_model_len: bool = False
    enable_responses_api_store: bool = False
    allow_chunked_local_attn_with_hybrid_kv_cache: bool = True
    process_name_prefix: str = "VLLM"
    loopback_ip: str = ""
    skip_model_name_validation: bool = False
    allow_insecure_serialization: bool = False
    disable_log_logo: bool = False
    tool_parse_regex_timeout_seconds: int = 1
    tool_json_error_automatic_retry: bool = False
    enforce_strict_tool_calling: bool = False
    custom_scopes_for_profiling: bool = False
    nvtx_scopes_for_profiling: bool = False
    mq_max_chunk_bytes_mb: int = 16

    @field_validator("port", mode="before")
    @classmethod
    def _parse_port(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError) as err:
            from urllib3.util import parse_url

            sval = str(v)
            try:
                parsed = parse_url(sval)
            except Exception:
                parsed = None
            if parsed is not None and parsed.scheme:
                raise ValueError(
                    f"VLLM_PORT '{sval}' appears to be a URI. "
                    "This may be caused by a Kubernetes service discovery issue,"
                    "check the warning in: https://docs.vllm.ai/en/stable/configuration/env_vars/?h=vllm_port#environment-variables"
                ) from None
            raise ValueError(f"VLLM_PORT '{sval}' must be a valid integer") from err


class LoggingSettings(BaseSettings):
    model_config = _SUB_CONFIG

    configure_logging: bool = True
    logging_level: str = "INFO"
    logging_prefix: str = ""
    logging_stream: str = "ext://sys.stdout"
    logging_color: str = "auto"
    no_color: bool = Field(default=False, alias="NO_COLOR")
    log_stats_interval: float = 10.0
    log_batchsize_interval: float = -1.0
    trace_function: int = 0
    ringbuffer_warning_interval: int = 60
    debug_workspace: bool = False
    debug_mfu_metrics: bool = False
    log_model_inspection: bool = False

    @field_validator("logging_level", mode="after")
    @classmethod
    def _upper_logging_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("log_stats_interval", mode="after")
    @classmethod
    def _clamp_log_stats_interval(cls, v: float) -> float:
        if v <= 0.0:
            return 10.0
        return v

    @field_validator("no_color", mode="before")
    @classmethod
    def _parse_no_color(cls, v: Any) -> Any:
        # Old logic: os.getenv("NO_COLOR", "0") != "0" — any non-"0" is True.
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v != "0"
        return bool(v)


class DistributedSettings(BaseSettings):
    model_config = _SUB_CONFIG

    dp_rank: int = 0
    dp_rank_local: int = Field(
        default=-1,
        description=(
            "Local data-parallel rank. Defaults to VLLM_DP_RANK when this "
            "variable is not explicitly set."
        ),
    )
    dp_size: int = 1
    dp_master_ip: str = "127.0.0.1"
    dp_master_port: int = 0
    randomize_dp_dummy_inputs: bool = False
    ray_dp_pack_strategy: Literal["strict", "fill", "span"] = "strict"
    ray_extra_env_var_prefixes_to_copy: str = ""
    ray_extra_env_vars_to_copy: str = ""
    ray_per_worker_gpus: float = 1.0
    ray_bundle_indices: str = ""
    use_ray_compiled_dag_channel_type: Literal["auto", "nccl", "shm"] = "auto"
    use_ray_compiled_dag_overlap_comm: bool = False
    use_ray_wrapped_pp_comm: bool = True
    use_ray_v2_executor_backend: bool = True
    worker_multiproc_method: Literal["spawn", "fork"] = "fork"
    enable_v1_multiprocessing: bool = True
    local_rank: int = Field(default=0, alias="LOCAL_RANK")
    cuda_visible_devices: str | None = Field(default=None, alias="CUDA_VISIBLE_DEVICES")
    disable_pynccl: bool = False
    skip_p2p_check: bool = True
    allreduce_use_symm_mem: bool = True
    allreduce_use_flashinfer: bool = False
    use_nccl_symm_mem: bool = False
    msgpack_zero_copy_threshold: int = 256


class CompilationSettings(BaseSettings):
    model_config = _SUB_CONFIG

    use_aot_compile: bool = False
    force_aot_load: bool = False
    use_mega_aot_artifact: bool = False
    use_bytecode_hook: bool = True
    use_standalone_compile: bool = True
    enable_pregrad_passes: bool = True
    enable_inductor_max_autotune: bool = True
    enable_inductor_coordinate_descent_tuning: bool = True
    disable_compile_cache: bool = False
    compile_cache_save_format: Literal["binary", "unpacked"] = "binary"
    use_layername: bool = True
    use_v2_model_runner: bool = False


class MediaSettings(BaseSettings):
    model_config = _SUB_CONFIG

    image_fetch_timeout: int = 5
    video_fetch_timeout: int = 30
    audio_fetch_timeout: int = 10
    media_cache: str = ""
    media_cache_max_size_mb: int = 5120
    media_cache_ttl_hours: float = 24
    media_fetch_max_retries: int = 3
    media_url_allow_redirects: bool = True
    media_loading_thread_count: int = 8
    max_audio_clip_filesize_mb: int = 25
    video_loader_backend: str = "opencv"
    media_connector: str = "http"
    mm_hasher_algorithm: Literal["blake3", "sha256", "sha512"] = "blake3"
    object_storage_shm_buffer_name: str | None = Field(
        default=None,
        description=(
            "Name of the POSIX shared-memory segment used for multimodal "
            "object storage. When unset, vLLM auto-generates a UUID-suffixed "
            "name and writes it back to the environment."
        ),
    )
    assets_cache_model_clean: bool = False

    @field_validator("mm_hasher_algorithm", mode="before")
    @classmethod
    def _lower_mm_hasher(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v


class CpuSettings(BaseSettings):
    model_config = _SUB_CONFIG

    cpu_kvcache_space: int | None = None
    cpu_omp_threads_bind: str = "auto"
    cpu_num_of_reserved_cpu: int | None = None
    cpu_sgl_kernel: bool = False
    cpu_attn_split_kv: bool = True
    cpu_int4_w4a8: bool = True
    zentorch_weight_prepack: bool = True


class RocmSettings(BaseSettings):
    model_config = _SUB_CONFIG

    rocm_sleep_mem_chunk_size: int = 256
    rocm_use_aiter: bool = False
    rocm_use_aiter_paged_attn: bool = False
    rocm_use_aiter_linear: bool = True
    rocm_use_aiter_moe: bool = True
    rocm_use_aiter_rmsnorm: bool = True
    rocm_use_aiter_mla: bool = True
    rocm_use_aiter_mha: bool = True
    rocm_use_aiter_fp4_asm_gemm: bool = False
    rocm_use_aiter_triton_rope: bool = False
    rocm_use_aiter_fp8bmm: bool = True
    rocm_use_aiter_fp4bmm: bool = True
    rocm_use_aiter_unified_attention: bool = False
    rocm_use_aiter_fusion_shared_experts: bool = False
    rocm_use_aiter_triton_gemm: bool = True
    rocm_use_skinny_gemm: bool = True
    rocm_fp8_padding: bool = True
    rocm_moe_padding: bool = True
    rocm_shuffle_kv_cache_layout: bool = False
    rocm_quick_reduce_quantization: Literal["FP", "INT8", "INT6", "INT4", "NONE"] = (
        "NONE"
    )
    rocm_quick_reduce_cast_bf16_to_fp16: bool = True
    rocm_quick_reduce_max_size_bytes_mb: int | None = None
    rocm_fp8_mfma_page_attn: bool = False


class TpuXpuSettings(BaseSettings):
    model_config = _SUB_CONFIG

    xla_check_recompilation: bool = False
    xla_use_spmd: bool = False
    tpu_bucket_padding_gap: int = 0
    tpu_most_model_len: int | None = None
    tpu_using_pathways: bool = Field(
        default_factory=_tpu_pathways_default,
        validation_alias="__VLLM_TPU_USING_PATHWAYS_UNSET_SENTINEL__",
    )
    xpu_enable_xpu_graph: bool = False
    xpu_use_sampler_kernel: bool = True
    sparse_indexer_max_logits_mb: int = 512


class FlashInferSettings(BaseSettings):
    model_config = _SUB_CONFIG

    use_flashinfer_sampler: bool = True
    use_flashinfer_moe_fp16: bool = False
    use_flashinfer_moe_fp8: bool = False
    use_flashinfer_moe_fp4: bool = False
    use_flashinfer_moe_int4: bool = False
    use_flashinfer_moe_mxfp4_mxfp8: bool = False
    use_flashinfer_moe_mxfp4_mxfp8_cutlass: bool = False
    use_flashinfer_moe_mxfp4_bf16: bool = False
    flashinfer_moe_backend: Literal["throughput", "latency", "masked_gemm"] = "latency"
    flashinfer_allreduce_backend: Literal["auto", "trtllm", "mnnvl"] = "auto"
    flashinfer_workspace_buffer_size: int = 394 * 1024 * 1024
    flashinfer_allreduce_fusion_thresholds_mb: Annotated[dict, NoDecode] = Field(
        default_factory=dict
    )
    blockscale_fp8_gemm_flashinfer: bool = True
    has_flashinfer_cubin: bool = False
    max_tokens_per_expert_fp4_moe: int = 163840

    @field_validator("flashinfer_allreduce_fusion_thresholds_mb", mode="before")
    @classmethod
    def _parse_json_thresholds(cls, v: Any) -> Any:
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            return json.loads(v)
        return v


class QuantSettings(BaseSettings):
    model_config = _SUB_CONFIG

    marlin_use_atomic_add: bool = False
    marlin_input_dtype: Literal["int8", "fp8"] | None = None
    humming_online_quant_config: Annotated[dict[str, Any] | None, NoDecode] = None
    humming_input_quant_config: Annotated[dict[str, Any] | None, NoDecode] = None
    humming_use_f16_accum: bool | None = False
    humming_moe_gemm_type: Literal["indexed", "grouped", "auto"] | None = None
    mxfp4_use_marlin: bool | None = None
    deepepll_nvfp4_dispatch: bool = False
    use_deep_gemm: bool = True
    moe_use_deep_gemm: bool = True
    use_deep_gemm_e8m0: bool = True
    use_deep_gemm_tma_aligned_scales: bool = True
    deep_gemm_warmup: Literal["skip", "full", "relax"] = "relax"
    use_fused_moe_grouped_topk: bool = True
    deepep_buffer_size_mb: int = 1024
    deepep_high_throughput_force_intra_node: bool = False
    deepep_low_latency_use_mnnvl: bool = False
    dbo_comm_sms: int = Field(default_factory=_default_dbo_comm_sms)
    multi_stream_gemm_token_threshold: int = 1024
    shared_experts_stream_token_threshold: int = 256
    disable_shared_experts_stream: bool = False
    moe_routing_simulation_strategy: str = ""
    nvfp4_gemm_backend: str | None = None
    use_nvfp4_ct_emulations: bool = False
    q_scale_constant: int = Field(default=200, alias="Q_SCALE_CONSTANT")
    k_scale_constant: int = Field(default=200, alias="K_SCALE_CONSTANT")
    v_scale_constant: int = Field(default=100, alias="V_SCALE_CONSTANT")
    kv_cache_layout: Literal["NHD", "HND"] | None = None
    ssm_conv_state_layout: Literal["SD", "DS"] | None = None
    mla_disable: bool = False
    compute_nans_in_logits: bool = False
    use_fbgemm: bool = False
    use_oink_ops: bool = False
    batch_invariant: bool = False
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"
    use_triton_awq: bool = False

    @field_validator("float32_matmul_precision", mode="before")
    @classmethod
    def _lower_float32(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("moe_routing_simulation_strategy", mode="after")
    @classmethod
    def _lower_moe_routing(cls, v: str) -> str:
        return v.lower()

    @field_validator(
        "humming_online_quant_config", "humming_input_quant_config", mode="before"
    )
    @classmethod
    def _parse_humming_json(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if os.path.exists(v):
                with open(v) as f:
                    return json.load(f)
            return json.loads(v)
        return v

    @field_validator("humming_use_f16_accum", mode="before")
    @classmethod
    def _parse_humming_f16(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return bool(int(v))
        return v

    @field_validator("mxfp4_use_marlin", mode="before")
    @classmethod
    def _parse_mxfp4(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return bool(int(v))
        return v


class ConnectorSettings(BaseSettings):
    model_config = _SUB_CONFIG

    nixl_side_channel_host: str = "localhost"
    nixl_side_channel_port: int = 5600
    nixl_ep_max_num_ranks: int = 32
    mooncake_bootstrap_port: int = 8998
    mooncake_abort_request_timeout: int = 480
    moriio_connector_read_mode: bool = False
    moriio_qp_per_transfer: int = 1
    moriio_post_batch_size: int = -1
    moriio_num_workers: int = 1
    kv_events_use_int_block_hashes: bool = True
    disable_request_id_randomization: bool = False
    elastic_ep_scale_up_launch: bool = False
    elastic_ep_drain_requests: bool = False
    use_simple_kv_offload: bool = False
    weight_offloading_disable_pin_memory: bool = False
    weight_offloading_disable_uva: bool = False
    enable_cudagraph_gc: bool = False
    memory_profiler_estimate_cudagraphs: bool = True
    v1_output_proc_chunk_size: int = 128
    v1_use_outlines_cache: bool = False
    xgrammar_cache_mb: int = 512


class UsageSettings(BaseSettings):
    model_config = _SUB_CONFIG

    usage_stats_server: str = "https://stats.vllm.ai"
    no_usage_stats: bool = False
    do_not_track: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLLM_DO_NOT_TRACK", "DO_NOT_TRACK"),
        description=(
            "Disable usage stats reporting. Also accepts the legacy "
            "DO_NOT_TRACK environment variable."
        ),
    )
    usage_source: str = "production"
    ci_use_s3: bool = False
    test_force_fp8_marlin: bool = False
    test_force_load_format: str = "dummy"
    use_modelscope: bool = False
    s3_access_key_id: str | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    plugins: Annotated[list[str] | None, NoDecode] = None
    disabled_kernels: Annotated[list[str], NoDecode] = Field(default_factory=list)
    allow_runtime_lora_updating: bool = False
    gpt_oss_system_tool_mcp_labels: Annotated[set[str], NoDecode] = Field(
        default_factory=set
    )
    gpt_oss_harmony_system_instructions: bool = False
    use_experimental_parser_context: bool = False
    lora_disable_pdl: bool = False
    lora_enable_dual_stream: bool = False
    enable_fla_packed_recurrent_decode: bool = True
    pp_layer_partition: str | None = None

    @field_validator("plugins", mode="before")
    @classmethod
    def _parse_plugins(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("disabled_kernels", mode="before")
    @classmethod
    def _parse_disabled_kernels(cls, v: Any) -> Any:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("gpt_oss_system_tool_mcp_labels", mode="before")
    @classmethod
    def _parse_gpt_oss_labels(cls, v: Any) -> Any:
        if v is None or v == "":
            return set()
        if isinstance(v, (set, list, tuple)):
            items = {str(x).strip() for x in v if str(x).strip()}
        elif isinstance(v, str):
            items = {p.strip() for p in v.split(",") if p.strip()}
        else:
            return v
        for label in items:
            if label not in _MCP_LABEL_CHOICES:
                raise ValueError(
                    f"Invalid value '{label}' in "
                    f"VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS. "
                    f"Valid options: {sorted(_MCP_LABEL_CHOICES)}."
                )
        return items


# ----------------------------------------------------------------------------
# Root settings
# ----------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    build: BuildSettings = Field(default_factory=BuildSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging_: LoggingSettings = Field(default_factory=LoggingSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    compilation: CompilationSettings = Field(default_factory=CompilationSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    cpu: CpuSettings = Field(default_factory=CpuSettings)
    rocm: RocmSettings = Field(default_factory=RocmSettings)
    tpu_xpu: TpuXpuSettings = Field(default_factory=TpuXpuSettings)
    flashinfer: FlashInferSettings = Field(default_factory=FlashInferSettings)
    quant: QuantSettings = Field(default_factory=QuantSettings)
    connector: ConnectorSettings = Field(default_factory=ConnectorSettings)
    usage: UsageSettings = Field(default_factory=UsageSettings)

    @model_validator(mode="after")
    def _apply_cross_field_logic(self) -> "Settings":
        # VLLM_USE_PRECOMPILED: if VLLM_PRECOMPILED_WHEEL_LOCATION set, force True.
        if not self.build.use_precompiled and os.environ.get(
            "VLLM_PRECOMPILED_WHEEL_LOCATION"
        ):
            self.build.use_precompiled = True

        # VLLM_DP_RANK_LOCAL: if unset, fall back to VLLM_DP_RANK value.
        if "VLLM_DP_RANK_LOCAL" not in os.environ:
            self.distributed.dp_rank_local = self.distributed.dp_rank

        # VLLM_USE_AOT_COMPILE: dynamic default based on torch version and
        # disable_compile_cache.
        if "VLLM_USE_AOT_COMPILE" not in os.environ:
            try:
                from vllm.utils.torch_utils import is_torch_equal_or_newer

                default_aot = (
                    "1"
                    if is_torch_equal_or_newer("2.10.0")
                    and not self.compilation.disable_compile_cache
                    else "0"
                ) == "1"
                self.compilation.use_aot_compile = default_aot
            except ImportError:
                pass

        # VLLM_USE_MEGA_AOT_ARTIFACT: depends on torch version AND use_aot_compile.
        if "VLLM_USE_MEGA_AOT_ARTIFACT" not in os.environ:
            try:
                from vllm.utils.torch_utils import is_torch_equal_or_newer

                default_mega = (
                    "1"
                    if is_torch_equal_or_newer("2.12.0.dev")
                    and self.compilation.use_aot_compile
                    else "0"
                ) == "1"
                self.compilation.use_mega_aot_artifact = default_mega
            except ImportError:
                pass

        # VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: generate UUID default if unset.
        if self.media.object_storage_shm_buffer_name is None:
            env_val = os.environ.get("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME")
            if env_val is not None:
                self.media.object_storage_shm_buffer_name = env_val
            else:
                new_name = f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}"
                os.environ["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = new_name
                self.media.object_storage_shm_buffer_name = new_name

        return self


# ----------------------------------------------------------------------------
# Registry: env var name -> (sub_attr, field_name)
# ----------------------------------------------------------------------------


def resolve_env_name(info: FieldInfo, field_name: str, prefix: str) -> str:
    """Resolve the canonical environment variable name for a pydantic field.

    Used by the registry builder and the docs generator
    (docs/mkdocs/plugins/gen_env_vars.py).

    - explicit ``alias=`` wins (used for non-VLLM_ env vars)
    - ``AliasChoices`` -> first choice (e.g., DO_NOT_TRACK fallback)
    - sentinel for VLLM_TPU_USING_PATHWAYS -> the canonical name
    - otherwise: ``prefix + field_name.upper()``
    """
    if info.alias is not None:
        return info.alias
    va = info.validation_alias
    if va is None:
        return prefix + field_name.upper()
    if isinstance(va, AliasChoices):
        first = va.choices[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(va, str):
        if va == "__VLLM_TPU_USING_PATHWAYS_UNSET_SENTINEL__":
            return "VLLM_TPU_USING_PATHWAYS"
        return va
    return prefix + field_name.upper()


def _build_registry() -> dict[str, tuple[str, str]]:
    registry: dict[str, tuple[str, str]] = {}
    for sub_attr, sub_field in Settings.model_fields.items():
        annotation = sub_field.annotation
        if not (isinstance(annotation, type) and issubclass(annotation, BaseSettings)):
            continue
        sub_cls: type[BaseSettings] = annotation
        prefix = sub_cls.model_config.get("env_prefix", "") or ""
        for field_name, info in sub_cls.model_fields.items():
            env_name = resolve_env_name(info, field_name, prefix)
            if env_name in registry:
                raise RuntimeError(
                    f"Env var {env_name!r} registered by multiple fields: "
                    f"{registry[env_name]} and {(sub_attr, field_name)}"
                )
            registry[env_name] = (sub_attr, field_name)
    return registry


_VAR_TO_PATH: dict[str, tuple[str, str]] = _build_registry()


# ----------------------------------------------------------------------------
# Lazy settings accessor
# ----------------------------------------------------------------------------


_settings: Settings | None = None


def _get_settings() -> Settings:
    """Return the cached Settings singleton.

    Only used after ``enable_envs_cache()`` has run; otherwise each attribute
    read constructs a fresh ``Settings`` so subsequent mutations to
    ``os.environ`` are visible (matching the pre-refactor lambda behavior).
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_attr(name: str) -> Any:
    sub_attr, field_name = _VAR_TO_PATH[name]
    # When the cache wrapper is active we read from the cached singleton.
    # Otherwise construct a fresh Settings each access so subsequent
    # mutations to os.environ are visible (matching the pre-refactor
    # lambda behavior of re-evaluating env vars on every read).
    settings = _get_settings() if _is_envs_cache_enabled() else Settings()
    sub = getattr(settings, sub_attr)
    return getattr(sub, field_name)


# ----------------------------------------------------------------------------
# Back-compat shim: environment_variables dict
# ----------------------------------------------------------------------------


def _make_env_getter(name: str) -> Callable[[], Any]:
    return lambda: _get_attr(name)


environment_variables: dict[str, Callable[[], Any]] = {
    name: _make_env_getter(name) for name in _VAR_TO_PATH
}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def __getattr__(name: str):
    """Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in _VAR_TO_PATH:
        return _get_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_VAR_TO_PATH.keys())


def is_set(name: str) -> bool:
    """Check if an environment variable is explicitly set."""
    if name in _VAR_TO_PATH:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in _VAR_TO_PATH:
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:
                logger.warning("Unknown vLLM environment variable detected: %s", env)


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache"""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Enable caching of environment variables.

    NOTE: Currently invoked after service initialization to reduce runtime
    overhead. This also means environment variables should NOT be updated
    after the service is initialized.
    """
    if _is_envs_cache_enabled():
        return
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    for key in _VAR_TO_PATH:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Reset the environment variables cache.

    Useful for isolating environments between unit tests.
    """
    global __getattr__, _settings
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__
    _settings = None


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggests k8s service discovery issue.
    """
    if "VLLM_PORT" not in os.environ:
        return None
    port = os.getenv("VLLM_PORT", "0")
    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/stable/configuration/env_vars/?h=vllm_port#environment-variables"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known vLLM env var; drop entries in ``ignored_factors``;
    hash everything else. This keeps the cache key aligned across workers."""

    ignored_factors: set[str] = {
        "MAX_JOBS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_USE_MODELSCOPE",
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        "VLLM_DEBUG_DUMP_PATH",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
        "LD_LIBRARY_PATH",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
        "VLLM_CI_USE_S3",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_HOST_IP",
        "VLLM_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOGGING_PREFIX",
        "VLLM_LOGGING_STREAM",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_COLOR",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
        "VLLM_TUNED_CONFIG_FOLDER",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        "VLLM_IMAGE_FETCH_TIMEOUT",
        "VLLM_VIDEO_FETCH_TIMEOUT",
        "VLLM_AUDIO_FETCH_TIMEOUT",
        "VLLM_MEDIA_CACHE",
        "VLLM_MEDIA_CACHE_MAX_SIZE_MB",
        "VLLM_MEDIA_CACHE_TTL_HOURS",
        "VLLM_MEDIA_FETCH_MAX_RETRIES",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
        "VLLM_VIDEO_LOADER_BACKEND",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "VLLM_ASSETS_CACHE",
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_MOE_PREPACK",
        "VLLM_ZENTORCH_WEIGHT_PREPACK",
        "VLLM_TEST_FORCE_LOAD_FORMAT",
        "VLLM_ENABLE_CUDA_COMPATIBILITY",
        "VLLM_CUDA_COMPATIBILITY_PATH",
        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
    }

    from vllm.config.utils import normalize_value

    factors: dict[str, object] = {}
    for factor in _VAR_TO_PATH:
        if factor in ignored_factors:
            continue
        try:
            raw = _get_attr(factor)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue
        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
