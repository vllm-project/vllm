# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drives a vLLM model's forward path outside the engine and records its ops.

The harness walks the same code a worker does -- model construction, KV cache
configuration, attention-metadata construction, forward, logits -- but without
booting an engine, so every hardware-selection decision (platform, attention
backend, KV cache layout, dtype, head and block sizes) is the genuine one for
the machine it runs on. With `device="meta"` nothing is materialized: the model
comes from the `meta` load format, so an HF config is all that is needed.
"""

import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import vllm.ir
from vllm.config import (
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import BatchedTensorInputs
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.profiler.op_capture.meta_ops import (
    has_kernel_for,
    register_meta_impls,
    skip_meta_triton_launches,
)
from vllm.profiler.op_capture.recorder import OpRecorder, RecordedOp, vllm_location
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_distributed_init_method, get_open_port
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    mamba_get_block_table_tensor,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    min_kv_cache_memory_bytes,
)
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


@contextmanager
def _model_runner_torch_cuda() -> Iterator[None]:
    """The `torch.cuda` aliases the XPU model runner leaves installed.

    Model code calls `torch.cuda.current_stream()` and the like, which on XPU
    only work once the runner has pointed them at `torch.xpu`. A harness
    shares its process, so the original attributes are restored on exit.
    """
    if not current_platform.is_xpu():
        yield
        return
    from vllm.v1.worker.xpu_model_runner import torch_cuda_wrapper

    saved = dict(vars(torch.cuda))
    try:
        with torch_cuda_wrapper():
            yield
    finally:
        for name in set(vars(torch.cuda)) - set(saved):
            delattr(torch.cuda, name)
        for name, value in saved.items():
            if vars(torch.cuda).get(name) is not value:
                setattr(torch.cuda, name, value)


@dataclass(frozen=True)
class BatchSpec:
    """A uniform batch to run the forward path on.

    Args:
        num_reqs: Number of requests in the batch.
        num_tokens: Total query tokens, split evenly across requests.
        num_computed_tokens: Context length already in the KV cache per request,
            0 for a pure prefill.
        num_mm_items: Multimodal items to encode this step, of the modality
            with the most tokens per item at its maximum size, as the model
            runner profiles; their embeddings replace the batch's first tokens.

    """

    num_reqs: int = 1
    num_tokens: int = 8
    num_computed_tokens: int = 0
    num_mm_items: int = 0

    @property
    def query_len(self) -> int:
        if self.num_tokens % self.num_reqs:
            raise ValueError(
                f"num_tokens={self.num_tokens} must divide evenly across "
                f"num_reqs={self.num_reqs}"
            )
        return self.num_tokens // self.num_reqs

    @property
    def seq_len(self) -> int:
        return self.num_computed_tokens + self.query_len

    @property
    def is_prefill(self) -> bool:
        """Whether the requests are still in their prompt, not decoding."""
        return self.num_computed_tokens == 0 or self.query_len > 1


@dataclass(frozen=True)
class SelectionMetadata:
    """The hardware-dependent choices the captured operators depend on.

    The same model yields different kernels per platform, so an operator list is
    only interpretable alongside these, and only valid for the platform it was
    captured on.
    """

    platform: str
    device: str
    dtype: str
    quantization: str | None
    attention_backends: dict[str, str]
    kv_cache_layout: str
    kv_cache_dtype: str
    block_size: int
    kernel_block_sizes: tuple[int, ...]
    num_attention_layers: int
    """Layers deriving from `AttentionLayerBase`, KV-cache-only ones included.

    Sparse attention's indexer and compressor own a KV cache without running
    attention, so on those models this exceeds the decoder layer count: 243
    against 61 for DeepSeek-V4-Pro. `layer_kinds` gives the composition.
    """
    num_query_heads: int | None
    num_kv_heads: int | None
    head_size: int | None
    """Shape of the first layer reporting one, `None` if no layer does.

    Hybrid models mix layer types -- Mamba mixers and sparse attention's indexer
    report no head counts, and a model may implement attention in a class that
    reports none either -- so these describe one representative layer, not the
    whole model. `attention_backends` is the per-layer breakdown.
    """
    tensor_parallel_size: int = 1
    """Ranks the model is sharded over; head counts are per rank."""
    attention_layers: tuple[str, ...] = ()
    """Names of the attention layers, in model order."""
    num_model_layers: int = 0
    """Decoder layers on this rank, the denominator for per-layer op counts."""
    layer_kinds: tuple[tuple[str, int], ...] = ()
    """Class name and count of each `attention_layers` kind, most common first."""
    sliding_windows: dict[str, int] = field(default_factory=dict)
    """Window of every sliding-window attention layer, keyed by layer name."""
    moe_experts: tuple[str, ...] = ()
    """Class names of the fused-experts implementations the MoE layers use."""


@dataclass(frozen=True)
class CaptureFailure:
    """Why a `keep_going` forward pass stopped before finishing."""

    error: str
    """Exception type and message."""
    module: str
    """Dotted path of the innermost module the exception left, `""` for none."""
    location: str
    """Innermost vLLM source line on the traceback, outside this package."""

    @classmethod
    def from_exception(
        cls, exception: BaseException, module: str | None
    ) -> "CaptureFailure":
        return cls(
            error=f"{type(exception).__name__}: {exception}",
            module=module or "",
            location=vllm_location(exception),
        )


@dataclass
class OpCapture:
    """An ordered operator capture plus everything needed to interpret it."""

    model: str
    batch: BatchSpec
    selection: SelectionMetadata
    ops: list[RecordedOp] = field(default_factory=list)
    module_types: dict[str, str] = field(default_factory=dict)
    """Class name of every module reached, keyed by dotted path."""
    trace_path: str | None = None
    """Where the Chakra execution trace was written, when one was requested."""
    missing_kernels: tuple[str, ...] = ()
    """Captured ops with no kernel for the platform's dispatch key, which would
    fail on the real device."""
    failure: CaptureFailure | None = None
    """Why the forward pass stopped early; `ops` then ends where it did."""
    materialized: tuple[str, ...] = ()
    """Parameters and buffers model code put on a real device despite `meta`,
    reported by a `keep_going` capture instead of failing it."""
    rank: int = 0
    """Tensor-parallel rank the capture ran as."""

    @property
    def custom_ops(self) -> list[RecordedOp]:
        return [op for op in self.ops if op.is_custom]

    @property
    def placeholder_ops(self) -> list[RecordedOp]:
        """Ops given placeholder outputs; shapes after the first are guesses."""
        return [op for op in self.ops if op.placeholder]

    @property
    def body_error_ops(self) -> list[RecordedOp]:
        """Custom ops whose kernel failed on meta, their outputs faked."""
        return [op for op in self.ops if op.body_error]


class ForwardHarness:
    """Builds a model and runs its forward path with no engine, worker or runner.

    Args:
        model: Model id or local path.
        device: Device to run on. `"meta"` loads no weights and allocates no
            accelerator memory; a real device (e.g. `"xpu"`) runs the same path
            on hardware, for comparison.
        batch: Batch shape to run.
        engine_args: Base engine args, for overriding e.g. `max_model_len` or
            `quantization`. `model`, `load_format` and `enforce_eager` are set
            by the harness.
        rank: Tensor-parallel rank to run as, when `engine_args` asks for more
            than one.
        distributed_init_method: Where the ranks rendezvous, e.g.
            `"tcp://127.0.0.1:29500"`. Required for more than one rank, each
            running its own harness in its own process, as `capture_ranks`
            arranges.

    Raises:
        ValueError: If the config asks for pipeline, context or data
            parallelism, or for tensor parallelism without
            `distributed_init_method`; or if the batch is one no real step
            could run: over the scheduler budget or `max_model_len`, or with
            multimodal items the model cannot take.
        NotImplementedError: If the model is encoder-decoder.

    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "meta",
        batch: BatchSpec | None = None,
        engine_args: EngineArgs | None = None,
        rank: int = 0,
        distributed_init_method: str | None = None,
    ):
        self.model_id = model
        self.batch = batch or BatchSpec()
        self.device = torch.device(device)
        self.is_meta = self.device.type == "meta"
        if not self.is_meta and self.device.index is None:
            self.device = torch.device(f"{self.device.type}:{rank}")
        self.rank = rank
        self._distributed_init_method = distributed_init_method

        engine_args = replace(engine_args) if engine_args else EngineArgs(model=model)
        engine_args.model = model
        engine_args.enforce_eager = True
        if self.is_meta:
            engine_args.load_format = "meta"
        self.vllm_config: VllmConfig = engine_args.create_engine_config()

        if self.vllm_config.model_config.is_encoder_decoder:
            raise NotImplementedError(
                "ForwardHarness runs no encoder, so it cannot capture "
                f"encoder-decoder models such as {model}."
            )
        parallel_config = self.vllm_config.parallel_config
        self.tensor_parallel_size = parallel_config.tensor_parallel_size
        if parallel_config.world_size_across_dp != self.tensor_parallel_size:
            raise ValueError(
                "ForwardHarness captures tensor-parallel ranks only, so it "
                "needs pipeline, context and data parallel sizes of 1."
            )
        if self.tensor_parallel_size > 1 and distributed_init_method is None:
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} needs one "
                "harness per rank; use capture_ranks() to run them."
            )
        self._mm_budget: MultiModalBudget | None = None
        self._check_batch(self.batch)

        self.model: torch.nn.Module | None = None
        self.attn_groups: list[list[AttentionGroup]] = []
        self.encoder_only_groups: list[AttentionGroup] = []
        self.kernel_block_sizes: list[int] = []
        self.kv_cache_specs: dict[str, KVCacheSpec] = {}
        self._attn_metadata: dict[str, Any] = {}
        self._slot_mappings: dict[str, torch.Tensor] = {}
        self._recorder: OpRecorder | None = None
        self.failure: CaptureFailure | None = None
        self._exit_stack = ExitStack()

    def _check_batch(self, batch: BatchSpec) -> None:
        scheduler_config = self.vllm_config.scheduler_config
        if (
            batch.num_tokens > scheduler_config.max_num_batched_tokens
            or batch.num_reqs > scheduler_config.max_num_seqs
        ):
            raise ValueError(
                f"{batch} exceeds the scheduler budget of "
                f"{scheduler_config.max_num_batched_tokens} tokens and "
                f"{scheduler_config.max_num_seqs} requests per step; raise "
                f"`max_num_batched_tokens` or `max_num_seqs` in `engine_args`."
            )
        max_model_len = self.vllm_config.model_config.max_model_len
        if batch.seq_len > max_model_len:
            raise ValueError(
                f"{batch} reaches {batch.seq_len} tokens per request, beyond "
                f"max_model_len={max_model_len}; raise it in `engine_args`."
            )
        if batch.num_mm_items:
            budget = self._multimodal_budget()
            max_items = budget.mm_max_items_per_batch.get(
                budget.get_modality_with_max_tokens(), 0
            )
            if batch.num_mm_items > max_items:
                raise ValueError(
                    f"{batch} has more multimodal items than the "
                    f"{max_items} the encoder budget allows per step."
                )

    def _multimodal_budget(self) -> MultiModalBudget:
        if self._mm_budget is not None:
            return self._mm_budget
        model_config = self.vllm_config.model_config
        if not model_config.supports_multimodal_inputs:
            raise ValueError(f"{self.model_id} takes no multimodal inputs.")
        if model_config.is_multimodal_raw_input_only_model:
            raise ValueError(
                f"{self.model_id} takes its multimodal inputs raw in "
                "forward, which the harness does not pass."
            )
        budget = MultiModalBudget(self.vllm_config, MULTIMODAL_REGISTRY)
        if not budget.mm_max_toks_per_item:
            raise ValueError(f"{self.model_id} has no multimodal encoder to run.")
        self._mm_budget = budget
        return budget

    def set_batch(self, batch: BatchSpec) -> None:
        """Run later forward passes on `batch`, reusing the built model and cache.

        Raises:
            ValueError: As `ForwardHarness` does for its batch.

        """
        self._check_batch(batch)
        self.batch = batch
        self.failure = None
        if self.model is not None:
            self._slot_mappings = {}
            self._attn_metadata = self._build_attn_metadata()

    def __enter__(self) -> "ForwardHarness":
        if self.is_meta:
            register_meta_impls()
            self._exit_stack.enter_context(
                skip_meta_triton_launches(self._record_triton_launch)
            )
        else:
            current_platform.set_device(self.device)
        # Held open for the harness's lifetime, as a worker does: model
        # construction and weight loading both read the current config.
        self._exit_stack.enter_context(set_current_vllm_config(self.vllm_config))
        # A worker settles these before it builds anything, and both decide
        # which operator a layer ends up calling. Unlike a worker, a harness
        # shares its process, so both are restored on exit.
        self._exit_stack.enter_context(
            self.vllm_config.kernel_config.ir_op_priority.set_priority()
        )
        self._exit_stack.enter_context(
            vllm.ir.enable_torch_wrap(
                self.vllm_config.compilation_config.ir_enable_torch_wrap
            )
        )
        self._exit_stack.enter_context(_model_runner_torch_cuda())
        self._init_distributed()
        # As a worker does before building the model: fused MoE kernels take
        # their scratch buffers from the workspace.
        init_workspace_manager(
            self.device, 2 if self.vllm_config.parallel_config.enable_dbo else 1
        )
        self._exit_stack.callback(reset_workspace_manager)
        self.model = get_model(vllm_config=self.vllm_config)
        # As an executor does once the model is loaded: the attention backend
        # picks the block size, which the KV cache shapes are built from.
        current_platform.update_block_size_for_backend(self.vllm_config)
        self._init_kv_cache()
        self._attn_metadata = self._build_attn_metadata()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.model = None
        self._attn_metadata = {}
        self._slot_mappings = {}
        self._exit_stack.close()

    def _init_distributed(self) -> None:
        """Build the process groups this model's layers expect.

        The model-parallel groups depend on the model -- only a MoE one gets an
        expert-parallel group -- so they are rebuilt here and torn down on exit,
        leaving a capture earlier in the same process no way to interfere. A
        world group the harness had to create is torn down with them.
        """
        destroy_model_parallel()
        world_size = self.tensor_parallel_size
        if not torch.distributed.is_initialized():
            # Gloo keeps a meta-device capture off the accelerator entirely:
            # its collectives have meta kernels, which exchange nothing.
            backend = "gloo"
            if world_size > 1 and not self.is_meta:
                backend = current_platform.dist_backend
            init_distributed_environment(
                world_size=world_size,
                rank=self.rank,
                distributed_init_method=self._distributed_init_method
                or get_distributed_init_method("127.0.0.1", get_open_port()),
                local_rank=self.rank,
                backend=backend,
            )
            self._exit_stack.callback(destroy_distributed_environment)
        initialize_model_parallel(world_size, 1)
        self._exit_stack.callback(destroy_model_parallel)

    def _init_encoder_only_attn(self) -> None:
        """Give encoder-only layers, which keep no KV cache, metadata builders.

        As the model runner does: one `EncoderOnlyAttentionSpec` group per
        backend and head configuration, bound to an empty cache tensor.
        """
        config = self.vllm_config
        cache_config = config.cache_config
        dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, config.model_config
        )
        groups: dict[tuple, AttentionGroup] = {}
        for name, layer in get_layers_from_vllm_config(config, Attention).items():
            if layer.attn_type != AttentionType.ENCODER_ONLY:
                continue
            layer.kv_cache = torch.empty(0, dtype=dtype, device=self.device)
            backend = layer.get_attn_backend()
            key = (
                backend.full_cls_name(),
                layer.num_heads,
                layer.num_kv_heads,
                layer.head_size,
            )
            if key not in groups:
                spec = EncoderOnlyAttentionSpec(
                    block_size=cache_config.block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=dtype,
                )
                groups[key] = AttentionGroup(backend, [], spec, len(groups))
            groups[key].layer_names.append(name)
        self.encoder_only_groups = list(groups.values())
        for group in self.encoder_only_groups:
            group.create_metadata_builders(config, self.device)

    def _attention_layers(self) -> dict[str, AttentionLayerBase]:
        return get_layers_from_vllm_config(
            self.vllm_config, cast(type[Any], AttentionLayerBase)
        )

    def _init_kv_cache(self) -> None:
        """Group layers by backend, size the cache, and bind it to the layers."""
        config = self.vllm_config
        self._init_encoder_only_attn()
        layers = self._attention_layers()
        self.kv_cache_specs = {
            name: spec
            for name, layer in layers.items()
            if (spec := layer.get_kv_cache_spec(config)) is not None
        }
        if not self.kv_cache_specs:
            return

        backends = get_current_attn_backends(config)
        layout_names = [
            layout.name for layout in get_supported_kv_cache_layouts(backends)
        ]
        resolve_kv_cache_layout(
            config, [layout_names], list(self.kv_cache_specs.values())
        )
        # Sized for one request of `max_model_len`, not by device memory: the
        # cache shapes reach the ops, so meta and real captures must agree.
        kv_cache_config = get_kv_cache_configs(
            config,
            [self.kv_cache_specs],
            [min_kv_cache_memory_bytes(config, self.kv_cache_specs)],
        )[0]
        initialize_mamba_ssu_backend(
            config.mamba_config,
            kv_cache_config,
            use_replayssm=config.cache_config.use_replayssm,
        )

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            # Split as the model runner does: a group's layers share a
            # metadata builder only if backend, own spec and query heads agree.
            by_key: dict[tuple, tuple[Any, KVCacheSpec, list[str]]] = {}
            for name in group.layer_names:
                backend = layers[name].get_attn_backend()
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                key = (
                    backend.full_cls_name(),
                    spec,
                    getattr(layers[name], "num_heads", 0),
                )
                by_key.setdefault(key, (backend, spec, []))[2].append(name)
            self.attn_groups.append(
                [
                    AttentionGroup(backend, names, spec, group_id)
                    for backend, spec, names in by_key.values()
                ]
            )

        self.kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        for group_id, groups in enumerate(self.attn_groups):
            for attn_group in groups:
                attn_group.create_metadata_builders(
                    config, self.device, self.kernel_block_sizes[group_id]
                )

        kv_caches = allocate_kv_cache(
            kv_cache_config,
            self.device,
            config.cache_config.get_resolved_kv_cache_layout(),
            self.kernel_block_sizes,
        )
        bind_kv_cache(
            kv_caches,
            config.compilation_config.static_forward_context,
            [],
            kv_cache_groups=kv_cache_config.kv_cache_groups,
        )

    def _common_attn_metadata(self, block_size: int) -> CommonAttentionMetadata:
        batch = self.batch
        query_lens = torch.full((batch.num_reqs,), batch.query_len, dtype=torch.int32)
        query_start_loc_cpu = torch.zeros(batch.num_reqs + 1, dtype=torch.int32)
        torch.cumsum(query_lens, dim=0, out=query_start_loc_cpu[1:])
        seq_lens_cpu = torch.full((batch.num_reqs,), batch.seq_len, dtype=torch.int32)
        blocks_per_req = cdiv(batch.seq_len, block_size)
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc_cpu.to(self.device),
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_cpu.to(self.device),
            num_reqs=batch.num_reqs,
            num_actual_tokens=batch.num_tokens,
            max_query_len=batch.query_len,
            max_seq_len=batch.seq_len,
            block_table_tensor=torch.arange(
                batch.num_reqs * blocks_per_req, dtype=torch.int32, device=self.device
            ).view(batch.num_reqs, blocks_per_req),
            slot_mapping=torch.arange(
                batch.num_tokens, dtype=torch.int64, device=self.device
            ),
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            positions=self._positions(),
            is_prefilling=torch.full((batch.num_reqs,), batch.is_prefill),
        )

    def _build_attn_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for group_id, groups in enumerate(self.attn_groups):
            common = self._common_attn_metadata(self.kernel_block_sizes[group_id])
            for group in groups:
                builder = group.get_metadata_builder()
                self._set_aligned_state_indices(builder, group.kv_cache_spec, common)
                built = builder.build(0, common)
                metadata.update((name, built) for name in group.layer_names)
                self._slot_mappings.update(
                    (name, common.slot_mapping) for name in group.layer_names
                )
        if self.encoder_only_groups:
            # Encoder-only attention reads neither, but the builders expect both.
            common = replace(
                self._common_attn_metadata(1),
                block_table_tensor=torch.zeros(
                    (self.batch.num_reqs, 1), dtype=torch.int32, device=self.device
                ),
            )
            for group in self.encoder_only_groups:
                built = group.get_metadata_builder().build(0, common)
                metadata.update((name, built) for name in group.layer_names)
        return metadata

    def _set_aligned_state_indices(
        self,
        builder: Any,
        kv_cache_spec: KVCacheSpec,
        common: CommonAttentionMetadata,
    ) -> None:
        """Hand a builder its aligned Mamba state indices, as Model Runner V2 does.

        Raises:
            NotImplementedError: If the builder takes them on a platform whose
                runner cannot compute them.

        """
        config = self.vllm_config
        if not (
            config.use_v2_model_runner
            and config.cache_config.mamba_cache_mode == "align"
            and isinstance(kv_cache_spec, MambaSpec)
            and hasattr(builder, "mamba_aligned_state_indices")
        ):
            return
        if not current_platform.is_cuda_alike():
            raise NotImplementedError(
                f"{type(builder).__name__} takes aligned Mamba state indices, "
                "which Model Runner V2 computes with a CUDA-only kernel "
                "(MambaSpecDecodeGPUContext.compute_aligned_state_indices), so "
                f"the {current_platform.device_name} runner cannot build its "
                "metadata. Under VLLM_USE_V2_MODEL_RUNNER=0 the builder computes "
                "them itself."
            )
        # The rows the runner's kernel gathers, per request.
        builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
            common.block_table_tensor, common.seq_lens, kv_cache_spec, "align"
        )

    def _record_triton_launch(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._recorder is not None:
            self._recorder.record_launch(f"triton::{name}", args, kwargs)

    def _positions(self) -> torch.Tensor:
        batch = self.batch
        return (
            torch.arange(batch.query_len, dtype=torch.long, device=self.device)
            .repeat(batch.num_reqs)
            .add_(batch.num_computed_tokens)
        )

    def _dummy_mm_kwargs(self) -> BatchedTensorInputs:
        budget = self._multimodal_budget()
        items = budget.get_dummy_encoder_profile_inputs(
            budget.get_modality_with_max_tokens(), self.batch.num_mm_items
        )
        _, _, mm_kwargs = next(group_and_batch_mm_kwargs(items, device=self.device))
        return mm_kwargs

    def _embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the batch as the model runner does for a multimodal model.

        The encoder outputs of `num_mm_items` items fill the batch's first
        tokens, cut off where the batch ends as a chunked prefill would.
        """
        model = cast(SupportsMultiModal, self.model)
        mm_embeds: list[torch.Tensor] = []
        room = self.batch.num_tokens
        if self.batch.num_mm_items:
            for output in model.embed_multimodal(**self._dummy_mm_kwargs()):
                if room > 0:
                    mm_embeds.append(output[:room])
                    room -= mm_embeds[-1].shape[0]
        # On the host, as the runner keeps it: a boolean mask indexes a meta
        # tensor only if its values can be read.
        is_multimodal = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        is_multimodal[: input_ids.shape[0] - room] = True
        return model.embed_input_ids(
            input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_multimodal
        )

    def _pooling_metadata(self, model: VllmModelForPooling) -> PoolingMetadata:
        """Pool every request's whole prompt, as the model runner does.

        The task is the pooler config's, else the first the pooler supports.
        """
        batch = self.batch
        pooler_config = self.vllm_config.model_config.pooler_config
        task = (pooler_config and pooler_config.task) or min(
            model.pooler.get_supported_tasks()
        )
        params = PoolingParams(task=task)
        params.verify(self.vllm_config.model_config)
        model.pooler.get_pooling_updates(task).apply(params)
        prompt_lens = torch.full((batch.num_reqs,), batch.seq_len, dtype=torch.int32)
        token_ids_cpu = torch.zeros((batch.num_reqs, batch.seq_len), dtype=torch.int32)
        metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=token_ids_cpu.to(self.device),
            prompt_token_ids_cpu=token_ids_cpu,
            pooling_params=[params] * batch.num_reqs,
            pooling_states=[PoolingStates() for _ in range(batch.num_reqs)],
        )
        metadata.build_pooling_cursor(
            np.full(batch.num_reqs, batch.query_len),
            seq_lens_cpu=prompt_lens,
            device=self.device,
        )
        return metadata

    def run_forward(self) -> Any:
        """Run one forward pass and its head: logits, or the pooler's output.

        Returns:
            The logits tensor, one row per request, or the pooler's output for
            a pooling model.

        """
        assert self.model is not None, "Use ForwardHarness as a context manager"
        batch = self.batch
        model_config = self.vllm_config.model_config
        input_ids = torch.zeros(batch.num_tokens, dtype=torch.long, device=self.device)
        positions = self._positions()
        if model_config.uses_mrope:
            # A text token's three M-RoPE positions are its 1-D one.
            positions = positions.repeat(3, 1)
        last_token_indices = torch.arange(
            batch.query_len - 1,
            batch.num_tokens,
            batch.query_len,
            device=self.device,
        )
        with torch.inference_mode():
            inputs_embeds = None
            if model_config.supports_multimodal_inputs:
                inputs_embeds = self._embed_inputs(input_ids)
            keep_ids = inputs_embeds is None or model_config.requires_raw_input_tokens
            with set_forward_context(
                self._attn_metadata,
                self.vllm_config,
                num_tokens=batch.num_tokens,
                slot_mapping=self._slot_mappings,
            ):
                hidden_states = self.model(
                    input_ids=input_ids if keep_ids else None,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
                if model_config.runner_type == "pooling":
                    model = cast(VllmModelForPooling, self.model)
                    return model.pooler(
                        hidden_states=hidden_states,
                        pooling_metadata=self._pooling_metadata(model),
                    )
                return self.model.compute_logits(hidden_states[last_token_indices])

    def record(self, keep_going: bool = False) -> OpRecorder:
        """Run the forward path, recording every dispatched operator in order.

        Args:
            keep_going: Give ops with unknown output shapes placeholder outputs,
                and if the forward pass still fails, keep the ops recorded so
                far and set `failure` instead of raising.

        Returns:
            The recorder, holding the ordered operators and the modules reached.

        """
        assert self.model is not None, "Use ForwardHarness as a context manager"
        key = "Meta" if self.is_meta else current_platform.dispatch_key
        with OpRecorder(
            self.model, dispatch_key=key, keep_going=keep_going
        ) as recorder:
            self._recorder = recorder
            try:
                self.run_forward()
            except Exception as exception:
                if not keep_going:
                    raise
                self.failure = CaptureFailure.from_exception(
                    exception, recorder.module_raising(exception)
                )
            finally:
                self._recorder = None
        return recorder

    def _all_attn_groups(self) -> list[AttentionGroup]:
        return [*chain.from_iterable(self.attn_groups), *self.encoder_only_groups]

    @staticmethod
    def _first_shaped_layer(
        layers: Iterable[AttentionLayerBase],
    ) -> AttentionLayerBase | None:
        """The first layer reporting a full head shape, `None` if none does.

        A hybrid model's layers are not all self-attention: Mamba mixers and the
        indexer of sparse attention are `AttentionLayerBase` too, and report
        either no heads or their own -- a Mamba mixer's `num_heads` counts state
        heads. Reading the counts off whichever layer comes first reports those
        as if they were attention's.
        """
        shape_attrs = ("num_heads", "num_kv_heads", "head_size")
        return next(
            (
                layer
                for layer in layers
                if all(getattr(layer, attr, None) is not None for attr in shape_attrs)
            ),
            None,
        )

    def selection_metadata(self) -> SelectionMetadata:
        """Collect the hardware-dependent choices this capture depends on."""
        config = self.vllm_config
        layers = self._attention_layers()
        shaped = self._first_shaped_layer(layers.values())
        return SelectionMetadata(
            platform=current_platform.device_name,
            device=str(self.device),
            dtype=str(config.model_config.dtype),
            quantization=config.model_config.quantization,
            attention_backends={
                group.layer_names[0]: ".".join(group.backend.full_cls_name())
                for group in self._all_attn_groups()
            },
            kv_cache_layout=config.cache_config.kv_cache_layout or "none",
            kv_cache_dtype=config.cache_config.cache_dtype,
            block_size=config.cache_config.block_size or 0,
            kernel_block_sizes=tuple(self.kernel_block_sizes),
            num_attention_layers=len(layers),
            num_query_heads=getattr(shaped, "num_heads", None),
            num_kv_heads=getattr(shaped, "num_kv_heads", None),
            head_size=getattr(shaped, "head_size", None),
            tensor_parallel_size=self.tensor_parallel_size,
            attention_layers=tuple(layers),
            num_model_layers=config.model_config.get_num_layers(config.parallel_config),
            layer_kinds=tuple(
                Counter(type(layer).__name__ for layer in layers.values()).most_common()
            ),
            sliding_windows={
                name: window
                for name, layer in layers.items()
                if isinstance(window := getattr(layer, "sliding_window", None), int)
            },
            moe_experts=self._moe_experts(),
        )

    def _moe_experts(self) -> tuple[str, ...]:
        assert self.model is not None, "Use ForwardHarness as a context manager"
        names = set()
        for module in self.model.modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, FusedMoEMethodBase):
                kernel = method.moe_kernel
                experts = kernel.fused_experts if kernel is not None else method
                names.add(type(experts).__name__)
        return tuple(sorted(names))

    def materialized_tensors(self) -> list[str]:
        """Parameters and buffers that model code placed on a real device."""
        assert self.model is not None, "Use ForwardHarness as a context manager"
        return [
            name
            for name, tensor in (
                *self.model.named_parameters(),
                *self.model.named_buffers(),
            )
            if tensor.device.type != "meta"
        ]

    def assert_on_meta(self) -> None:
        """Check that nothing about the model escaped the meta device.

        Raises:
            AssertionError: If any parameter or buffer was materialized.

        """
        materialized = self.materialized_tensors()
        assert not materialized, f"Tensors left the meta device: {materialized}"


def _capture(
    harness: ForwardHarness, keep_going: bool, trace_path: Path | None
) -> OpCapture:
    from vllm.profiler.op_capture.trace import capture_execution_trace

    with nullcontext() if trace_path is None else capture_execution_trace(trace_path):
        recorder = harness.record(keep_going=keep_going)
    materialized: list[str] = []
    if harness.is_meta and keep_going:
        materialized = harness.materialized_tensors()
    elif harness.is_meta:
        harness.assert_on_meta()
    dispatch_key = current_platform.dispatch_key
    # Asked once per distinct operator: a forward pass records tens of thousands
    # of calls but only hundreds of names.
    dispatched = {op.name for op in recorder.ops if not op.name.startswith("triton::")}
    missing = {name for name in dispatched if not has_kernel_for(name, dispatch_key)}
    return OpCapture(
        model=harness.model_id,
        batch=harness.batch,
        selection=harness.selection_metadata(),
        ops=recorder.ops,
        module_types=recorder.module_types,
        trace_path=None if trace_path is None else str(trace_path),
        missing_kernels=tuple(sorted(missing)),
        failure=harness.failure,
        materialized=tuple(materialized),
        rank=harness.rank,
    )


def capture_batches(
    model: str,
    batches: Sequence[BatchSpec],
    *,
    device: str = "meta",
    engine_args: EngineArgs | None = None,
    keep_going: bool = False,
    rank: int = 0,
    distributed_init_method: str | None = None,
) -> list[OpCapture]:
    """Capture the operators a model executes on each of several batches.

    Layers pick kernels by batch -- prefill or decode, short or long context --
    so one batch rarely reaches every code path. The model is built once.

    Args:
        model: Model id or local path. Only its HF config is needed on `meta`.
        batches: Batch shapes to run, in order.
        device: As for `capture_model_ops`.
        engine_args: As for `capture_model_ops`. `max_model_len` must cover
            the longest batch.
        keep_going: As for `capture_model_ops`.
        rank: As for `ForwardHarness`.
        distributed_init_method: As for `ForwardHarness`.

    Returns:
        One capture per batch, in the same order.

    """
    if not batches:
        raise ValueError("capture_batches needs at least one batch")
    with ForwardHarness(
        model,
        device=device,
        batch=batches[0],
        engine_args=engine_args,
        rank=rank,
        distributed_init_method=distributed_init_method,
    ) as harness:
        captures = []
        for batch in batches:
            harness.set_batch(batch)
            captures.append(_capture(harness, keep_going, trace_path=None))
        return captures


def capture_model_ops(
    model: str,
    *,
    device: str = "meta",
    batch: BatchSpec | None = None,
    trace_path: str | os.PathLike | None = None,
    engine_args: EngineArgs | None = None,
    keep_going: bool = False,
) -> OpCapture:
    """Capture the operators a model executes under vLLM.

    Args:
        model: Model id or local path. Only its HF config is needed on `meta`.
        device: `"meta"` (default) to run without weights or accelerator
            memory, or a real device to run the same path on hardware.
        batch: Batch shape to run; a single 8-token prefill by default.
        trace_path: When set, also write a Chakra execution trace of the
            recorded forward pass there.
        engine_args: Base engine args, for overrides such as `max_model_len`.
        keep_going: Collect every gap in one run rather than stopping at the
            first: ops with unknown output shapes get placeholder outputs, a
            custom op whose kernel fails on `meta` is finished by its fake
            kernel, and a forward pass that still fails returns what it
            recorded, with `failure` set. A model that fails to build still
            raises.

    Returns:
        The ordered operator list together with the selection metadata it is
        only valid under.

    """
    with ForwardHarness(
        model, device=device, batch=batch, engine_args=engine_args
    ) as harness:
        return _capture(
            harness, keep_going, None if trace_path is None else Path(trace_path)
        )
