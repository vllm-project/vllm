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
import traceback
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

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
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.platforms import current_platform
from vllm.profiler.op_capture.meta_ops import (
    has_kernel_for,
    register_meta_impls,
    skip_meta_triton_launches,
)
from vllm.profiler.op_capture.recorder import OpRecorder, RecordedOp
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_distributed_init_method, get_open_port
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import (
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs
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


def _kv_cache_budget(config: VllmConfig, specs: dict[str, KVCacheSpec]) -> int:
    """Bytes to size the KV cache with: one request at the model's maximum length.

    The cache shapes reach the operators, so a meta and a real capture have to
    agree on them rather than on whatever memory each device happens to have.
    Sized with the engine's own admission check, over the same cache groups it
    would build, plus the null block the block pool holds back. Shrink the
    budget by lowering `max_model_len`.

    Args:
        config: Config the specs were built from.
        specs: KV cache spec of every attention layer.

    Returns:
        The budget in bytes.

    """
    # Copied: grouping may unify the specs of a hybrid model in place.
    groups = get_kv_cache_groups(config, dict(specs))
    return _max_memory_usage_bytes_from_groups(config, groups) + _pool_bytes_per_block(
        groups
    )


@dataclass(frozen=True)
class BatchSpec:
    """A uniform batch to run the forward path on.

    Args:
        num_reqs: Number of requests in the batch.
        num_tokens: Total query tokens, split evenly across requests.
        num_computed_tokens: Context length already in the KV cache per request,
            0 for a pure prefill.

    """

    num_reqs: int = 1
    num_tokens: int = 8
    num_computed_tokens: int = 0

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
    num_query_heads: int
    num_kv_heads: int
    head_size: int


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
        vllm_root = Path(vllm.__file__).parent
        package = Path(__file__).parent
        location = ""
        for frame in traceback.extract_tb(exception.__traceback__):
            path = Path(frame.filename)
            if path.is_relative_to(vllm_root) and not path.is_relative_to(package):
                relative = path.relative_to(vllm_root.parent)
                location = f"{relative}:{frame.lineno} in {frame.name}"
        return cls(
            error=f"{type(exception).__name__}: {exception}",
            module=module or "",
            location=location,
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

    @property
    def custom_ops(self) -> list[RecordedOp]:
        return [op for op in self.ops if op.is_custom]

    @property
    def placeholder_ops(self) -> list[RecordedOp]:
        """Ops given placeholder outputs; shapes after the first are guesses."""
        return [op for op in self.ops if op.placeholder]


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

    Raises:
        ValueError: If the config asks for more than one rank; the harness is
            single-process. Or if the batch exceeds the scheduler's per-step
            token or request budget, which no real step could.

    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "meta",
        batch: BatchSpec | None = None,
        engine_args: EngineArgs | None = None,
    ):
        self.batch = batch or BatchSpec()
        self.is_meta = torch.device(device).type == "meta"
        self.device = torch.device(device)
        if not self.is_meta and self.device.index is None:
            self.device = torch.device(f"{self.device.type}:0")

        engine_args = replace(engine_args) if engine_args else EngineArgs(model=model)
        engine_args.model = model
        engine_args.enforce_eager = True
        if self.is_meta:
            engine_args.load_format = "meta"
        self.vllm_config: VllmConfig = engine_args.create_engine_config()

        parallel_config = self.vllm_config.parallel_config
        if parallel_config.world_size_across_dp != 1:
            raise ValueError(
                "ForwardHarness runs in a single process, so it needs "
                "tensor, pipeline and data parallel sizes of 1; got "
                f"{parallel_config.world_size_across_dp} ranks."
            )
        scheduler_config = self.vllm_config.scheduler_config
        if (
            self.batch.num_tokens > scheduler_config.max_num_batched_tokens
            or self.batch.num_reqs > scheduler_config.max_num_seqs
        ):
            raise ValueError(
                f"{self.batch} exceeds the scheduler budget of "
                f"{scheduler_config.max_num_batched_tokens} tokens and "
                f"{scheduler_config.max_num_seqs} requests per step; raise "
                f"`max_num_batched_tokens` or `max_num_seqs` in `engine_args`."
            )

        self.model: torch.nn.Module | None = None
        self.attn_groups: list[list[AttentionGroup]] = []
        self.kernel_block_sizes: list[int] = []
        self.kv_cache_specs: dict[str, KVCacheSpec] = {}
        self._attn_metadata: dict[str, Any] = {}
        self._slot_mappings: dict[str, torch.Tensor] = {}
        self._recorder: OpRecorder | None = None
        self.failure: CaptureFailure | None = None
        self._exit_stack = ExitStack()

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
        """Build the single-rank process groups this model's layers expect.

        The model-parallel groups depend on the model -- only a MoE one gets an
        expert-parallel group -- so they are rebuilt here and torn down on exit,
        leaving a capture earlier in the same process no way to interfere. A
        world group the harness had to create is torn down with them.
        """
        destroy_model_parallel()
        if not torch.distributed.is_initialized():
            # Gloo keeps a meta-device capture off the accelerator entirely; at
            # world size 1 no collective ever reaches the backend anyway.
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=get_distributed_init_method(
                    "127.0.0.1", get_open_port()
                ),
                local_rank=0,
                backend="gloo",
            )
            self._exit_stack.callback(destroy_distributed_environment)
        initialize_model_parallel(1, 1)
        self._exit_stack.callback(destroy_model_parallel)

    def _init_kv_cache(self) -> None:
        """Group layers by backend, size the cache, and bind it to the layers."""
        config = self.vllm_config
        layers: dict[str, AttentionLayerBase] = get_layers_from_vllm_config(
            config, cast(type[Any], AttentionLayerBase)
        )
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
        kv_cache_config = get_kv_cache_configs(
            config,
            [self.kv_cache_specs],
            [_kv_cache_budget(config, self.kv_cache_specs)],
        )[0]

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
        )

    def _build_attn_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for group_id, groups in enumerate(self.attn_groups):
            common = self._common_attn_metadata(self.kernel_block_sizes[group_id])
            for group in groups:
                built = group.get_metadata_builder().build(0, common)
                metadata.update((name, built) for name in group.layer_names)
                self._slot_mappings.update(
                    (name, common.slot_mapping) for name in group.layer_names
                )
        return metadata

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

    def run_forward(self) -> torch.Tensor:
        """Run one forward pass and the logits computation.

        Returns:
            The logits tensor, one row per request.

        """
        assert self.model is not None, "Use ForwardHarness as a context manager"
        batch = self.batch
        input_ids = torch.zeros(batch.num_tokens, dtype=torch.long, device=self.device)
        positions = self._positions()
        last_token_indices = torch.arange(
            batch.query_len - 1,
            batch.num_tokens,
            batch.query_len,
            device=self.device,
        )
        with (
            torch.inference_mode(),
            set_forward_context(
                self._attn_metadata,
                self.vllm_config,
                num_tokens=batch.num_tokens,
                slot_mapping=self._slot_mappings,
            ),
        ):
            hidden_states = self.model(input_ids=input_ids, positions=positions)
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

    def selection_metadata(self) -> SelectionMetadata:
        """Collect the hardware-dependent choices this capture depends on."""
        config = self.vllm_config
        layers: dict[str, AttentionLayerBase] = get_layers_from_vllm_config(
            config, cast(type[Any], AttentionLayerBase)
        )
        first = next(iter(layers.values()), None)
        return SelectionMetadata(
            platform=current_platform.device_name,
            device=str(self.device),
            dtype=str(config.model_config.dtype),
            quantization=config.model_config.quantization,
            attention_backends={
                group.layer_names[0]: ".".join(group.backend.full_cls_name())
                for groups in self.attn_groups
                for group in groups
            },
            kv_cache_layout=config.cache_config.kv_cache_layout or "none",
            kv_cache_dtype=config.cache_config.cache_dtype,
            block_size=config.cache_config.block_size or 0,
            kernel_block_sizes=tuple(self.kernel_block_sizes),
            num_attention_layers=len(self.kv_cache_specs),
            num_query_heads=getattr(first, "num_heads", 0),
            num_kv_heads=getattr(first, "num_kv_heads", 0),
            head_size=getattr(first, "head_size", 0),
        )

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
            first: ops with unknown output shapes get placeholder outputs, and a
            forward pass that still fails returns what it recorded, with
            `failure` set. A model that fails to build still raises.

    Returns:
        The ordered operator list together with the selection metadata it is
        only valid under.

    """
    from vllm.profiler.op_capture.trace import capture_execution_trace

    with ForwardHarness(
        model, device=device, batch=batch, engine_args=engine_args
    ) as harness:
        with (
            nullcontext()
            if trace_path is None
            else capture_execution_trace(Path(trace_path))
        ):
            recorder = harness.record(keep_going=keep_going)
        materialized: list[str] = []
        if harness.is_meta and keep_going:
            materialized = harness.materialized_tensors()
        elif harness.is_meta:
            harness.assert_on_meta()
        dispatch_key = current_platform.dispatch_key
        missing = {
            op.name
            for op in recorder.ops
            if not op.name.startswith("triton::")
            and not has_kernel_for(op.name, dispatch_key)
        }
        return OpCapture(
            model=model,
            batch=harness.batch,
            selection=harness.selection_metadata(),
            ops=recorder.ops,
            module_types=recorder.module_types,
            trace_path=None if trace_path is None else str(trace_path),
            missing_kernels=tuple(sorted(missing)),
            failure=harness.failure,
            materialized=tuple(materialized),
        )
