# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone host for the existing MRV2 EAGLE3 and own-KV MTP speculators."""

import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.errors import CompletionUnknownError, runtime_unusable
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.adapter import (
    DraftStepExecutor,
    Features,
    TargetConditionedDraftAdapter,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.kv_layout import (
    make_standalone_kv_config as _make_standalone_kv_config,
)
from vllm.v1.spec_decode.remote.runtime.kv_layout import (
    select_standalone_kv_spec,
)
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.step import (
    DraftStepPlan,
    DraftStepPlanner,
    DraftStepRow,
)
from vllm.v1.spec_decode.remote.runtime.weights import (
    materialize_target_shell,
    model_weight_bytes,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

logger = init_logger(__name__)


def _resolve_local_rank(
    device: torch.device, assigned_physical_gpu_ids: list[int] | None
) -> int:
    """Bind distributed groups to the same visible GPU as the draft model."""
    from vllm.platforms import current_platform
    from vllm.platforms.interface import (
        get_assigned_physical_gpu_ids,
        set_assigned_physical_gpu_ids,
    )

    if device.type != "cuda" or device.index is None:
        raise ValueError("standalone drafter requires an indexed CUDA device")
    if device.index >= torch.accelerator.device_count():
        raise ValueError(
            f"standalone drafter device {device} is not a visible CUDA GPU"
        )
    if assigned_physical_gpu_ids is not None:
        set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
    assigned = get_assigned_physical_gpu_ids()
    if assigned is None:
        return device.index
    physical_device_id = current_platform.visible_device_id_to_physical_device_id(
        device.index
    )
    try:
        return assigned.index(physical_device_id)
    except ValueError as e:
        raise ValueError(f"standalone drafter device {device} is not assigned") from e


def make_input_batch(
    plan: DraftStepPlan,
    input_ids: torch.Tensor,
    buffers: "InputBuffers",
) -> "InputBatch":
    """Map the planner's ragged batch onto the MRV2 runner input contract."""
    from vllm.v1.worker.gpu.input_batch import InputBatch

    n, t = plan.num_reqs, plan.num_tokens
    device = buffers.device
    buffers.input_ids[:t].copy_(input_ids)
    buffers.positions[:t].copy_(torch.from_numpy(plan.positions))
    buffers.query_start_loc[: n + 1].copy_(torch.from_numpy(plan.query_start_loc))
    buffers.query_start_loc[n + 1 :].fill_(t)
    buffers.seq_lens[:n].copy_(torch.from_numpy(plan.seq_lens))
    buffers.seq_lens[n:].zero_()
    buffers.is_padding[:t].zero_()
    indices = torch.from_numpy(plan.req_indices).to(device=device, dtype=torch.int64)
    # The planner already selected the final prompt or decode rows. These
    # flags affect metadata optimizations only; each row still uses its true
    # query length and committed position below.
    prefilling = np.array(
        [
            row.num_sampled == 0 or row.num_tokens > row.num_sampled + row.num_rejected
            for row in plan.rows
        ],
        dtype=np.bool_,
    )
    cu_logits_np = np.arange(n + 1, dtype=np.int32)
    return InputBatch(
        req_ids=[str(row.key.sequence_id) for row in plan.rows],
        num_reqs=n,
        num_reqs_after_padding=n,
        idx_mapping=indices,
        idx_mapping_np=plan.req_indices,
        expanded_idx_mapping=indices,
        expanded_local_pos=torch.zeros(n, dtype=torch.int32, device=device),
        num_scheduled_tokens=plan.num_scheduled_tokens,
        num_tokens=t,
        num_tokens_after_padding=t,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=buffers.query_start_loc[: n + 1],
        query_start_loc_np=plan.query_start_loc,
        seq_lens=buffers.seq_lens[:n],
        seq_lens_cpu_upper_bound=torch.from_numpy(plan.seq_lens),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=plan.num_computed_tokens,
        prefill_len_np=np.where(prefilling, plan.seq_lens, 0),
        num_computed_prefill_tokens_np=plan.num_computed_tokens,
        is_prefilling_np=prefilling,
        has_prefill=bool(prefilling.any()),
        decode_graph_eligible=False,
        input_ids=buffers.input_ids[:t],
        positions=buffers.positions[:t],
        is_padding=buffers.is_padding[:t],
        logits_indices=buffers.query_start_loc[1 : n + 1].to(torch.int64) - 1,
        cu_num_logits=torch.from_numpy(cu_logits_np).to(device),
        cu_num_logits_np=cu_logits_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


class GPUDraftStepExecutor(DraftStepExecutor):
    """Own only draft weights/cache; delegate every forward to the local runner."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        device: torch.device,
        kv_cache_bytes: int,
        shared_weights_source: str = "auto",
    ) -> None:
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import (
            set_random_seed,
            set_torch_threads_for_runtime,
        )
        from vllm.v1.attention.backends.utils import (
            get_supported_kv_cache_layouts,
            resolve_kv_cache_layout,
        )
        from vllm.v1.worker.block_table import get_block_table_width
        from vllm.v1.worker.gpu.attn_utils import (
            get_kv_cache_spec,
            init_attn_backend,
            init_kv_cache,
        )
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.gpu.input_batch import InputBuffers
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
        from vllm.v1.worker.workspace import init_workspace_manager

        self.contract = create_runtime_contract(vllm_config)
        self.vllm_config = vllm_config
        self.device = device
        parallel = vllm_config.parallel_config
        if parallel.world_size != 1 or parallel.data_parallel_size != 1:
            raise ValueError("standalone drafter requires world_size=1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("standalone drafter currently requires enforce_eager")
        local_rank = _resolve_local_rank(device, parallel.assigned_physical_gpu_ids)
        if kv_cache_bytes <= 0:
            raise ValueError("draft KV cache budget must be positive")
        torch.accelerator.set_device_index(device)
        current_platform.check_if_supports_dtype(vllm_config.model_config.dtype)
        current_platform.import_kernels()
        torch.set_float32_matmul_precision(envs.VLLM_FLOAT32_MATMUL_PRECISION)
        set_random_seed(vllm_config.model_config.seed)
        set_torch_threads_for_runtime()
        init_workspace_manager(device, num_ubatches=1)
        # WorkerBase normally establishes these process-wide IR settings.
        # The standalone host deliberately does not construct a target worker.
        import vllm.ir

        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )
        self._dist_directory = tempfile.TemporaryDirectory(prefix="vllm-remote-draft-")
        init_method = Path(self._dist_directory.name, "rendezvous").as_uri()
        with set_current_vllm_config(vllm_config), torch.inference_mode():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=local_rank,
                distributed_init_method=init_method,
                backend="nccl",
            )
            initialize_model_parallel()
            target_shell = materialize_target_shell(
                vllm_config.model_config,
                device,
                shared_weights_source=shared_weights_source,
            )
            spec = vllm_config.speculative_config
            assert spec is not None
            speculator_cls = MTPSpeculator if spec.method == "mtp" else EagleSpeculator
            self.speculator = speculator_cls(vllm_config, device)
            self.speculator.load_model(target_shell)
            del target_shell
            self.speculator.model.eval()

            specs = get_kv_cache_spec(vllm_config)
            first = select_standalone_kv_spec(specs)
            self.block_size = first.block_size
            layers = get_layers_from_vllm_config(
                vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            )
            layouts = get_supported_kv_cache_layouts(
                layer.get_attn_backend() for layer in layers.values()
            )
            layout = resolve_kv_cache_layout(
                vllm_config, [[layout.name for layout in layouts]], specs.values()
            )
            self.kv_config = _make_standalone_kv_config(
                first, list(specs), kv_cache_bytes, layout
            )
            self.num_blocks = self.kv_config.num_blocks
            scheduler = vllm_config.scheduler_config
            self.input_buffers = InputBuffers(
                scheduler.max_num_seqs, scheduler.max_num_batched_tokens, device
            )
            self.block_tables = BlockTables(
                block_sizes=[self.block_size],
                max_num_reqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
                max_num_blocks_per_group=[
                    get_block_table_width(
                        math.ceil(
                            vllm_config.model_config.max_model_len / self.block_size
                        ),
                        self.block_size,
                        token_alignment=first.block_table_token_alignment,
                    )
                ],
                device=device,
            )
            self.attn_groups, _, kernel_sizes = init_attn_backend(
                self.kv_config, vllm_config, device
            )
            self.model_state = DefaultModelState(
                vllm_config, self.speculator.model, None, device
            )
            self.speculator.set_attn(
                self.model_state,
                self.kv_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
            self.speculator.init_cudagraph_manager(CUDAGraphMode.NONE)
            self.kv_caches = init_kv_cache(
                vllm_config.compilation_config.static_forward_context,
                self.kv_config,
                device,
                kernel_sizes,
                vllm_config,
                block_tables=self.block_tables,
                attn_groups=(
                    g
                    for groups in self.attn_groups + self.speculator.attn_groups
                    for g in groups
                ),
            )
            self._init_sampling_buffers(scheduler.max_num_seqs)
        draft_bytes, embedding_bytes, head_bytes = model_weight_bytes(
            self.speculator.model
        )
        logger.info(
            "Standalone memory bytes: draft_specific=%d embedding=%d lm_head=%d "
            "draft_kv=%d cuda_graph=0 ipc_rings=0 allocated=%d; dtype=%s "
            "draft_quantization=%s shared_weights_source=%s blocks=%d block_size=%d",
            draft_bytes,
            embedding_bytes,
            head_bytes,
            sum(t.size for t in self.kv_config.kv_cache_tensors),
            torch.accelerator.memory_allocated(device),
            vllm_config.model_config.dtype,
            spec.draft_model_config.quantization,
            shared_weights_source,
            self.num_blocks,
            self.block_size,
        )

    def _init_sampling_buffers(self, max_num_reqs: int) -> None:
        self.last_sampled = torch.zeros(
            max_num_reqs, dtype=torch.int64, device=self.device
        )
        # Match MRV2 RequestState: its Triton prefill kernel merges this value
        # with a sampled token explicitly cast to int32, regardless of wire dtype.
        self.lookahead = torch.zeros(
            1, max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.temperature = torch.zeros(
            max_num_reqs, dtype=torch.float32, device=self.device
        )
        self.seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=self.device)

    @property
    def max_num_tokens(self) -> int:
        return self.input_buffers.max_num_tokens

    @torch.inference_mode()
    def warmup(self) -> None:
        """Run real prefill/decode kernels before starting live RPC deadlines."""
        config = self.vllm_config
        spec = config.speculative_config
        assert spec is not None
        # A separate pool keeps synthetic request/KV ownership out of live state.
        # Each new live request overwrites the warmed request slot's block table.
        planner = DraftStepPlanner(
            DraftBlockPool(self.num_blocks, self.block_size),
            max_num_reqs=1,
            max_model_len=config.model_config.max_model_len,
            num_speculative_tokens=spec.num_speculative_tokens,
        )
        key = SequenceKey("warmup", 0, 0)
        planner.open(key)
        failure: BaseException | None = None
        try:
            for prefill in (True, False):
                num_tokens = min(
                    self.block_size + 1 if prefill else spec.num_speculative_tokens + 1,
                    self.max_num_tokens,
                    config.model_config.max_model_len,
                )
                while True:
                    row = DraftStepRow(
                        key,
                        num_tokens,
                        num_sampled=0 if prefill else 1,
                        num_rejected=0 if prefill else num_tokens - 1,
                        next_prefill_tokens=(0,) if prefill else (),
                    )
                    plan, failures = planner.plan([row])
                    if not failures:
                        break
                    # Let live admission account for rejection and model-length
                    # clamping instead of duplicating its KV extent calculation.
                    if num_tokens > 1:
                        num_tokens = max(1, num_tokens // 2)
                    elif not prefill and planner.num_computed(key):
                        planner.close(key)
                        planner.open(key)
                    else:
                        raise ValueError(
                            "draft KV budget cannot hold a warmup step: "
                            f"{failures[0][1]}"
                        )
                features: Features = []
                for slot in self.contract.schema.slots:
                    if slot.kind == "positions":
                        tensor = torch.from_numpy(plan.positions)
                    else:
                        count = plan.num_reqs if slot.per_sequence else num_tokens
                        tensor = torch.zeros(
                            (count, *slot.trailing_shape),
                            dtype=getattr(torch, slot.dtype),
                        )
                    features.append(tensor)
                self.execute(plan, features)
                planner.commit(plan)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            planner.close(key)
            if failure is None or not runtime_unusable(failure):
                try:
                    self.last_sampled.zero_()
                    self.lookahead.zero_()
                    torch.accelerator.synchronize(self.device)
                except Exception as cleanup_error:
                    if failure is None or runtime_unusable(cleanup_error):
                        raise
                    failure.add_note(f"warmup cleanup also failed: {cleanup_error}")

    def prepare_block_tables(self, plan: DraftStepPlan) -> None:
        """Retire abandoned writes before failed request slots can be reused."""
        try:
            for req_index, update in plan.block_table_updates.items():
                self.block_tables.append_block_ids(
                    req_index, (update.block_ids,), overwrite=update.start == 0
                )
            self.block_tables.apply_staged_writes()
        except BaseException as error:
            try:
                for table in self.block_tables.block_tables:
                    table.clear_staged_writes()
                if not runtime_unusable(error):
                    # Upload or launch may have submitted work before failing.
                    # Finish its readers before reusing the UVA staging pools.
                    torch.accelerator.synchronize(self.device)
                    for req_index, update in plan.block_table_updates.items():
                        self.block_tables.num_blocks.np[:, req_index] = update.start
            except BaseException as cleanup_error:
                if runtime_unusable(error):
                    raise error from cleanup_error
                if runtime_unusable(cleanup_error):
                    raise
                raise CompletionUnknownError(
                    "failed block-table preparation could not be retired"
                ) from cleanup_error
            raise

    @torch.inference_mode()
    def execute(
        self,
        plan: DraftStepPlan,
        features: Features,
        *,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        if plan.num_tokens > self.max_num_tokens:
            raise ValueError("draft step exceeds the configured feature-token budget")
        values = {
            slot.kind: tensor
            for slot, tensor in zip(self.contract.schema.slots, features)
        }
        tokens = values["token_ids"]
        positions = values["positions"]
        assert tokens is not None and positions is not None
        if not torch.equal(positions, torch.from_numpy(plan.positions)):
            raise ValueError(
                "transported positions disagree with committed draft state"
            )
        vocab = self.vllm_config.model_config.get_vocab_size()
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= vocab):
            raise ValueError("target input token is outside the vocabulary")
        next_tokens = np.where(
            plan.num_sampled > 0, plan.last_sampled, plan.next_prefill_tokens[:, 0]
        )
        if np.any((next_tokens < 0) | (next_tokens >= vocab)):
            raise ValueError("target next input token is outside the vocabulary")
        torch.accelerator.set_device_index(self.device)
        with set_current_vllm_config(self.vllm_config):
            self.prepare_block_tables(plan)
            batch = make_input_batch(plan, tokens, self.input_buffers)
            tables = self.block_tables.gather_block_tables(
                batch.idx_mapping, plan.num_reqs
            )
            slots = self.block_tables.compute_slot_mappings(
                batch.idx_mapping,
                batch.query_start_loc,
                batch.positions,
                batch.num_tokens,
            )
            metadata = self.model_state.prepare_attn(
                batch,
                CUDAGraphMode.NONE,
                tables,
                slots,
                self.attn_groups,
                self.kv_config,
            )
            self.last_sampled[batch.idx_mapping] = torch.from_numpy(
                plan.last_sampled
            ).to(self.device)
            self.lookahead[0, batch.idx_mapping] = torch.from_numpy(
                plan.next_prefill_tokens[:, 0]
            ).to(self.lookahead)
            aux = values.get("aux_hidden_states")
            hidden = values.get("last_hidden_states")
            if aux is not None:
                aux_states = list(
                    aux.to(self.device).split(
                        self.vllm_config.model_config.get_hidden_size(), dim=-1
                    )
                )
                hidden = aux_states[0]
            else:
                aux_states = None
                assert hidden is not None
                hidden = hidden.to(self.device)
            drafts = self.speculator.propose(
                batch,
                metadata,
                build_slot_mappings_by_layer(slots, self.kv_config),
                hidden,
                aux_states,
                torch.from_numpy(plan.num_sampled).to(self.device),
                torch.from_numpy(plan.num_rejected).to(self.device),
                self.last_sampled,
                self.lookahead,
                self.temperature,
                self.seeds,
                num_speculative_tokens=num_speculative_tokens,
            )
            # The return copy completes the step before another client uses
            # the shared runner buffers or acknowledges committed KV.
            return drafts.cpu()


def make_gpu_adapter(
    vllm_config: VllmConfig,
    *,
    device: torch.device,
    kv_cache_bytes: int,
    shared_weights_source: str = "auto",
) -> TargetConditionedDraftAdapter:
    executor = GPUDraftStepExecutor(
        vllm_config,
        device=device,
        kv_cache_bytes=kv_cache_bytes,
        shared_weights_source=shared_weights_source,
    )
    spec = vllm_config.speculative_config
    assert spec is not None and spec.method is not None
    planner = DraftStepPlanner(
        DraftBlockPool(executor.num_blocks, executor.block_size),
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_model_len=vllm_config.model_config.max_model_len,
        num_speculative_tokens=spec.num_speculative_tokens,
    )
    return TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=executor.contract.capabilities,
        feature_schema=executor.contract.schema,
        methods=(spec.method,),
        draft_checkpoint_fingerprint=executor.contract.draft_fingerprint,
    )
