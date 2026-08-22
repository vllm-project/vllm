# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import LATE_INTERACTION_MODE_SCORE_DOC
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

_SUPPORTED_TASKS: frozenset[PoolingTask] = frozenset(
    {"embed", "classify", "token_embed", "token_classify", "embed&token_classify"}
)


class PoolingRunner:
    def __init__(self, model: nn.Module, vllm_config: VllmConfig):
        self.model = cast(VllmModelForPooling, model)
        self.model_config = vllm_config.model_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        model_tasks = tuple(sorted(self.model.pooler.get_supported_tasks()))
        selected_task = self.model_config.get_pooling_task(model_tasks)
        if selected_task not in _SUPPORTED_TASKS:
            hint = (
                "Set an explicitly supported task or VLLM_USE_V2_MODEL_RUNNER=0."
                if _SUPPORTED_TASKS.intersection(model_tasks)
                else "Set VLLM_USE_V2_MODEL_RUNNER=0 to use this model."
            )
            raise ValueError(
                "Model Runner V2 supports pooling tasks "
                f"{sorted(_SUPPORTED_TASKS)}, but this model selects "
                f"{selected_task!r} from {list(model_tasks)}. {hint}"
            )
        self.supported_tasks = frozenset(self.get_supported_tasks(model))
        if not self.supported_tasks:
            raise ValueError(
                "Model Runner V2 supports pooling tasks "
                f"{sorted(_SUPPORTED_TASKS)}, but this model supports "
                f"{list(model_tasks)}. "
                "Set VLLM_USE_V2_MODEL_RUNNER=0 to use this model."
            )
        self.pooling_params: dict[int, PoolingParams] = {}
        self.pooling_states: dict[int, PoolingStates] = {}
        self.prompt_token_ids: dict[int, torch.Tensor] = {}
        self.late_interaction_runner = LateInteractionRunner()

        # Flash late-interaction scoring: honor the frontend opt-out
        # (PoolerConfig.enable_flash_late_interaction) and warm the Triton
        # kernels at load only when the model can actually use them —
        # same capability + pooling-method gates as the runtime zero-copy
        # path (StepPool subclasses AllPool but filters tokens, so exact
        # type check, not isinstance).
        pooler_config = getattr(self.model_config, "pooler_config", None)
        self._flash_late_interaction_enabled = pooler_config is None or getattr(
            pooler_config, "enable_flash_late_interaction", True
        )
        pooler = getattr(self.model, "pooler", None)
        pooler_head = getattr(pooler, "head", None) if pooler is not None else None
        if (
            self._flash_late_interaction_enabled
            and pooler_head is not None
            and hasattr(pooler_head, "project_batch")
            and type(getattr(pooler, "pooling", None)) is AllPool
        ):
            self.late_interaction_runner.warmup_kernels()

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        return sorted(model.pooler.get_supported_tasks() & _SUPPORTED_TASKS)

    def add_request(
        self,
        req_id: str,
        req_index: int,
        pooling_params: PoolingParams,
        prompt_token_ids: list[int],
    ) -> None:
        task = pooling_params.task
        if task not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task: {task!r}. "
                f"Supported tasks: {sorted(self.supported_tasks)}"
            )
        self.model.pooler.get_pooling_updates(task).apply(pooling_params)
        self.late_interaction_runner.register_request(req_id, pooling_params)
        self.pooling_params[req_index] = pooling_params
        self.pooling_states[req_index] = PoolingStates()
        if pooling_params.requires_token_ids:
            self.prompt_token_ids[req_index] = torch.tensor(
                prompt_token_ids, dtype=torch.int64
            )

    def remove_request(self, req_index: int) -> None:
        self.pooling_params.pop(req_index, None)
        if state := self.pooling_states.pop(req_index, None):
            state.clean()
        self.prompt_token_ids.pop(req_index, None)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        self.late_interaction_runner.on_requests_finished(req_ids)

    def clear(self) -> None:
        self.late_interaction_runner.clear()

    def _get_pooling_metadata(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        device: torch.device,
    ) -> PoolingMetadata:
        req_indices = input_batch.idx_mapping_np.tolist()
        pooling_params = [self.pooling_params[i] for i in req_indices]
        pooling_states = [self.pooling_states[i] for i in req_indices]
        prompt_lens = torch.from_numpy(
            req_states.prompt_len.np[input_batch.idx_mapping_np].copy()
        )

        prompt_token_ids_cpu = None
        prompt_token_ids = None
        if any(params.requires_token_ids for params in pooling_params):
            max_prompt_len = int(prompt_lens.max())
            prompt_token_ids_cpu = torch.zeros(
                (input_batch.num_reqs, max_prompt_len),
                dtype=torch.int64,
                pin_memory=PIN_MEMORY,
            )
            for i, (req_index, params) in enumerate(zip(req_indices, pooling_params)):
                if not params.requires_token_ids:
                    continue
                token_ids = self.prompt_token_ids[req_index]
                prompt_token_ids_cpu[i, : token_ids.numel()] = token_ids
            prompt_token_ids = prompt_token_ids_cpu.to(device, non_blocking=True)

        return PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        hidden_states = hidden_states[: input_batch.num_tokens]
        pooling_metadata = self._get_pooling_metadata(
            input_batch, req_states, hidden_states.device
        )
        num_reqs = input_batch.num_reqs
        # Pooling has no speculative tokens, so this CPU upper bound is exact.
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
        )
        # Zero-copy path: project the entire batch in one matmul, then
        # build per-request output by slicing (views, no copy). Doc slices
        # stay contiguous in projected_batch — the rerank kernel reads them
        # directly, skipping torch.cat entirely.
        #
        # Requirements / fallbacks to the normal pooler (same gates as the
        # former MRv1 hook):
        #   - Chunked prefill: AllPool caches partial chunks that the
        #     slice bypass cannot reconstruct.
        #   - Matryoshka (dimensions != embed_dim): forward_chunk truncates
        #     BEFORE activation; project_batch normalizes the full vector.
        #   - use_activation=False: project_batch unconditionally applies
        #     activation.
        #   - Pooling method must be exactly AllPool: StepPool (an AllPool
        #     subclass) filters tokens via step_tag_id/returned_token_ids
        #     BEFORE the head — hence type(...) is AllPool, not isinstance.
        projected_batch = None
        cursor = pooling_metadata.pooling_cursor
        # Gate on a SCORE_DOC request in THIS batch (not just globally
        # pending docs): a step carrying only queries or unrelated pooling
        # requests must not pay the full-batch projection or its memory.
        batch_has_docs = any(
            p.late_interaction_params is not None
            and p.late_interaction_params.mode == LATE_INTERACTION_MODE_SCORE_DOC
            for p in pooling_metadata.pooling_params
        )
        use_zerocopy = (
            self._flash_late_interaction_enabled
            and batch_has_docs
            and not os.environ.get("VLLM_DISABLE_ZEROCOPY")
            and hasattr(self.model.pooler, "head")
            and hasattr(self.model.pooler.head, "project_batch")
            and type(getattr(self.model.pooler, "pooling", None)) is AllPool
            and cursor is not None
            and not cursor.is_partial_prefill()
        )

        if use_zerocopy:
            # Every doc request must use params compatible with
            # project_batch (no matryoshka truncation, activation on).
            for p in pooling_metadata.pooling_params:
                lip = p.late_interaction_params
                if (
                    lip is not None
                    and lip.mode == LATE_INTERACTION_MODE_SCORE_DOC
                    and (p.dimensions is not None or p.use_activation is False)
                ):
                    use_zerocopy = False
                    break

        if use_zerocopy:
            assert cursor is not None  # narrowed by use_zerocopy
            pooler = self.model.pooler
            # One matmul for the full batch — all tokens projected.
            projected_batch = pooler.head.project_batch(hidden_states)  # type: ignore[operator,union-attr]
            # Per-request slicing: AllPool extracts views of hidden_states.
            pooled_data = pooler.pooling(hidden_states, pooling_metadata)  # type: ignore[operator]
            # Offsets from CPU-side scheduled-token metadata — no GPU sync
            # (MRv2's execution loop must stay free of CPU sync points).
            fcpu = cursor.first_token_indices_cpu
            lcpu = cursor.last_token_indices_cpu
            if fcpu is not None and lcpu is not None:
                firsts = fcpu.tolist()
                lasts = lcpu.tolist()
            else:  # fallback: single sync (older cursor producers)
                firsts = cursor.first_token_indices_gpu.tolist()
                lasts = cursor.last_token_indices_gpu.tolist()
            params_list = pooling_metadata.pooling_params
            # Build output: docs get projected_batch slices (views),
            # queries/other go through the normal forward_chunk.
            zerocopy_output: list[torch.Tensor | None] = []
            for i in range(num_reqs):
                lip = params_list[i].late_interaction_params
                if lip is not None and lip.mode == LATE_INTERACTION_MODE_SCORE_DOC:
                    # View into projected_batch — no copy, no extra alloc.
                    zerocopy_output.append(projected_batch[firsts[i] : lasts[i] + 1])
                else:
                    zerocopy_output.append(
                        pooler.head.forward_chunk(  # type: ignore[operator,union-attr]
                            pooled_data[i], params_list[i]
                        )
                    )
            pooler_output: PoolerOutput = zerocopy_output
            # Release AllPool views so hidden_states can be freed; doc
            # entries are views of projected_batch (not hidden_states).
            del pooled_data
        else:
            pooler_output = self.model.pooler(hidden_states, pooling_metadata)

        finished_mask = pooling_metadata.get_pooling_cursor().is_finished().tolist()
        pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=input_batch.req_ids,
            finished_mask=finished_mask,
            projected_batch=projected_batch,
            pooling_cursor=pooling_metadata.pooling_cursor,
        )
        return pooler_output, finished_mask

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        num_reqs = min(num_tokens, self.max_num_reqs)
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        prompt_lens = torch.from_numpy(num_scheduled_tokens)

        pooling_params = PoolingParams(task=task)
        pooling_params.verify(self.model_config)
        self.model.pooler.get_pooling_updates(task).apply(pooling_params)
        prompt_token_ids = None
        if pooling_params.requires_token_ids:
            prompt_token_ids = torch.zeros(
                (num_reqs, int(prompt_lens.max())),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=None
            if prompt_token_ids is None
            else prompt_token_ids.cpu(),
            pooling_params=[pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens,
            seq_lens_cpu=prompt_lens,
            device=hidden_states.device,
        )
        try:
            return self.model.pooler(hidden_states, pooling_metadata)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            raise RuntimeError(
                "CUDA out of memory occurred when warming up pooler "
                f"({task=}) with {num_reqs} dummy requests. Please try "
                "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                "initializing the engine."
            ) from e

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        for task in sorted(self.supported_tasks):
            self._dummy_pooler_run_task(hidden_states, task)
