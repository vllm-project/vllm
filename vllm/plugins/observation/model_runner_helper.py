# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import TYPE_CHECKING, Any, List
from vllm.plugins.observation.interface import RequestContext, ObservationAction, load_observation_plugins
from vllm.plugins.observation.hook import ObservationHook

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.config.vllm import VllmConfig

logger = logging.getLogger(__name__)

def init_observation(runner: "GPUModelRunner", vllm_config: "VllmConfig") -> None:
    """
    Initializes the Observation Plugin API on a GPUModelRunner instance.
    Wraps methods dynamically to minimize footprint in core vLLM files.
    """
    observation_plugins = getattr(vllm_config, "observation_plugins", None)
    if not observation_plugins:
        return

    runner.plugin_manager = load_observation_plugins(
        observation_plugins, vllm_config
    )
    if not runner.plugin_manager:
        return

    if runner.plugin_manager.observe_decode:
        logger.warning(
            "observe_decode=True requested but decode observation is not yet "
            "implemented. Only prefill phases will be observed."
        )

    def _is_prefill_batch() -> bool:
        if not getattr(runner, "input_batch", None) or not getattr(runner, "requests", None):
            return False
        for req_id in runner.input_batch.req_ids:
            req_state = runner.requests.get(req_id)
            if req_state and req_state.num_computed_tokens < req_state.num_prompt_tokens:
                return True
        return False

    # Attach helper to instance
    setattr(runner, "_is_prefill_batch", _is_prefill_batch)

    # Wrap load_model to initialize the hook once model is ready
    original_load_model = runner.load_model
    def wrapped_load_model(*args, **kwargs) -> None:
        original_load_model(*args, **kwargs)
        runner.observation_hook = ObservationHook(
            runner.plugin_manager, runner.model
        )
    runner.load_model = wrapped_load_model

    # Wrap _model_forward to install PyTorch hooks only during forward pass
    original_model_forward = runner._model_forward
    def wrapped_model_forward(*args, **kwargs) -> Any:
        is_prefill = _is_prefill_batch()
        if is_prefill and getattr(runner, "observation_hook", None):
            runner.observation_hook.install_hooks()
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            if is_prefill and getattr(runner, "observation_hook", None):
                runner.observation_hook.remove_hooks()
    runner._model_forward = wrapped_model_forward

    # Wrap execute_model to handle request lifecycle and step processing
    original_execute_model = runner.execute_model
    def wrapped_execute_model(scheduler_output: Any) -> Any:
        if runner.plugin_manager:
            for req_id in scheduler_output.finished_req_ids:
                runner.plugin_manager.on_request_complete(req_id)
            for new_req_data in scheduler_output.scheduled_new_reqs:
                runner.plugin_manager.on_request_start(new_req_data.req_id, prompt=None)

        output = original_execute_model(scheduler_output)

        is_prefill = _is_prefill_batch()
        if is_prefill and getattr(runner, "observation_hook", None) and getattr(runner, "input_batch", None):
            request_contexts = []
            current_offset = 0
            for req_id in runner.input_batch.req_ids:
                num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                req_state = runner.requests.get(req_id)
                if not req_state:
                    continue
                
                req_is_prefill = req_state.num_computed_tokens < req_state.num_prompt_tokens

                request_contexts.append(RequestContext(
                    request_id=req_id,
                    is_prefill=req_is_prefill,
                    batch_offset=current_offset,
                    num_tokens=num_tokens,
                    num_cached_tokens=req_state.num_computed_tokens,
                ))
                current_offset += num_tokens

            observation_results = runner.observation_hook.process_step(
                request_contexts
            )

            aborted_req_ids = []
            for req_ctx, result in zip(request_contexts, observation_results):
                if result.action == ObservationAction.ABORT:
                    req_id = req_ctx.request_id
                    message = result.metadata.get("message", "No message")
                    logger.warning(
                        "Aborting request %s due to observation plugin trigger: %s",
                        req_id,
                        message,
                    )
                    req_state = runner.requests.get(req_id)
                    if req_state:
                        req_state.aborted_by_observation = True

            # Check for aborted requests in output
            if hasattr(output, "req_ids"):
                for req_id in output.req_ids:
                    if req_id in runner.requests:
                        req_state = runner.requests[req_id]
                        if getattr(req_state, "aborted_by_observation", False):
                            aborted_req_ids.append(req_id)
            
            if hasattr(output, "aborted_req_ids"):
                output.aborted_req_ids = list(set(getattr(output, "aborted_req_ids", []) or []) | set(aborted_req_ids))

        return output

    runner.execute_model = wrapped_execute_model
