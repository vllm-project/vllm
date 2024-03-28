"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py --forked`.
"""
import gc
import re

import pytest
import torch

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

MODELS = ["facebook/opt-125m"]


class LLMExecutionTracer:
    """Trace the execution of LLM model and capture the outputs of each layer.
    
    This class is used to compare the outputs of HF and vLLM. It's only
    expected to run single request for each batch.
    """

    def __init__(self, model, is_hf: bool = False):
        """Initialize the tracer.

        Args:
            model: The torch.nn to trace.
            is_hf: Whether the model is from huggingface transformer.
        """
        self._captured = {}
        self._layer_modules = {}
        self._register_module(model)
        self._start()
        self.request_id = 0
        self.is_hf = is_hf

    def _hook(self, module, input_, output) -> None:
        if module not in self._layer_modules:
            return
        layer = self._layer_modules[module]

        if self.is_hf:
            # hf model has an extra dimension for batch size.
            output = output[0].view(-1, output[0].shape[-1])

        # store output by request_id, layer, iteration
        self._captured.setdefault(self.request_id, dict()).setdefault(layer, list())\
            .append((None, output.to("cpu")))

    def start_new_request(self, request_id) -> None:
        self.request_id = request_id

    @property
    def captured(self) -> None:
        return self._captured

    def stop(self) -> None:
        self.handle.remove()

    def _start(self) -> None:
        self.handle = torch.nn.modules.module.register_module_forward_hook(
            self._hook)

    def _register_module(self, module) -> None:
        for module_name, module in module.named_modules():
            layer = self._find_layer_module(module_name)
            if layer is not None:
                self._layer_modules[module] = layer

    def _find_layer_module(self, module_name: str):
        if m := re.match("^.*layers\.([0-9]+)$", module_name):
            print(f"matched {module_name}")
            return int(m.group(1))
        return None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_outputs = []
    vllm_outputs = []
    example_prompts = example_prompts[:7]

    vllm_model = vllm_runner(model, dtype=dtype, enforce_eager=True)
    vllm_tracer = LLMExecutionTracer(
        vllm_model.model.llm_engine.model_executor.driver_worker.model_runner.model, is_hf=False)
    for id, prompt in enumerate(example_prompts):
        vllm_tracer.start_new_request(id)
        vllm_outputs.extend(vllm_model.generate_greedy([prompt], max_tokens))
    vllm_captured = vllm_tracer.captured
    vllm_tracer.stop()
    del vllm_tracer
    del vllm_model.model
    del vllm_model
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    hf_model = hf_runner(model, dtype=dtype)
    hf_tracer = LLMExecutionTracer(hf_model.model, is_hf=True)
    for id, prompt in enumerate(example_prompts):
        hf_tracer.start_new_request(id)
        hf_outputs.extend(hf_model.generate_greedy([prompt], max_tokens))
    hf_captured = hf_tracer.captured
    hf_tracer.stop()
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    for request_id in range(len(example_prompts)):
        diffs = list()
        for iteration in range(max_tokens):
            for layer in range(len(vllm_captured)):
                _, hf_output = hf_captured[request_id][layer][iteration]
                _, vllm_output_padded = vllm_captured[request_id][
                    layer][iteration]
                vllm_output = vllm_output_padded[:hf_output.shape[0], :]
                diff = torch.sum(
                    torch.abs(hf_output - vllm_output)) / torch.sum(
                        torch.abs(hf_output))
                diffs.append(diff.item())

        print("request_id {} avg difference {:.2f}%".format(
            request_id,
            torch.mean(torch.FloatTensor(diffs)).item() * 100))
        assert torch.mean(torch.FloatTensor(diffs)).item() < 0.005
