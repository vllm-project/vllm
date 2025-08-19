# Custom Logits Processors

This document shows you how to augment vLLM with custom logits processors.

## Build a Custom Logits Processor and Pass It to the vLLM Engine

Subclass `vllm.v1.sample.logits_processor.LogitsProcessor` in order to implement a custom logits processor.

The contrived example below implements a custom logits processor which masks out all tokens except for one (`target_token`) with `float(-inf)`. The logits processor is disabled for any request that does not specify `target_token`.

??? code "Example custom logits processor definition"

    ```python
    from typing import Optional
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (BatchUpdate,
                                                LogitsProcessor,
                                                MoveDirectionality)

    class DummyLogitsProcessor(LogitsProcessor):
        """Fake logit processor to support unit testing and examples"""

        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool):
            self.req_info: dict[int, SamplingParams] = {}

        def is_argmax_invariant(self) -> bool:
            """Never impacts greedy sampling"""
            return False

        def update_state(self, batch_update: Optional[BatchUpdate]):
            if not batch_update:
                return

            # Process added requests.
            for index, params, _, _ in batch_update.added:
                assert params is not None
                if params.extra_args and (target_token :=
                                        params.extra_args.get("target_token")):
                    self.req_info[index] = target_token

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # Save target values before modification
            rows_list = list(self.req_info.keys())
            cols = torch.tensor([self.req_info[i] for i in rows_list],
                                dtype=torch.long,
                                device=logits.device)
            rows = torch.tensor(rows_list, dtype=torch.long, device=logits.device)
            values_to_keep = logits[rows, cols].clone()
    ```

Pass your custom logits processor to the `LLM` constructor in the form of (1) a class object or (2) a fully-qualified class name (FQCN), as shown in the example below (which assumes that `DummyLogitsProcessor` is defined in `your.module.path`):

??? code "Passing custom logits processor to `LLM` in Python"

    ``` python
    # Pass in class object
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[DummyLogitsProcessor],
    )

    # Pass in FQCN
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "Passing custom logits processor to vLLM server via CLI"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

## Configure The Custom Logits Processor for a Request

To enable the logits processor for a request, pass `target_token` in with the request as a vLLM [custom argument](./custom_arguments.md):

??? code "Python: configure custom logits processor for a request"

    ``` python
    outputs_logitproc = llm.generate("your prompt", 
                                     SamplingParams(...,
                                        extra_args={"target_token": 67}))
    ```

??? code "vLLM REST API: configure custom logits processor for a request"

    ``` bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            ...
            "vllm_xargs": {"target_token": 67}
        }'
    ```

??? code "OpenAI SDK: configure custom logits processor for a request"

    ``` python
    batch = await client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        ...,
        extra_body={
            "vllm_xargs": {
                "target_token": 67
            }
        }
    )
    ```

## Logits Processor Programming Model

* `__init__(self, vllm_config: vllm.config.VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`: vLLM engine configuration
* `is_argmax_invariant(self)`
* `update_state(self, batch_update: Optional[vllm.v1.sample.logits_processor.BatchUpdate])`
    * `batch_update`: representation of added/removed/moved requests in the vLLM persistent batch during the most recent engine step
* `apply(self, logits: torch.Tensor)`
    * `logits`: a $num\_reqs \times vocab\_size$ tensor representing the unprocessed token probability distribution for each request.
