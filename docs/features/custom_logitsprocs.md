# Custom Logits Processors

A "custom" logits processor is written by a user of vLLM and is loaded into vLLM at initialization without needing to modify or recompile the vLLM source code. It is the opposite of a built-in logits processor.

This document shows how to write, load and use a custom logits processor.

Review the [logits processor design documentation](../design/logits_processors.md) for baseline guidance on writing correct and efficient logits processors.

## Writing a Custom Logits Procesor

Custom logits processors must be subclasses of `vllm.v1.sample.logits_processor.LogitsProcessor`. Unlike built-in logits processors, custom logits processors may require configuration arguments that are not hard-coded into `SamplingParams` or the vLLM server REST API. To solve this problem, custom logits processors may leverage vLLM [custom arguments](./custom_arguments.md) support to receive configuration settings from the user (although your are also free to design a custom logits processor which utilizes the pre-existing fields in `SamplingParams`.)

In vLLM logits processors operate at batch granularity. The contrived example below implements a custom logits processor which consumes a `(num\_requests) \times (vocab\_size)` logits tensor and masks out all tokens except for one (`target_token`) with `float(-inf)`. The logits processor is disabled for any request that does not specify `target_token`. To determine whether the logits processor is enabled and which token to leave unmasked, the logits processor checks `SamplingParams.extra_args` for a `target_token` custom argument associated with each request:

??? code "Example custom logits processor definition"

    ``` python
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
            self.req_info: dict[int, int] = {}

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

Throughout this document, we will use `DummyLogitsProcessor` as an example of a custom logits processor.

Once vLLM loads a logits processor during initialization, then vLLM will invoke `update_state()` and `apply()` against that logits processor in every engine step. Both methods operate on all requests which currently reside in the vLLM persistent batch. Thus it is important to implement these methods efficiently.

The `DummyLogitsProcessor.update_state()` implementation maintains a "sparse" representation of the batched requests in the `self.req_info` dictionary: only those requests which specify a `target_token` value have a key in the dictionary. `update_state()` adjusts the stored request indices and `target_token` values (keys and values respectively in `self.req_info`) in response to Add, Remove and Move operations against the persistent batch.

## Ways to Load Your Custom Logits Processor in vLLM

Logits processors are loaded at initialization. Critically, the set of loaded logits processors cannot be modified after the vLLM engine finishes loading, and new logits logits processors cannot be loaded on-demand for individual requests.

This section details different ways of making your logits processor visible to vLLM and triggering vLLM to load your logits processor.

### Method 1: Pass the Custom Logits Processor Fully-Qualified Class Name (FQCN) to vLLM at Initialization Time

This method is supported in both offline and online vLLM usage scenarios. The custom logits processor's FQCN (in the form of `dotted.path.to.module:ClassName`) can be passed as an argument to the `LLM` Python constructor, or as a CLI argument to `vllm serve` with the following syntax

``` bash
vllm serve ... --logits_processors <logits processor 1> <logits processor 2> ...
```

The only requirements on the FQCN are

1. Python's `importlib.import_module()` must be able to resolve the dotted path portion of the FQCN and load it as a module

2. The class-name portion of the FQCN must be possible to import from the loaded module

3. The object pointed to by the FQCN must be a subclass of `LogitsProcessor`

See examples below:

??? code "Passing custom logits processor FQCN to `LLM` in Python"

    ``` python
    # Pass in FQCN
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "Passing custom logits processor FQCN to vLLM server via CLI"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

### Method 2: Automatically Detect Custom Logits Processors Installed in Your Python Environment As Entry Points

[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) can enable installed packages to make themselves available as plugins to other Python programs, via pieces of metadata known as "entry points".

During initialization, vLLM automatically scans the `vllm.logits_processors` entry point group and loads any installed logits processors which it finds.

Suppose that you have developed a Python package that holds your custom logits processors. You can expose each logits processor to vLLM by adding a unique entrypoint for each logits processor to your logits processor Python package. The example below shows how to add an entrypoint to your project's `.toml` file:

??? code "Exposing a custom logits processor as a Python entrypoint"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to the `LLM` constructor or to the vLLM server explicitly at initialization time if your logits processor is exposed as an entry point.

!!! note
    vLLM will *always* load *all* logits processors which are exposed via entrypoints under the `vllm.logits_processors` grouping.

### Method 3 (Offline-only): Pass a Python Class Object to the vLLM Constructor

You can pass one or more custom logits processor class objects to the `LLM` constructor. This option is very flexible, as the logits processor classes may either be (1) defined locally within the same Python source file where `LLM` is instantiated, or (2) imported from a Python package.

??? code "Passing custom logits processor class object to `LLM` in Python"

    ``` python
    # Import custom logits processor
    from some.module import DummyLogitsProcessor

    # ...or...

    # Define custom logits processor locally
    from vllm.v1.sample.logits_processor import LogitsProcessor

    class DummyLogitsProcessor(LogitsProcessor):
        # See DummyLogitsProcessor implementation above
        ...

    # Pass class object to LLM constructor
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[DummyLogitsProcessor],
    )
    ```

## Invoking a Custom Logits Processor Against a Request

The design of the custom logits processor determines whether the logits processor must be enabled/disabled for a given request, and what arguments must be provided to configure the logits processor. For more information, review [the logits processors design documentation](../design/logits_processors.md).

The examples below show how a user would pass a custom argument (`target_token`) to `DummyLogitsProcessor` in order to (1) enable the logits processor for that particular request and (2) control the logits processor's behavior.

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

??? code "Offline: configure custom logits processor for a request"

    ``` python
    outputs_logitproc = llm.generate("your prompt", 
                                     SamplingParams(...,
                                        extra_args={"target_token": 67}))
    ```
