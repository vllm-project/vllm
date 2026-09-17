# Custom Logits Processors

!!! important
    Some logits processors design changes are still in progress and the API may
    change in the near future. We hope to stabilize this part of the API soon

A "custom" logits processor is written by a user of vLLM and is loaded into vLLM at initialization without needing to modify or recompile the vLLM source code. It is the opposite of a built-in logits processor.

This document shows how to write, load and use a custom logits processor.

!!! note
    This document covers the Model Runner V2 (MRV2) interface. The legacy V1
    model runner (`VLLM_USE_V2_MODEL_RUNNER=0`) uses a different interface,
    `vllm.v1.sample.logits_processor.LogitsProcessor`, which is not covered
    here.

## Logits Processors Background

A logits processor adjusts the next-token probability distribution, usually with the intention of steering the model towards a desired type of behavior.

In vLLM, logits processors operate at batch granularity. During a given engine step, the logits processor consumes a `(num_logits_rows) x (vocab_size)` tensor of raw logits output by the model. Note that a logits row is not a request: rows are reordered every step, and under speculative decoding a request owns one row per draft token. The logits processor applies a transformation to the rows of the logits tensor, while leaving other rows unmodified. The transformed logits tensor is then used for sampling.

## Creating a Custom Logits Processor

Custom logits processors must subclass `vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor` and define (at minimum) the following methods:

* `__init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState)`:
    * `vllm_config`: engine configuration data structure
    * `req_states`: a narrow, read-only view of the persistent batch, exposing the on-device token history (`all_token_ids`, `prompt_len`, `prefill_len`, `total_len`) plus `device`, `max_num_reqs` and `vocab_size`

* `add_request(self, req_idx, sampling_params) -> bool`:
    * Initialize per-slot state for a request entering the batch. Slots are recycled through a free list, so per-slot state must be fully (re)initialized here; there is no removal hook, since freed slots are never read
    * Return whether this processor modifies logits for the request. The sampler skips the logits-processing pipeline for batches in which no request needs it

* `apply_staged_writes(self) -> None` (optional):
    * Flush host-side writes staged by `add_request()` to the device; called once per step before the forward pass

* `validate_params(cls, sampling_params) -> None` (optional classmethod):
    * Raise `ValueError` for invalid per-request arguments (especially custom arguments); runs at request admission, so invalid arguments fail the request instead of reaching the sampler

* `apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor`:
    * Consume a `(num_logits_rows) x (vocab_size)` logits tensor and the step's batch layout (`ctx`: row-to-slot mappings, `input_ids`, positions)
    * Apply logits processor transformation at batch granularity
    * Return a transformed logits tensor. You can modify the input logits in-place or out-of-place; in-place is more memory-efficient

### Passing Custom Argument to a Custom Logits Processor

Unlike built-in logits processors, custom logits processors may require configuration arguments that are not hard-coded into `SamplingParams` or the vLLM server REST API. To solve this problem, custom logits processors may leverage vLLM [custom arguments](./custom_arguments.md) support to receive configuration settings from the user (although you are also free to design a custom logits processor which utilizes the pre-existing fields in `SamplingParams`.)

### Example Custom Logits Processor Implementation

The contrived example below implements a custom logits processor which masks out all tokens except for one (`target_token`) for the requests that enable it. The processor is disabled for any request that does not specify `target_token`; `add_request()` reports this via its return value, so the sampler can skip the logits-processing pipeline entirely when no request enables the processor. To determine whether the logits processor is enabled and which token to leave unmasked, the processor checks `SamplingParams.extra_args` for a `target_token` custom argument associated with each request:

??? code "Example custom logits processor definition"

    ``` python
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor import (
        LogitsProcRequestState,
        LogitsContext,
        LogitsProcessor,
    )


    class TargetTokenLogitsProcessor(LogitsProcessor):
        """Masks out all tokens except `target_token` (a per-request custom
        argument); requests without it are left alone."""

        def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
            # Per-slot target; -1 means disabled. Staged on the host and
            # flushed to the device once per step in apply_staged_writes().
            self.target_token = torch.full(
                (req_states.max_num_reqs,), -1, dtype=torch.int64
            )
            self.target_token_dev = torch.full(
                (req_states.max_num_reqs,), -1, dtype=torch.int64, device=req_states.device
            )

        def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
            target = (sampling_params.extra_args or {}).get("target_token")
            self.target_token[req_idx] = target if target is not None else -1
            return target is not None

        def apply_staged_writes(self) -> None:
            self.target_token_dev.copy_(self.target_token, non_blocking=True)

        def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
            # ctx.expanded_idx_mapping maps each logits row to its request slot.
            cols = self.target_token_dev[ctx.expanded_idx_mapping.long()]
            rows = torch.nonzero(cols >= 0).squeeze(1)
            if rows.numel() == 0:
                return logits
            kept = logits[rows, cols[rows]].clone()
            logits[rows] = float("-inf")
            logits[rows, cols[rows]] = kept
            return logits
    ```

Per-request state is keyed by the request slot index, and slots are recycled through a free list. The example keeps a per-slot state where `-1` means disabled; note that `add_request()` overwrites the slot's entry unconditionally, which is what keeps recycled slots from leaking state between requests.

## Ways to Load Your Custom Logits Processor in vLLM

Logits processors are loaded at initialization. Critically, the set of loaded logits processors cannot be modified after the vLLM engine finishes loading, and new logits processors cannot be loaded on-demand for individual requests.

Loaded classes are validated against the Model Runner V2 interface; a class that does not subclass `vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor` is rejected with a clear error.

This section details different ways of making your logits processor visible to vLLM and triggering vLLM to load your logits processor.

### Method 1: Pass the Custom Logits Processor Fully-Qualified Class Name (FQCN) to vLLM at Initialization Time

This method is supported in both offline and online vLLM usage scenarios. The custom logits processor's FQCN (in the form of `dotted.path.to.module:ClassName`) can be passed as an argument to the `LLM` and `AsyncLLM` Python constructors, or as a CLI argument to `vllm serve` with the following syntax

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
        logits_processors=["your.module.path:TargetTokenLogitsProcessor"],
    )
    ```

??? code "Passing custom logits processor FQCN to `AsyncLLM` in Python"

    ``` python
    # Pass in FQCN
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=["your.module.path:TargetTokenLogitsProcessor"])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

??? code "Passing custom logits processor FQCN to vLLM server via CLI"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:TargetTokenLogitsProcessor
    ```

### Method 2: Automatically Detect Custom Logits Processors Installed in Your Python Environment As Entry Points

[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) can enable installed packages to make themselves available as plugins to other Python programs, via pieces of metadata known as "entry points".

During initialization, vLLM automatically scans the `vllm.logits_processors` entry point group and loads any installed logits processors which it finds.

Suppose that you have developed a Python package that holds your custom logits processors. You can expose each logits processor to vLLM by adding a unique entrypoint for each logits processor to your logits processor Python package. The example below shows how to add an entrypoint to your project's `pyproject.toml` file:

??? code "Exposing a custom logits processor as a Python entrypoint"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    target_token_logits_processor = "your.module.path:TargetTokenLogitsProcessor"
    ```

Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to the `LLM` or `AsyncLLM` constructors or to the vLLM server explicitly at initialization time if your logits processor is exposed as an entry point.

!!! note
    vLLM will *always* load *all* logits processors which are exposed via entrypoints under the `vllm.logits_processors` grouping.

### Method 3 (Offline-only): Pass a Python Class Object to the vLLM Constructor

You can pass one or more custom logits processor class objects to the `LLM` and `AsyncLLM` constructors. This option is very flexible, as the logits processor classes may either be (1) defined locally within the same Python source file where `LLM` or `AsyncLLM` is instantiated, or (2) imported from a Python package.

??? code "Passing custom logits processor class object to `LLM` or `AsyncLLM` in Python"

    ``` python
    # Import custom logits processor
    from some.module import TargetTokenLogitsProcessor

    # ...or...

    # Define custom logits processor locally
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

    class TargetTokenLogitsProcessor(LogitsProcessor):
        # See TargetTokenLogitsProcessor implementation above
        ...

    # Pass class object to LLM constructor
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[TargetTokenLogitsProcessor],
    )

    # Pass class object to AsyncLLM constructor
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=[TargetTokenLogitsProcessor])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

## Invoking a Custom Logits Processor Against a Request

The design of the custom logits processor determines whether the logits processor must be enabled/disabled for a given request, and what arguments must be provided to configure the logits processor.

The examples below show how a user would pass a custom argument (`target_token`) to `TargetTokenLogitsProcessor` in order to (1) enable the logits processor for that particular request and (2) control the logits processor's behavior.

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

??? code "Offline: configure custom logits processor for an `LLM` request"

    ``` python
    outputs_logitproc = llm.generate("your prompt",
                                     SamplingParams(...,
                                        extra_args={"target_token": 67}))
    ```

??? code "Offline: configure custom logits processor for an `AsyncLLM` request"

    ``` python
    async for out in engine.generate(request_id="your request id",
                                     prompt="your prompt",
                                     sampling_params=SamplingParams(...,
                                        extra_args={"target_token": 67})):

        # Process async request outputs
        ...
    ```

## Best Practices for Writing Custom Logits Processors

* Write an efficient `apply()` implementation in light of the fact that logits processors operate at batch granularity. For example, you may be able to use efficient vectorized operations to implement `apply()`, and to stage per-request state into contiguous tensors in `add_request()` instead of per-row Python loops

* It is up to the logits processor author to determine:

    1. **The per-request attributes which configure the logits processor's behavior against that request.** Your custom logits processor's `add_request()` override determines how `SamplingParams` fields are mapped into logits processor state

    2. **The conditions under which the logits processor is or is not enabled on a per-request basis.** Unless your intention is for the custom logits processor to act on all requests all the time, you should write your logits processor in such a way that it is possible to disable the logits processor for a given request, i.e. by defaulting an argument to `None` or by passing in a specific do-nothing argument value i.e. `0.0`. Return `False` from `add_request()` for such requests so that the sampler can skip the logits-processing pipeline for batches in which no request needs it

* Stage host-side writes in `add_request()` and flush them once per step in `apply_staged_writes()`, instead of writing device tensors directly; this is the discipline every built-in sampler state follows

* Since there is no removal hook, `add_request()` must fully (re)initialize the slot's state; do not rely on state left behind by the slot's previous occupant
