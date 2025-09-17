# Custom Logits Processors

!!! important
    Some logits processors design changes are still in progress and the API may
    change in the near future. We hope to stabilize this part of the API soon

A "custom" logits processor is written by a user of vLLM and is loaded into vLLM at initialization without needing to modify or recompile the vLLM source code. It is the opposite of a built-in logits processor.

This document shows how to write, load and use a custom logits processor.

## Logits Processors Background

A logits processor adjusts the next-token probability distribution, usually with the intention of steering the model towards a desired type of behavior.

In vLLM, logits processors operate at batch granularity. During a given engine step, the logits processor consumes a `(num_requests) x (vocab_size)` tensor of raw logits output by the model. For all requests which enable the logits processor, the logits processor applies a transformation to the corresponding row of the logits tensor, while leaving other rows unmodified. The transformed logits tensor is then passed to softmax.  

## Creating a Custom Logits Processor

Custom logits processors must subclass `vllm.v1.sample.logits_processor.LogitsProcessor` and define (at minimum) the following methods:

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`: engine configuration data structure
    * `device`: hardware accelerator device info
    * `is_pin_memory`: flag indicating whether pin memory is available to support logits processor implementation

* `apply(self, logits: torch.Tensor) -> torch.Tensor`:
    * Consume a `(num_requests) x (vocab_size)` logits tensor (`logits`)
    * Apply logits processor transformation at batch granularity
    * Return a transformed `(num_requests) x (vocab_size)` logits tensor
    * You can modify the input logits processors in-place or out-of-place; in-place is more memory-efficient

* `is_argmax_invariant(self) -> bool`:
    * Return `True` if the logits processor is argmax invariant (never changes what is the highest-logit-value token ID for a given request), `False` if the logits processor may modify argmax
    * `is_argmax_invariant()` is evaluated once at startup; if `True`, vLLM will skip applying this logits processor in a given step when all requests use greedy sampling

* `update_state(self, batch_update: Optional["BatchUpdate"]) -> None`:
    * Consume a `BatchUpdate` data structure representing persistent batch state changes at the beginning of the current engine step
    * Use the `BatchUpdate` members to update logits processor internal state
    * **Note:** batch update data structure may be `None`, signaling no change to the batch constituents. In this case, the LogitsProcessor might still want to update its state based on the updated `output_token_ids` lists that it could have retained when they were added.

### How the vLLM engine builds the `BatchUpdate` data structure

!!! important
    Some logits processors design changes are still in progress. We expect
    that in the future you will not need to account for batch state changes
    when implementing a logits processor, and the information in this section
    will become irrelevant.

Logits processor `update_state()` implementations should assume the following model for how the model runner updates persistent batch state (expressed here in terms of the `BatchUpdate` abstraction):

1. Identify indices of requests which finished in the current engine step

2. Identify new requests introduced in the current step

3. Use Add operations to replace as many finished requests with new requests, in order of increasing index of the replaced request starting with the lowest index

4. Based on the relative number of new and finished requests:

    1. If the numbers of new and finished requests are the same, proceed to next step

    2. *If there are more new requests than finished requests:* apply Add operations to extend the batch with the remaining new requests which did not replace finished requests. Assign consecutive indices to these new requests, starting with `current_max_batch_index + 1`

    3. *If there are fewer new requests than finished requests:*

        * Apply Remove operations to finished requests which were not replaced with new requests. These removed request indices will necessarily be greater than the greatest index of the finished requests which were replaced in the previous step. The Removes may leave the batch in a non-contiguous state

        * **"Condense" the batch to be contiguous:** starting with the lowest-index empty slot (which was caused by a Remove), apply a Unidirectional Move from the current highest non-empty slot in the batch to fill the empty slot. Proceed with additional Unidirectional Move operations in order of increasing empty slot destination index and decreasing non-empty slot source index until the batch is contiguous

        * **Shrink the batch:** a side-effect of condensing the batch is that empty slots resulting from Remove operations are grouped in a contiguous block at the end of the batch array. Thus, after condensing, update `BatchUpdate.batch_size` to reflect the number of non-empty slots

5. Reorder the batch for improved efficiency. Depending on the attention backend implementation and the current characteristics of the batch, zero or more Swap Move operations may be applied to reorder the batch

Notes:

* A logits processor `update_state()` method must process batch update operations in the following order: removes, adds, moves

* The index argument for Add operations refers to the index *at the time the Add occurred*, i.e. before any Move operations
    * Example: if a request is Added at index 5 and then swapped with index 3, the Add operation in `BatchUpdate.added` will be associated with index 5 not 3
    * In other words Move operations can be assumed to be applied after Adds and Removes

* Move operations can be assumed to be applied in the order in which they appear in `BatchUpdate.moved`

* If there are no new/finished requests and there is no batch reordering, then the batch update for the logits processors will be `None`

### Passing Custom Argument to a Custom Logits Processor

Unlike built-in logits processors, custom logits processors may require configuration arguments that are not hard-coded into `SamplingParams` or the vLLM server REST API. To solve this problem, custom logits processors may leverage vLLM [custom arguments](./custom_arguments.md) support to receive configuration settings from the user (although you are also free to design a custom logits processor which utilizes the pre-existing fields in `SamplingParams`.)

### Example Custom Logits Processor Implementation

The contrived example below implements a custom logits processor which consumes a `(num\_requests) \times (vocab\_size)` logits tensor and masks out all tokens except for one (`target_token`) with `float(-inf)`. The logits processor is disabled for any request that does not specify `target_token`. To determine whether the logits processor is enabled and which token to leave unmasked, the logits processor checks `SamplingParams.extra_args` for a `target_token` custom argument associated with each request:

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
                else: 
                    self.req_info.pop(index, None)

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
            cols = torch.tensor(
                list(self.req_info.values()), dtype=torch.long, device=logits.device
            )
            rows = torch.tensor(
                list(self.req_info.keys()), dtype=torch.long, device=logits.device
            )
            values_to_keep = logits[rows, cols].clone()

            # Mask all but target tokens
            logits[rows] = float('-inf')
            logits[rows, cols] = values_to_keep

            return logits
    ```

In the rest of this document, we will use `DummyLogitsProcessor` as an example of a custom logits processor.

The `DummyLogitsProcessor.update_state()` implementation maintains a "sparse" representation of the batched requests in the `self.req_info` dictionary: only those requests which specify a `target_token` value have a key in the dictionary. `update_state()` adjusts the stored request indices and `target_token` values (keys and values respectively in `self.req_info`) in response to Add, Remove and Move operations against the persistent batch.

### Wrapping an Existing Request-Level Logits Processor

Although the vLLM engine applies logits processors at batch granularity, some users may want to use vLLM with a "request-level" logits processor implementation - an implementation which operates on individual requests. This will be especially true if your logits processor was developed for vLLM version 0, which required it to be a `Callable` (as described [here](https://docs.vllm.ai/en/v0.10.1.1/api/vllm/logits_process.html)) conforming to the following type annotation:

``` python
RequestLogitsProcessor = Union[

    # (output token ids, logits tensor) -> logits tensor
    Callable[[list[int], Tensor], Tensor],

    # (prompt token ids, output token ids, logits tensor) -> logits tensor
    Callable[[list[int], list[int], Tensor], Tensor],
]
```

While request-level logits processors are explicitly *not* supported in the vLLM engine, vLLM *does* provide a convenient process to wrap an existing `Callable` request-level logits processor and create a batch-level logits processor that is compatible with vLLM. The `Callable` must conform to the type annotation above; if your request-level logits processor has a different interface, then in order to wrap it, you may need to modify it or implement an additional wrapper layer to comply with the interface specification above.

You can wrap the request-level logits processor by subclassing `AdapterLogitsProcessor` as shown in the example below (in this example, `DummyPerReqLogitsProcessor` is a stand-in for your request-level logits processor which needs to be wrapped.) Override `AdapterLogitsProcessor.is_argmax_invariant(self)` to accurately reflect whether your request-level logits processor may impact which token has the highest-value logit. Override `AdapterLogitsProcessor.new_req_logits_processor(self,params)` to create a new request-level logits processor instance from a `SamplingParams` instance:

??? code "Example of Wrapping a Request-Level Logits Processor"

    ``` python
    ...

    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor, # Wrapper base-class
        RequestLogitsProcessor, # Request-level logitsproc type annotation
    )

    ...

    # Stand-in for your request-level logits processor:
    class DummyPerReqLogitsProcessor:
        """The request-level logits processor masks out all logits except the
        token id identified by `target_token`"""

        def __init__(self, target_token: int) -> None:
            """Specify `target_token`"""
            self.target_token = target_token

        def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
        ) -> torch.Tensor:
            val_to_keep = logits[self.target_token].item()
            logits[:] = float("-inf")
            logits[self.target_token] = val_to_keep
            return logits

    ...

    # Example of wrapping the request-level logits processor:
    class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
        """Example of wrapping a fake request-level logit processor to create a
        batch-level logits processor"""

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(
            self,
            params: SamplingParams,
        ) -> Optional[RequestLogitsProcessor]:
            """This method returns a new request-level logits processor, customized
            to the `target_token` value associated with a particular request.

            Returns None if the logits processor should not be applied to the
            particular request. To use the logits processor the request must have
            a "target_token" custom argument with an integer value.

            Args:
            params: per-request sampling params

            Returns:
            `Callable` request logits processor, or None
            """
            target_token: Optional[Any] = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is None:
                return None
            if not isinstance(target_token, int):
                logger.warning(
                    "target_token value %s is not int; not applying logits"
                    " processor to request.",
                    target_token,
                )
                return None
            return DummyPerReqLogitsProcessor(target_token)
    ```

!!! note
    Your `new_req_logits_processor()` override can return `None` to signal that the wrapped logits processor should not be applied to the request in question.

Once you have created a custom subclass (like `WrappedPerReqLogitsProcessor`) which wraps your request level logits processor, you can pass the custom subclass to vLLM via any of the methods described in the following section.

## Ways to Load Your Custom Logits Processor in vLLM

Logits processors are loaded at initialization. Critically, the set of loaded logits processors cannot be modified after the vLLM engine finishes loading, and new logits logits processors cannot be loaded on-demand for individual requests.

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
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "Passing custom logits processor FQCN to `AsyncLLM` in Python"

    ``` python
    # Pass in FQCN
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=["your.module.path:DummyLogitsProcessor"])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

??? code "Passing custom logits processor FQCN to vLLM server via CLI"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

### Method 2: Automatically Detect Custom Logits Processors Installed in Your Python Environment As Entry Points

[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) can enable installed packages to make themselves available as plugins to other Python programs, via pieces of metadata known as "entry points".

During initialization, vLLM automatically scans the `vllm.logits_processors` entry point group and loads any installed logits processors which it finds.

Suppose that you have developed a Python package that holds your custom logits processors. You can expose each logits processor to vLLM by adding a unique entrypoint for each logits processor to your logits processor Python package. The example below shows how to add an entrypoint to your project's `pyproject.toml` file:

??? code "Exposing a custom logits processor as a Python entrypoint"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to the `LLM` or `AsyncLLM` constructors or to the vLLM server explicitly at initialization time if your logits processor is exposed as an entry point.

!!! note
    vLLM will *always* load *all* logits processors which are exposed via entrypoints under the `vllm.logits_processors` grouping.

### Method 3 (Offline-only): Pass a Python Class Object to the vLLM Constructor

You can pass one or more custom logits processor class objects to the `LLM` and `AsyncLLM` constructors. This option is very flexible, as the logits processor classes may either be (1) defined locally within the same Python source file where `LLM` or `AsyncLLM` is instantiated, or (2) imported from a Python package.

??? code "Passing custom logits processor class object to `LLM` or `AsyncLLM` in Python"

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

    # Pass class object to AsyncLLM constructor
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=[DummyLogitsProcessor])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

## Invoking a Custom Logits Processor Against a Request

The design of the custom logits processor determines whether the logits processor must be enabled/disabled for a given request, and what arguments must be provided to configure the logits processor.

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

Once vLLM loads a logits processor during initialization, then vLLM will invoke `update_state()` and `apply()` against that logits processor in every engine step. Both methods operate on all requests which currently reside in the vLLM persistent batch. Thus it is important to implement these methods efficiently.

* Write efficient `apply()` and `update_state()` implementations in light of the fact that logits processors operate at batch granularity
    * For example, you may be able to use efficient vectorized operations to implement `apply()` or update internal state vectors in `update_state()`
    * However, if you think that a logits processor may be used infrequently, it may be appropriate to use a "sparse" representation of request state i.e. the class can represent request configuration using a dictionary which only stores metadata about requests that enable the logits processor
    * **Note:** wrapped request-level logits processors do not need to implement `apply()` and `update_state()`; the default `AdapterLogitsProcessor.update_state()` implementation maintains a sparse representation of request state, wherein requests for which `new_req_logits_processor()` returns `None` are not represented in the base-class state dictionary. The default implementation of `AdapterLogitsProcessor.apply()` applies the request-level logits processor to each row of input logits sequentially and assembles the output logits tensor. If the performance of this `AdapterLogitsProcessor` default implementation is insufficient, then avoid wrapping your request-level logits processor and instead re-implement it as a `LogitsProcessor` subclass with optimized `apply()` and `update_state()` implementations that operate at batch granularity

* It is up to the logits processor author to determine:

    1. **The per-request attributes which configure the logits processor's behavior against that request.** Your custom logits processor's `update_state()` override determines how `SamplingParams` fields are mapped into logits processor state

        * **Note:** for wrapped request-level logits processors, `new_req_logits_processor()` determines how `SamplingParams` fields are used to initialize a request-level logits processor instance.

    2. **The conditions under which the logits processor is or is not enabled on a per-request basis.** Unless your intention is for the custom logits processor to act on all requests all the time, you should write your logits processor in such a way that it is possible to disable the logits processor for a given request, i.e. by defaulting an argument to `None` or by passing in a specific do-nothing argument value i.e. `0.0`. Try to save compute and memory for requests which disable the logits processor

        * **Note:** for wrapped per-request logits processors, the default `AdapterLogitsProcessor.update_state()` implementation ensures that the request-level logits processor is disabled when `new_req_logits_processor()` returns `None` for that request

    3. **The conditions under which the logits processor is short-circuited at the batch level.** Even if you have defined a way to disable the custom logits processor at the request level, it may be difficult to translate this into compute savings i.e. if your `update_state()` and `apply()` implementations use efficient vectorized implementations that operate on the whole persistent batch in a single command. For example, you cannot skip an entire vectorized operation in `apply()` just because one request disabled the logits processor. To save compute in the edge-case where no running requests utilize the custom logits processor, we recommend designing `apply()` to return the unmodified input tensor if all requests have the logits processor disabled. Similarly, consider whether steps can be skipped in `update_state()` if no requests enable the logits processor

        * Additionally, an easy way to save compute in `update_state()` is to exit early when the `batch_update` is `None`

        * **Note:** for wrapped per-request logits processors, the `AdapterLogitsProcessor` base-class implements the above optimizations by default

* Ensure that the logits processor `update_state` method discards information about finished requests (i.e. requests which are replaced by an Add or which are subject to a Remove)

    * **Note:** for wrapped per-request logits processors, the `AdapterLogitsProcessor` base-class handles this by default

* `is_argmax_invariant()` can be hard-coded to `True` or `False` if the logits processor has consistent behavior. However the argmax invariance may also be determined programmatically (i.e. if your logits processor is user-customizable in some way that impacts whether the logits processor is argmax invariant). For this reason, `is_argmax_invariant()` is not a class method
