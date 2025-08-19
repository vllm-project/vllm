# Custom Logits Processors

This document shows you how to augment vLLM with custom logits processors.

## Ways to Pass Your Custom Logits Processor to vLLM

### 1. Offline-only: pass a Python class object to the vLLM constructor

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

### 2. Pass the custom logits processor fully-qualified class name (FQCN) to vLLM at initialization time

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

### 3. Automatically detect installed custom logits processors in your Python environment via Python entry points

During initialization, vLLM automatically scans the `vllm.logits_processors` [entry point](https://setuptools.pypa.io/latest/userguide/entry_point.html) group and loads any installed logits processors which it finds.

Suppose that you have developed a Python package that holds your custom logits processors. You can expose each logits processor to vLLM by adding a unique entrypoint for each logits processor to your Python package; see example below:

??? code "Exposing a custom logits processor as a Python entrypoint"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to `logits_processors` at initialization time.

**Note:** vLLM will *always* load *all* logits processors which are exposed via entrypoints under `vllm.logits_processors`.

## Writing a vLLM Custom Logits Procesor

Custom logits processors must be subclasses of `vllm.v1.sample.logits_processor.LogitsProcessor`. The contrived example below implements a custom logits processor which masks out all tokens except for one (`target_token`) with `float(-inf)`. The logits processor is disabled for any request that does not specify `target_token`.

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

## Defining How the Custom Logits Processor Can Be Used

Once vLLM loads a logits processor during initialization, then vLLM will invoke `update_state()` and `apply()` against that logits processor in every engine step. Both methods operate on all requests which currently reside in the vLLM persistent batch. It is up to the logits processor author to determine:

1. **The per-request attributes which configure the logits processor's behavior against that request.** vLLM supports [custom arguments](./custom_arguments.md): the user may pass in a `dict` of custom request arguments, which will be accessible to all logits processors via `SamplingParams.extra_args`. If your logits processor requires arguments not already supported by `SamplingParams` and the vLLM REST API, we recommended designing your custom logits processor to look for these arguments as keys in the `SamplingParams.extra_args` dict. In the `DummyLogitsProcessor` example above, the logits processor looks for `target_tokens` as a custom argument.

2. **The conditions under which the logits processor is or is not enabled on a per-request basis.** Unless your intention is for the custom logits processor to act on all requests all the time, we recommended writing your logits processor in such a way that it is possible to disable the logits processor for a given request, i.e. through the absence of a particular custom argument or by passing in a specific argument value. In the `DummyLogitsProcessor` example above, the absence of `target_token` disables the logits processor for a given request.

3. **The conditions under which the logits processor is short-circuited at the batch level.** Even if you have defined a way to disable the custom logits processor at the request level, it may be difficult to translate this into compute savings i.e. if your `update_state()` and `apply()` implementations use efficient vectorized implementations that operate on the whole persistent batch in a single command. To save compute in the edge-case where no running requests utilize the custom logits processor, we recommend designing `update_state()` and `apply()` to exit early if all requests have the logits processor disabled. 

The examples below show how a user would pass a custom argument (`target_token`) to `DummyLogitsProcessor` in order to (1) enable the logits processor for that particular request and (2) control the logits processor's behavior.

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
