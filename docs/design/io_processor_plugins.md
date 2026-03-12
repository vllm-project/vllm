# IO Processor Plugins

IO Processor plugins are a feature that allows pre- and post-processing of the model input and output. The primary use case is still pooling models, where users can pass a custom input to vLLM that is converted into one or more model prompts and fed to the model `encode` method. One potential use-case of such plugins is that of using vLLM for generating multi-modal data. Say users feed an image to vLLM and get an image in output.

When performing an inference with IO Processor plugins, the prompt type is defined by the plugin and the same is valid for the final request output. vLLM does not perform any validation of input/output data, and it is up to the plugin to ensure the correct data is being fed to the model and returned to the user. Pooling remains the most complete integration and can be triggered via the `encode` method in `LLM` and `AsyncLLM`, or in online serving mode via the `/pooling` endpoint. Generative models currently support a narrower one-to-one `DataPrompt` path through `LLM.generate` and `AsyncLLM.generate`, where a plugin can reshape a single request into a single engine prompt, merge sampling params, and post-process the final `RequestOutput`.

## Writing an IO Processor Plugin

IO Processor plugins implement the [`IOProcessor`][vllm.plugins.io_processors.interface.IOProcessor] interface:

```python
IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Abstract interface for pre/post-processing of engine I/O."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__()

        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_sampling_params_for_prompt(
        self,
        prompt: IOProcessorInput,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return self.merge_sampling_params(params)

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to vLLM.
        # Let's sort them by id before post_processing
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)

    def post_process_generate(
        self,
        model_output: RequestOutput,
        request_id: str | None = None,
        **kwargs,
    ) -> RequestOutput:
        return model_output
```

The `parse_data` method is used for validating the user data and converting it into the input expected by the `pre_process*` methods.
The `merge_sampling_params`, `merge_sampling_params_for_prompt`, and `merge_pooling_params` methods merge input `SamplingParams` or `PoolingParams` (if any) with the default one.
The `pre_process*` methods take the validated plugin input to generate vLLM's model prompts for regular inference.
The `post_process*` methods take `PoolingRequestOutput` objects as input and generate a custom plugin output.
The `post_process_generate*` methods can rewrite the final `RequestOutput` for generative models.

An example implementation of a plugin that enables generating geotiff images with the PrithviGeospatialMAE model is available [here](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation). Please, also refer to our online ([examples/pooling/plugin/prithvi_geospatial_mae_online.py](../../examples/pooling/plugin/prithvi_geospatial_mae_online.py)) and offline ([examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py](../../examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py)) inference examples.

## Using an IO Processor plugin

IO Processor plugins are loaded at engine startup and there are two methods for specifying the name of the plugin to be loaded:

1. Via vLLM's `EngineArgs`: setting the `io_processor_plugin` argument in the `EngineArgs` used to initialize the `AsyncLLM`. The same can be achieved by passing the `io_processor_plugin` argument to `LLM` in offline mode, or by passing the `--io-processor-plugin` argument in serving mode.
2. Via the model HF configuration: adding an `io_processor_plugin` field to the model config (config.json).

The order also determines method priority. i.e., setting the plugin name via `EngineArgs` will override any plugin name specified in the model HF config (config.json).
