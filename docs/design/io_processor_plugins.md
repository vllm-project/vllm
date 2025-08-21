# IO Processor Plugins

IO Processor plugins are a feature that allows pre and post processing of the model input and output for pooling models. The idea is that users are allowed to pass a custom input to vLLM that is conferted into one or more model prompts and fed to the model `encode` method. One potential use-case of such plugins is that of usnig vLLM for generating multi-modal data. Say users feed an image tovLLM and get an image in output.

When performing inference with plugins the prompt type is defined by the plugin and the same is valid for the model output. vLLM does not perform any validation of input/output data and it is up to the plugin to ensure the correct data is being fed to the model and returned to the user. As of now these plugins can be used only for pooling models and the are automatically applied to the `encode` methods in the `LLM` an `AsyncLLM` classes when invoking the `encode_with_io_processor_plugin` method, or in online serving mode via the `/plugin_pooling` endpoint.

## Writing a IO Processor Plugin

IO Processor plugins implement the `IOProcessor` interface (<gh-file:vllm/plugins/io_processors/interface.py>):

```python
class IOProcessor(ABC):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompt: Any,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: Any,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(self,
                     model_out: Sequence[Optional[PoolingRequestOutput]],
                     request_id: Optional[str] = None,
                     **kwargs) -> Any:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_out: Sequence[Optional[PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        return self.post_process(model_out, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def plugin_out_to_response(self,
                               plugin_out: Any) -> IOProcessorPluginResponse:
        raise NotImplementedError
```

The `parse` method is used for validating the user prompt and converting it into the input expected by the `pre_process`/`pre_process_async` methods.
The `pre_process*` methods take the validated plugin input to generate vLLM's model prompts for regular inference.
The `post_process*` methods take `PoolingRequestOutput` objects in input and generate a custom plugin output.
An implementation of the `encode_with_io_processor_plugin` method is available [here](../../vllm/entrypoints/llm.py) and [here](../../vllm/v1/engine/async_llm.py).

The `plugin_out_to_response` method is used only for online serving and converts the plugin output to the `IOProcessorPluginResponse` type that is then returned by the APIServer. The implementation of the `/plugin_pooling` serving endpoint is [here](../../vllm/entrypoints/openai/serving_pooling_with_io_plugin.py).

An example implementation of a plugin that enables generating geotiff images with the PrithviGeospatialMAE model is available [here](https://github.com/christian-pinto/prithvi_io_processor_plugin). Please, also refer to our [online](../../examples/online_serving/prithvi_geospatial_mae.py) and [offline](../../examples/offline_inference/prithvi_geospatial_mae_io_processor.py) inferences examples.

## Using a IO Processor plugin

IO Processor plugins are loaded at engine startup and there are two ways of specifying the name of the plugin to be loaded:

1. Via vLLM's `EngineArgs`: setting the `io_processor_plugin` argument in the `EngineArgs` used to initialie the `AsyncLLM`. The same can be achieved by passing the `io_processor_plugin` argument to `LLM` in offline mode, or by passing the `--io-processor-plugin` argument in serving mode.
2. Via the model HF configuration: adding a `io_processor_plugin` field to the model config (config.json).

The order also identifies the priority of the methods. e.g., setting the plugin name via `EngineArgs` will override any plugin name specified in the model HF config (config.json).
