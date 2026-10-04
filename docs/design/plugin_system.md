# Plugin System

The community frequently requests the ability to extend vLLM with custom features. To facilitate this, vLLM includes a plugin system that allows users to add custom features without modifying the vLLM codebase. This document explains how plugins work in vLLM and how to create a plugin for vLLM.

## How Plugins Work in vLLM

Plugins are user-registered code that vLLM executes. Given vLLM's architecture (see [Arch Overview](arch_overview.md)), multiple processes may be involved, especially when using distributed inference with various parallelism techniques. To enable plugins successfully, every process created by vLLM needs to load the plugin. This is done by the [load_plugins_by_group][vllm.plugins.load_plugins_by_group] function in the `vllm.plugins` module.

## How vLLM Discovers Plugins

vLLM's plugin system uses the standard Python `entry_points` mechanism. This mechanism allows developers to register functions in their Python packages for use by other packages. An example of a plugin:

??? code

    ```python
    # inside `setup.py` file
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })

    # inside `vllm_add_dummy_model/__init__.py` file
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

For more information on adding entry points to your package, please check the [official documentation](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).

Every plugin has three parts:

1. **Plugin group**: The name of the entry point group. vLLM uses the entry point group `vllm.general_plugins` to register general plugins. This is the key of `entry_points` in the `setup.py` file. Always use `vllm.general_plugins` for vLLM's general plugins.
2. **Plugin name**: The name of the plugin. This is the value in the dictionary of the `entry_points` dictionary. In the example above, the plugin name is `register_dummy_model`. Plugins can be filtered by their names using the `VLLM_PLUGINS` environment variable. To load only a specific plugin, set `VLLM_PLUGINS` to the plugin name.
3. **Plugin value**: The fully qualified name of the function or module to register in the plugin system. In the example above, the plugin value is `vllm_add_dummy_model:register`, which refers to a function named `register` in the `vllm_add_dummy_model` module.

## Types of supported plugins

- **General plugins** (with group name `vllm.general_plugins`): The primary use case for these plugins is to register custom, out-of-the-tree models into vLLM. This is done by calling `ModelRegistry.register_model` to register the model inside the plugin function. For an example of an official model plugin, see the [bart-plugin](https://github.com/vllm-project/bart-plugin) which adds support for `BartForConditionalGeneration`.

- **Platform plugins** (with group name `vllm.platform_plugins`): The primary use case for these plugins is to register custom, out-of-the-tree platforms into vLLM. The plugin function should return `None` when the platform is not supported in the current environment, or the platform class's fully qualified name when the platform is supported.

- **IO Processor plugins** (with group name `vllm.io_processor_plugins`): The primary use case for these plugins is to register custom pre-/post-processing of the model prompt and model output for pooling models. The plugin function returns the IOProcessor's class fully qualified name.

- **Stat logger plugins** (with group name `vllm.stat_logger_plugins`): The primary use case for these plugins is to register custom, out-of-the-tree loggers into vLLM. The entry point should be a class that subclasses StatLoggerBase.

- **Endpoint plugins** (with group name `vllm.endpoint_plugins`): The primary use case for these plugins is to register custom, out-of-the-tree HTTP routes on the OpenAI compatible API server. Unlike the other plugin groups above, endpoint plugins are loaded only in the API server front end process and are **not loaded by default**. See [Endpoint Plugins](endpoint_plugins.md) for the interface and [Security](../usage/security.md#endpoint-plugins) for the opt-in and trust model.

## Model metadata providers

A general plugin can register a `MetadataProvider` from `vllm.plugins.model_metadata` to make non-weight files available before vLLM reads them. This includes model configs, tokenizer files, processor configs, and remote code. Providers must not download weights or rewrite model and tokenizer paths; native resolution and model loaders keep their existing responsibilities.

There are two phases at three call sites, all immediately after `load_general_plugins()`:

| Call site | Phase | Timing |
| --- | --- | --- |
| `EngineArgs.__post_init__` | `prepare_source(source)` | Before offline `get_model_path()` resolution |
| `EngineCore.__init__` | `prepare_consumer(source, model=..., tokenizer=...)` | Before executor construction |
| `WorkerWrapperBase.init_worker` | `prepare_consumer(source, model=..., tokenizer=...)` | Before worker class resolution and construction, and thus before `init_device()` |

`MetadataSource` is a frozen dataclass containing the model and tokenizer values passed to an `EngineArgs` instance, their requested revisions, `code_revision`, the Hugging Face cache root, and the offline and ModelScope settings. The cache root is `huggingface_hub.constants.HF_HUB_CACHE` at source preparation time, not the weight loader's download directory. Unspecified tokenizer and revision values remain `None`.

The inputs are not necessarily repo IDs: `LLM.from_engine_args()` and the default `EngineArgs` instances used for argument logging can receive already-resolved local paths. `LLM.from_engine_args()` retains the original source only if the model, tokenizer, and revisions are unchanged since `EngineArgs` construction. Providers must make `prepare_source` a no-op for local paths and already-prepared inputs, including repeated calls.

The source phase may populate the cache, but vLLM still resolves the model normally afterward. vLLM retains the original source in `ModelConfig.metadata_source` and serializes it with the config to other processes. This field does not affect the computation graph hash; the provider itself is never stored in the config.

`prepare_consumer` receives the source together with `ModelConfig.model` and `ModelConfig.tokenizer`. For Hub inputs, offline mode passes local snapshot paths, while online mode retains repo IDs. Prepare any resolved snapshot path at that exact location on each process's filesystem, using the commit in the path rather than re-resolving a branch that might have moved.

Providers should first check whether the required metadata is already available locally and return immediately if it is. This avoids failing Core or Worker initialization when frontend preparation succeeded but the metadata service is now unreachable. Providers must handle repeated calls safely and return without work for sources they do not support. Without a registered provider, both phases are no-ops and the source field remains `None`.

Failure handling differs by phase:

- A source exception is logged as a warning with the model and revision. Native resolution continues, preserving its usual diagnostics.
- A consumer exception stops initialization. `MetadataUnavailable` propagates unchanged; other exceptions are wrapped in `MetadataUnavailable` with their original cause. vLLM does not probe native metadata or attempt fallback after a consumer failure.

For example, a plugin can expose the following `register` function through its `vllm.general_plugins` entry point. The plugin-defined helpers implement the no-op and local-readiness checks above and must only prepare metadata:

```python
from vllm.plugins.model_metadata import (
    MetadataSource,
    register_model_metadata_provider,
)

from .cache import prepare_resolved_files, prepare_source_files


class CacheMetadataProvider:
    def prepare_source(self, source: MetadataSource) -> None:
        prepare_source_files(source)

    def prepare_consumer(
        self, source: MetadataSource, *, model: str, tokenizer: str
    ) -> None:
        prepare_resolved_files(source, model=model, tokenizer=tokenizer)


_provider = CacheMetadataProvider()


def register() -> None:
    register_model_metadata_provider(_provider)
```

Only one provider can be registered per process. Re-registering the same object is a no-op; registering a different object raises `RuntimeError`. If two plugins each register their own provider, the second registration raises during `load_general_plugins()` and startup fails. Both preparation methods must be callable, otherwise registration raises `TypeError`. The registry is process-local: an EngineCore or worker started with `spawn` begins with no provider and relies on its own `load_general_plugins()` call to register one again.

Direct `ModelConfig(...)` construction that bypasses `EngineArgs` does not run the source phase. Neither the source nor the consumer phase covers:

- Independent repositories selected with `--hf-config-path`.
- Speculative draft models.
- Speculators-format checkpoints, where `create_engine_config()` replaces the model and tokenizer with the verifier model after the source phase.

This interface does not change the Run:AI path, remote-code loading rules, or weight loading.

## Guidelines for Writing Plugins

- **Being re-entrant**: The function specified in the entry point should be re-entrant, meaning it can be called multiple times without causing issues. This is necessary because the function might be called multiple times in some processes.

### Platform plugins guidelines

1. Create a platform plugin project, for example, `vllm_add_dummy_platform`. The project structure should look like this:

    ```shell
    vllm_add_dummy_platform/
    ├── vllm_add_dummy_platform/
    │   ├── __init__.py
    │   ├── my_dummy_platform.py
    │   ├── my_dummy_worker.py
    │   ├── my_dummy_attention.py
    │   ├── my_dummy_device_communicator.py
    │   ├── my_dummy_custom_ops.py
    ├── setup.py
    ```

2. In the `setup.py` file, add the following entry point:

    ```python
    setup(
        name="vllm_add_dummy_platform",
        ...
        entry_points={
            "vllm.platform_plugins": [
                "my_dummy_platform = vllm_add_dummy_platform:register"
            ]
        },
        ...
    )
    ```

    Please make sure `vllm_add_dummy_platform:register` is a callable function and returns the platform class's fully qualified name. for example:

    ```python
    def register():
        return "vllm_add_dummy_platform.my_dummy_platform.MyDummyPlatform"
    ```

3. Implement the platform class `MyDummyPlatform` in `my_dummy_platform.py`. The platform class should inherit from `vllm.platforms.interface.Platform`. Please follow the interface to implement the functions one by one. There are some important functions and properties that should be implemented at least:

    - `_enum`: This property is the device enumeration from [PlatformEnum][vllm.platforms.interface.PlatformEnum]. Usually, it should be `PlatformEnum.OOT`, which means the platform is out-of-tree.
    - `device_type`: This property should return the type of the device which pytorch uses. For example, `"cpu"`, `"cuda"`, etc.
    - `device_name`: This property is set the same as `device_type` usually. It's mainly used for logging purposes.
    - `check_and_update_config`: This function is called very early in the vLLM's initialization process. It's used for plugins to update the vllm configuration. For example, the block size, graph mode config, etc., can be updated in this function. The most important thing is that the **worker_cls** should be set in this function to let vLLM know which worker class to use for the worker process.
    - `get_attn_backend_cls`: This function should return the attention backend class's fully qualified name.
    - `get_device_communicator_cls`: This function should return the device communicator class's fully qualified name.

4. Implement the worker class `MyDummyWorker` in `my_dummy_worker.py`. The worker class should inherit from [WorkerBase][vllm.v1.worker.worker_base.WorkerBase]. Please follow the interface to implement the functions one by one. Basically, all interfaces in the base class should be implemented, since they are called here and there in vLLM. To make sure a model can be executed, the basic functions should be implemented are:

    - `init_device`: This function is called to set up the device for the worker.
    - `initialize_cache`: This function is called to set cache config for the worker.
    - `load_model`: This function is called to load the model weights to device.
    - `get_kv_cache_spec`: This function is called to generate the kv cache spec for the model.
    - `determine_available_memory`: This function is called to profiles the peak memory usage of the model to determine how much memory can be used for KV cache without OOMs.
    - `initialize_from_config`: This function is called to allocate device KV cache with the specified kv_cache_config
    - `execute_model`: This function is called every step to inference the model.

    Additional functions that can be implemented are:

    - If the plugin wants to support sleep mode feature, please implement the `sleep` and `wakeup` functions.
    - If the plugin wants to support graph mode feature, please implement the `compile_or_warm_up_model` function.
    - If the plugin wants to support speculative decoding feature, please implement the `take_draft_token_ids` function.
    - If the plugin wants to support lora feature, please implement the `add_lora`,`remove_lora`,`list_loras` and `pin_lora` functions.
    - If the plugin wants to support data parallelism feature, please implement the `execute_dummy_batch` functions.

    Please look at the worker base class [WorkerBase][vllm.v1.worker.worker_base.WorkerBase] for more functions that can be implemented.

5. Implement the attention backend class `MyDummyAttention` in `my_dummy_attention.py`. The attention backend class should inherit from [AttentionBackend][vllm.v1.attention.backend.AttentionBackend]. It's used to calculate attentions with your device. Take `vllm.v1.attention.backends` as examples, it contains many attention backend implementations.

6. Implement custom ops for high performance. Most ops can be run by pytorch native implementation, while the performance may not be good. In this case, you can implement specific custom ops for your plugins. Currently, there are kinds of custom ops vLLM supports:

    - pytorch ops
      there are 3 kinds of pytorch ops:

        - `communicator ops`: Device communicator op. Such as all-reduce, all-gather, etc.
          Please implement the device communicator class `MyDummyDeviceCommunicator` in `my_dummy_device_communicator.py`. The device communicator class should inherit from [DeviceCommunicatorBase][vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase].
        - `common ops`: Common ops. Such as matmul, softmax, etc.
          Please implement the common ops by register oot way. See more detail in [CustomOp][vllm.model_executor.custom_op.CustomOp] class.
        - `csrc ops`: C++ ops. This kind of ops are implemented in C++ and are registered as torch custom ops.
          Following csrc module and `vllm._custom_ops` to implement your ops.

    - triton ops
      Custom way doesn't work for triton ops now.

7. (optional) Implement other pluggable modules, such as lora, graph backend, quantization, mamba attention backend, etc.

## Compatibility Guarantee

vLLM guarantees the interface of documented plugins, such as `ModelRegistry.register_model`, will always be available for plugins to register models. However, it is the responsibility of plugin developers to ensure their plugins are compatible with the version of vLLM they are targeting. For example, `"vllm_add_dummy_model.my_llava:MyLlava"` should be compatible with the version of vLLM that the plugin targets.

The interface for the model/module may change during vLLM's development. If you see any deprecation log info, please upgrade your plugin to the latest version.

## Deprecation announcement

!!! warning "Deprecations"
    - `use_v1` parameter in `Platform.get_attn_backend_cls` is deprecated. It has been removed in v0.13.0.
    - `_Backend` in `vllm.attention` is deprecated. It has been removed in v0.13.0. Please use `vllm.v1.attention.backends.registry.register_backend` to add new attention backend to `AttentionBackendEnum` instead.
    - `seed_everything` platform interface is deprecated. It has been removed in v0.16.0. Please use `vllm.utils.torch_utils.set_random_seed` instead.
    - `prompt` in `Platform.validate_request` is deprecated. It has been removed in v0.18.0.
