(engine-args)=

# Engine Arguments

Below, you can find an explanation of every engine argument for vLLM organized by category:

## Model Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Model Arguments
    :nodefaultconst:
```

## KV Cache Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: KV Cache Arguments
    :nodefaultconst:
```

## Parallel Processing Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Parallel Processing Arguments
    :nodefaultconst:
```

## Scheduler Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Scheduler Arguments
    :nodefaultconst:
```

## Quantization Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Quantization Arguments
    :nodefaultconst:
```

## LoRA Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: LoRA Arguments
    :nodefaultconst:
```

## Speculative Decoding Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Speculative Decoding Arguments
    :nodefaultconst:
```

## Observability Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Observability Arguments
    :nodefaultconst:
```

## System Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: System Arguments
    :nodefaultconst:
```

## Multimodal Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :group: Multimodal Arguments
    :nodefaultconst:
```

## Other Arguments

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :nogroup:
    :nodefaultconst:
```

## Async Engine Arguments

Below are the additional arguments related to the asynchronous engine:

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _async_engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
```
