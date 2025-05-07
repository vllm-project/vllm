(engine-args)=

# Engine Arguments

Engine arguments control the behavior of the vLLM engine.

- For [offline inference](#offline-inference), they are part of the arguments to `LLM` class.
- For [online serving](#openai-compatible-server), they are part of the arguments to `vllm serve`.

Below, you can find an explanation of every engine argument:

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
    :markdownhelp:
```

## Async Engine Arguments

Additional arguments are available to the asynchronous engine which is used for online serving:

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _async_engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
    :markdownhelp:
```
