(engine-args)=

# Engine Arguments

Below, you can find an explanation of every engine argument for vLLM:

<!--- pyml disable-num-lines 7 no-space-in-emphasis-->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
```

## Async Engine Arguments

Below are the additional arguments related to the asynchronous engine:

<!--- pyml disable-num-lines 7 no-space-in-emphasis-->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _async_engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
```
