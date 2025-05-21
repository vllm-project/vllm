(serve-args)=

# Server Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.

## CLI Arguments

The following are all arguments available from the `vllm serve` command:

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.entrypoints.openai.cli_args
    :func: create_parser_for_docs
    :prog: vllm serve
    :nodefaultconst:
    :markdownhelp:
```

## Configuration file

You can load CLI arguments via a [YAML](https://yaml.org/) config file.
The argument names must be the long form of those outlined [above](#serve-args).

For example:

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

To use the above config file:

```bash
vllm serve --config config.yaml
```

:::{note}
In case an argument is supplied simultaneously using command line and the config file, the value from the command line will take precedence.
The order of priorities is `command line > config file values > defaults`.
e.g. `vllm serve SOME_MODEL --config config.yaml`, SOME_MODEL takes precedence over `model` in config file.
:::
