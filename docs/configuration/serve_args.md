# Server Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.

## CLI Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.
To see the available options, take a look at the [CLI Reference](../cli/README.md#options)!

## Configuration file

You can load CLI arguments via a [YAML](https://yaml.org/) config file.
The argument names must be the long form of those outlined [above](serve_args.md).

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

!!! note
    In case an argument is supplied simultaneously using command line and the config file, the value from the command line will take precedence.
    The order of priorities is `command line > config file values > defaults`.
    e.g. `vllm serve SOME_MODEL --config config.yaml`, SOME_MODEL takes precedence over `model` in config file.
