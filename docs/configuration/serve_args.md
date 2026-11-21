# Server Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.

## CLI Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.
To see the available options, take a look at the [CLI Reference](../cli/README.md)!

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

Lists become repeated values, and nested mappings are passed as JSON, so
JSON-valued arguments such as `--lora-modules` can be written in YAML:

```yaml
lora-modules:
  - sql-lora=jeeejeee/llama32-3b-text2sql-spider
  - name: sql-lora-2
    path: jeeejeee/llama32-3b-text2sql-spider
    base_model_name: meta-llama/Llama-3.2-3B-Instruct
```

### Generate a configuration from vLLM Recipes

[vLLM Recipes](https://recipes.vllm.ai/) can be converted into `config.yaml`
and `env.sh`. See the
[Recipes conversion tool README](../../tools/recipes/README.md) for usage.

Source the generated environment before starting vLLM:

```bash
source env.sh
vllm serve --config config.yaml
```

!!! note
    In case an argument is supplied simultaneously using command line and the config file, the value from the command line will take precedence.
    The order of priorities is `command line > config file values > defaults`.
    e.g. `vllm serve SOME_MODEL --config config.yaml`, SOME_MODEL takes precedence over `model` in config file.
