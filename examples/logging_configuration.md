# Logging Configuration

vLLM leverages Python's `logging.config.dictConfig` functionality to enable
robust and flexible configuration of the various loggers used by vLLM.

vLLM offers two environment variables that can be used, individually or in
combination, to accommodate a range of logging configurations that vary in
flexibility from:

- vLLM's default logging configuration (least flexible)
- coarse grained custom logging configuration
- fine-grained custom logging configuration (most flexible)


## Logging Configuration Environment Variables

### `VLLM_CONFIGURE_LOGGING`

`VLLM_CONFIGURE_LOGGING` controls whether or not vLLM propagates configuration
from the root vLLM logger ("vllm") to other loggers used by vLLM. This
functionality is enabled by default, but can be disabled by setting
`VLLM_CONFIGURE_LOGGING=0` when running vLLM.

If `VLLM_CONFIGURE_LOGGING` is enabled and no value is given for
`VLLM_LOGGING_CONFIG_PATH`, vLLM will use built-in default configuration to
configure the root vLLM logger and will use that same configuration when
configuring other loggers used by vLLM.

### `VLLM_LOGGING_CONFIG_PATH`

`VLLM_LOGGING_CONFIG_PATH` allows users to specify a path to a JSON file of
alternative, custom logging configuration that will be used instead of vLLM's
default logging configuration. The logging configuration should be provided in
JSON format following the schema specified by Python's [logging configuration
dictionary schema](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

If `VLLM_LOGGING_CONFIG_PATH` is specified, but `VLLM_CONFIGURE_LOGGING` is
disabled, vLLM will apply the given logging configuration, but will take no
additional actions to propagate configuration from the root vLLM logger to
other vLLM loggers.

If `VLLM_LOGGING_CONFIG_PATH` is specified and `VLLM_CONFIGURE_LOGGING` is
enabled, vLLm will only configure loggers that were not created when applying
`logging.config.dictConfig` (or by other means) to ensure that custom
configuration of vLLM loggers is not overwritten.


## Examples

### Example 1: Customize vLLM root logger and propagate to other vLLM loggers

For this example, we will customize the vLLM root logger to use
[`python-json-logger`](https://github.com/madzak/python-json-logger) to log to
STDOUT of the console in JSON format with a log level of `INFO`.

Because `VLLM_CONFIGURE_LOGGING` is enabled by default, the configuration we
apply to the vLLM root logger will be used when configuring other vLLM loggers,
effectively making it so all vLLM loggers log to STDOUT of the console in JSON
format.

To begin, first, create an appropriate JSON logging configuration file:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "console": {
      "class" : "logging.StreamHandler",
      "formatter": "json",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["console"],
      "level": "INFO",
      "propagate": false
    }
  },
  "version": 1
}
```

Next, install the `python-json-logger` package if it's not already installed:

```bash
pip install python-json-logger
```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    python3 -m vllm.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


### Example 2: Silence a particular vLLM logger

To silence a particular vLLM logger, it is necessary to provide custom logging
configuration for the target logger that configures the logger to have no
handlers. When custom configuration is provided for any logger, it is also
necessary to provide configuration for the root vLLM logger since any custom
logger configuration overrides the default logging configuration behavior of
vLLM.

First, create an appropriate JSON logging configuration file that includes
configuration for the root vLLM logger and for the logger you wish to silence:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "vllm": {
      "class": "vllm.logging.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "vllm": {
      "class" : "logging.StreamHandler",
      "formatter": "vllm",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["vllm"],
      "level": "DEBUG",
      "propagage": false
    },
    "vllm.example_noisy_logger": {
      "handlers": []
    }
  },
  "version": 1
}
```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    python3 -m vllm.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


### Example 3: Configure root vLLM logger without configuring other vLLM loggers

This example is very similar to example 2, except it sets
`VLLM_CONFIGURE_LOGGING=0` to prevent vLLM from copying logging configuration
from the root vLLM logger to other vLLM loggers.

First, create an appropriate JSON logging configuration file that includes
configuration for the root vLLM logger and any other loggers you wish to
configure:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "vllm": {
      "class": "vllm.logging.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "vllm": {
      "class" : "logging.StreamHandler",
      "formatter": "vllm",
      "level": "WARN",
      "stream": "ext://sys.stderr"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["vllm"],
      "level": "WARN",
      "propagage": false
    }
  },
  "version": 1
}
```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file and the
`VLLM_CONFIGURE_LOGGING` environment variable set to `0`:

```bash
VLLM_CONFIGURE_LOGGING=0 \
    VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    python3 -m vllm.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


### Example 4: Disable vLLM default logging configuration

To disable vLLM's default configuration of loggers, simple set
`VLLM_CONFIGURE_LOGGING=0` when running vLLM. This will prevent vLLM for
configuring the root vLLM logger and from taking any configuration actions on
other vLLM loggers.

```bash
VLLM_CONFIGURE_LOGGING=0 \
    python3 -m vllm.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


## Additional resources

- [`logging.config` Dictionary Schema Details](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
