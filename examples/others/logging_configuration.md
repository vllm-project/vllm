# Logging Configuration

vLLM leverages Python's `logging.config.dictConfig` functionality to enable
robust and flexible configuration of the various loggers used by vLLM.

vLLM offers two environment variables that can be used to accommodate a range
of logging configurations that range from simple-and-inflexible to
more-complex-and-more-flexible.

- No vLLM logging (simple and inflexible)
    - Set `VLLM_CONFIGURE_LOGGING=0` (leaving `VLLM_LOGGING_CONFIG_PATH` unset)
- vLLM's default logging configuration (simple and inflexible)
    - Leave `VLLM_CONFIGURE_LOGGING` unset or set `VLLM_CONFIGURE_LOGGING=1`
- Fine-grained custom logging configuration (more complex, more flexible)
    - Leave `VLLM_CONFIGURE_LOGGING` unset or set `VLLM_CONFIGURE_LOGGING=1` and
    set `VLLM_LOGGING_CONFIG_PATH=<path-to-logging-config.json>`

## Logging Configuration Environment Variables

### `VLLM_CONFIGURE_LOGGING`

`VLLM_CONFIGURE_LOGGING` controls whether or not vLLM takes any action to
configure the loggers used by vLLM. This functionality is enabled by default,
but can be disabled by setting `VLLM_CONFIGURE_LOGGING=0` when running vLLM.

If `VLLM_CONFIGURE_LOGGING` is enabled and no value is given for
`VLLM_LOGGING_CONFIG_PATH`, vLLM will use built-in default configuration to
configure the root vLLM logger. By default, no other vLLM loggers are
configured and, as such, all vLLM loggers defer to the root vLLM logger to make
all logging decisions.

If `VLLM_CONFIGURE_LOGGING` is disabled and a value is given for
`VLLM_LOGGING_CONFIG_PATH`, an error will occur while starting vLLM.

### `VLLM_LOGGING_CONFIG_PATH`

`VLLM_LOGGING_CONFIG_PATH` allows users to specify a path to a JSON file of
alternative, custom logging configuration that will be used instead of vLLM's
built-in default logging configuration. The logging configuration should be
provided in JSON format following the schema specified by Python's [logging
configuration dictionary
schema](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

If `VLLM_LOGGING_CONFIG_PATH` is specified, but `VLLM_CONFIGURE_LOGGING` is
disabled, an error will occur while starting vLLM.

## Examples

### Example 1: Customize vLLM root logger

For this example, we will customize the vLLM root logger to use
[`python-json-logger`](https://github.com/nhairs/python-json-logger)
(which is part of the container image) to log to
STDOUT of the console in JSON format with a log level of `INFO`.

To begin, first, create an appropriate JSON logging configuration file:

??? note "/path/to/logging_config.json"

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

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 2: Silence a particular vLLM logger

To silence a particular vLLM logger, it is necessary to provide custom logging
configuration for the target logger that configures the logger so that it won't
propagate its log messages to the root vLLM logger.

When custom configuration is provided for any logger, it is also necessary to
provide configuration for the root vLLM logger since any custom logger
configuration overrides the built-in default logging configuration used by vLLM.

First, create an appropriate JSON logging configuration file that includes
configuration for the root vLLM logger and for the logger you wish to silence:

??? note "/path/to/logging_config.json"

    ```json
    {
      "formatters": {
        "vllm": {
          "class": "vllm.logging_utils.NewLineFormatter",
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
          "propagate": false
        },
        "vllm.example_noisy_logger": {
          "propagate": false
        }
      },
      "version": 1
    }
    ```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 3: Disable vLLM default logging configuration

To disable vLLM's default logging configuration and silence all vLLM loggers,
simple set `VLLM_CONFIGURE_LOGGING=0` when running vLLM. This will prevent vLLM
for configuring the root vLLM logger, which in turn, silences all other vLLM
loggers.

```bash
VLLM_CONFIGURE_LOGGING=0 \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 4: Disable access logs for health check endpoints

In production environments, health check endpoints like `/health`, `/metrics`,
and `/ping` are frequently called by load balancers and monitoring systems,
generating a large volume of repetitive access logs. To reduce log noise while
keeping logs for other endpoints, use the `--disable-access-log-for-endpoints`
option.

**Disable access logs for health and metrics endpoints:**

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048 \
    --disable-access-log-for-endpoints /health,/metrics,/ping
```

**Common endpoints to consider filtering:**

| Endpoint   | Description            | Typical Caller                                       |
| ---------- | ---------------------- | ---------------------------------------------------- |
| `/health`  | Health check           | Kubernetes liveness/readiness probes, load balancers |
| `/metrics` | Prometheus metrics     | Prometheus scraper (every 15-60s)                    |
| `/ping`    | SageMaker health check | SageMaker infrastructure                             |
| `/load`    | Server load metrics    | Custom monitoring                                    |

**Notes:**

- This option only affects uvicorn access logs, not vLLM application logs
- Specify multiple endpoints by separating them with commas (no spaces)
- The filter uses exact path matching, query parameters are ignored (e.g., `/health?verbose=true` matches `/health`)
- If you need to completely disable all access logs, use `--disable-uvicorn-access-log` instead

## Additional resources

- [`logging.config` Dictionary Schema Details](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
