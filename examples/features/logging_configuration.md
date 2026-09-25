# Logging Configuration

vLLM leverages Python's `logging.config.dictConfig` functionality to enable
robust and flexible configuration of the various loggers used by vLLM.

For `vllm serve`, configure logging with CLI arguments:

- Use the built-in configuration, optionally with `--log-level`.
- Use a custom Python logging configuration file with
  `--logging-config.pylogging_config_file`.
- Disable vLLM logging configuration with
  `--logging-config.configure_logging false`.

## CLI logging configuration

`--logging-config` accepts a JSON object. Its fields are `log_level`,
`configure_logging`, and `pylogging_config_file`. For example:

```bash
vllm serve mistralai/Mistral-7B-v0.1 \
    --logging-config '{"log_level":"DEBUG","configure_logging":true}'
```

Fields can also be set individually with dotted arguments:

```bash
vllm serve mistralai/Mistral-7B-v0.1 \
    --logging-config.log_level DEBUG
```

`--log-level` is a shortcut for `--logging-config.log_level` and takes
precedence if both are supplied. It sets the level of vLLM's built-in logging
configuration. `--log-config-file` is deprecated and will be removed in
v0.33.0; use `--logging-config.pylogging_config_file` in new commands.

If `configure_logging` is `false`, vLLM does not apply a logging
configuration. It cannot be combined with `pylogging_config_file`. This has
the same effect as the legacy `VLLM_CONFIGURE_LOGGING=0` setting.

The custom logging configuration file must be JSON following Python's [logging
configuration dictionary
schema](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

!!! note "Custom configurations override `--log-level`"
    When `pylogging_config_file` is set, vLLM loads that JSON file and replaces
    its built-in `dictConfig`; it does not merge the two. Therefore,
    `--log-level` applies only when no custom configuration file is provided.
    Set logger and handler levels in the custom file itself. vLLM applies the
    resolved configuration in its child processes as well.

## Environment variables

The CLI configuration is recommended for `vllm serve`. The legacy
`VLLM_CONFIGURE_LOGGING`, `VLLM_LOGGING_LEVEL`, and
`VLLM_LOGGING_CONFIG_PATH` variables remain supported as defaults. The default
handler's stream, prefix, and color are currently controlled only through
environment variables. Values from a YAML `--config` file or the command line
override those defaults. See [Environment Variables](https://docs.vllm.ai/en/latest/configuration/env_vars/)
for those settings.

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

Finally, run vLLM with the custom logging configuration JSON file:

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048 \
    --logging-config.pylogging_config_file /path/to/logging_config.json
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

Finally, run vLLM with the custom logging configuration JSON file:

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048 \
    --logging-config.pylogging_config_file /path/to/logging_config.json
```

### Example 3: Disable vLLM default logging configuration

To disable vLLM's default logging configuration and silence vLLM log output,
set `--logging-config.configure_logging false` when running vLLM. This prevents
vLLM from configuring the root vLLM logger, which in turn silences other vLLM
loggers unless the application or Python root logger configures a handler.

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048 \
    --logging-config.configure_logging false
```

For legacy launch scripts, `VLLM_CONFIGURE_LOGGING=0` has the same effect.

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
