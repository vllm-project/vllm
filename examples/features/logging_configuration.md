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

### Correlate logs with OpenTelemetry traces

Set `VLLM_LOGGING_TRACE_CONTEXT=1` before starting vLLM to include
`trace_id` (32 hexadecimal characters) and `span_id` (16 hexadecimal characters)
in the default text logs. This environment variable is disabled by default;
it is not a `vllm serve` command-line option.
It requires `opentelemetry-api`; it does not initialize a tracer or export logs.

```bash
VLLM_LOGGING_TRACE_CONTEXT=1 VLLM_LOGGING_COLOR=0 \
    vllm serve facebook/opt-125m \
    --otlp-traces-endpoint=http://localhost:4317
```

Only logs emitted within an active OpenTelemetry context carry valid IDs.
Logs outside a span omit the trace header entirely. Unsampled
spans still supply IDs, but might not be available in the trace backend.
In particular, creating a request span retrospectively does not attach its
context to earlier logs. Worker batch logs may cover multiple requests and
cannot be assigned a single request trace automatically.

For HTTP serving, this switch also binds valid incoming W3C `traceparent` and
`tracestate` headers for the duration of the request, including streaming
responses. No OTLP endpoint is needed for this log correlation. The logged
span ID is the incoming parent span's ID; this middleware does not create a
server span. An existing active span from other instrumentation takes precedence.
Missing or invalid headers do not produce a trace header. Context is restored
on completion or failure, and this does not propagate context to engine worker
processes. Trace export still requires the existing tracing configuration.

To verify incoming HTTP headers, start the server with request logging enabled
(request logs can include prompt content):

```bash
VLLM_LOGGING_TRACE_CONTEXT=1 VLLM_LOGGING_COLOR=0 \
    vllm serve /path/to/model --served-model-name test-model --enable-log-requests

curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'traceparent: 00-11111111111111111111111111111111-2222222222222222-01' \
    -d '{"model":"test-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'
```

The vLLM request log should include
`[trace_id=11111111111111111111111111111111 span_id=2222222222222222]`.
Repeat without the `traceparent` header to verify that the trace prefix is
absent. Startup logs, aggregate engine statistics, and Uvicorn access logs
are not the request log being checked here. Without trace export configured,
vLLM may still warn that engine tracing is disabled; HTTP log correlation
does not enable engine tracing.

The switch applies to the built-in vLLM logging configuration. For a custom
`VLLM_LOGGING_CONFIG_PATH`, add this filter to the handlers that need correlation
and include `%(trace_context)s` before `%(message)s` in the format. This field
includes a trailing space when a valid context exists and is empty otherwise.
For structured logging, `trace_id`, `span_id`, and `trace_sampled` remain
available as record attributes (zero IDs and `False` outside a span):

```json
"filters": {
  "trace_context": {
    "()": "vllm.logging_utils.trace_context.TraceContextFilter"
  }
}
```

Add `"filters": ["trace_context"]` to the handler definition. For queue-based
logging, attach the filter to the producer's `QueueHandler` so that context is
captured before crossing threads. Uvicorn access logs use separate handlers;
the default switch does not change them.

To verify without a GPU or a collector, run in an installed vLLM environment:

```bash
VLLM_LOGGING_TRACE_CONTEXT=1 VLLM_LOGGING_COLOR=0 .venv/bin/python - <<'PY'
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vllm.logger import init_logger

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
logger = init_logger("vllm.correlation_check")
with provider.get_tracer("verification").start_as_current_span("log-check"):
    logger.info("inside span")
logger.info("outside span")
provider.shutdown()
PY
```

The `inside span` log IDs must match the console-exported span IDs (ignoring
the exporter's `0x` prefix); the `outside span` log must omit the trace header.
Repeat with `VLLM_LOGGING_TRACE_CONTEXT=0` to verify the original log format.
For platform validation, collect stdout with your logging agent, parse these
fields, and configure a trace lookup using `trace_id`. Export spans to the same
observability platform using the existing OTLP configuration. Log collection
and the platform's log-to-trace link must be configured separately.

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
          "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
          "format": "%(asctime)s %(levelname)s %(name)s %(vllm_process_name)s %(process)d %(message)s"
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

Each vLLM log record is one JSON object and includes `vllm_process_name`, which
identifies vLLM's logical process (including worker ranks where applicable).
The standard `process` record attribute contains the operating-system PID.
vLLM avoids altering `stdout` or `stderr`, so no text is prepended to JSON
log records.

This applies to records emitted through the configured Python loggers. A JSON
formatter cannot convert unrelated output into JSON, such as a third-party
library writing directly to `stdout` or `stderr`; configure or route such
output separately when a consumer requires every collected line to be JSON.

When serving, `VLLM_LOGGING_CONFIG_PATH` is also used as Uvicorn's logging
configuration. This example configures only `vllm`; configure `uvicorn`,
`uvicorn.error`, and `uvicorn.access` with a JSON handler when those server
logs must also be structured.

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
