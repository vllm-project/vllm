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
