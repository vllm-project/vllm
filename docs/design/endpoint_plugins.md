# Endpoint Plugins

Endpoint plugins let out-of-tree packages add HTTP routes to the OpenAI compatible API server. They register routes and optionally per app state used by those routes, and may serve a listener of their own (see [Optional lifecycle methods](#optional-lifecycle-methods)). A plugin reaches the engine the same way an in-tree serving handler does, through the `EngineClient` it is handed at startup (e.g. `engine_client.collective_rpc(...)`). No new engine access path is introduced.

!!! warning "Security"
    Endpoint plugins are **not loaded by default** and must be explicitly allowlisted. Read [Endpoint Plugins security posture](../usage/security.md#endpoint-plugins) before enabling one, especially the route shadowing warning.

## The `EndpointPlugin` protocol

Endpoint plugins implement the [`EndpointPlugin`][vllm.plugins.endpoint_plugins.interface.EndpointPlugin] runtime checkable `Protocol`:

```python
class EndpointPlugin(Protocol):
    name: str
    required_tasks: tuple[SupportedTask, ...] | None

    def attach_router(self, app: FastAPI) -> None: ...

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None: ...
```

- `name`: a unique identifier used in logs and for `VLLM_PLUGINS` allowlisting
- `required_tasks`: the tasks the server must support for this plugin to load. `None` means the plugin has no task requirement
- `attach_router`: registers routes on `app`
- `init_state`: initializes per app state the routes read at request time

## The two phase lifecycle

Routes are registered before the engine exists. This means the interface has to expose two hooks that run at two different points in server startup:

| Phase | Called from | `engine_client` available? | Work |
| --- | --- | --- | --- |
| A. Route registration | `build_app()` | No | `attach_router(app)` add routes. Do not touch the engine here. |
| B. State init | `init_app_state()` | Usually but `None` on the CPU only render server | `init_state(engine_client, state, args)` build a serving handler holding `engine_client` and store it on `state`. |

Because `app.state` *is* the `state` object passed to `init_app_state()`, an object stored during phase A is visible in phase B and an object stored in phase B is visible to route handlers at request time via `request.app.state`. This is the same pattern in-tree endpoints already use.

### Engine less servers (the render server)

The CPU only render server (`init_render_app_state()`) has no `EngineClient`. It still runs both phases for any plugin eligible for the `render` task (`required_tasks` is `None` or includes `"render"`). `attach_router` is called as usual but `init_state` is called with `engine_client=None`.

A plugin that needs an engine to function has two options:

- Exclude `"render"` from `required_tasks` so it is never loaded on the render server in the first place
- Accept being loaded on `render` and check for `None` in `init_state` or in the route handler returning an error response (e.g. HTTP 503) instead of dereferencing a client that doesn't exist

`tests/plugins/vllm_add_dummy_endpoint_plugin` demonstrates the second option. Its route handler returns a 503 when `state.dummy_engine_client` is `None`.

### Reaching the engine from a route handler

`init_state` is where a plugin captures `engine_client` into a small serving handler and stashes it on `state`. The route added in `attach_router` reads that handler off `request.app.state` at request time and calls the engine through it, typically via `engine_client.collective_rpc(...)`.

This minimal example omits the `None` check from the previous section for brevity since `required_tasks` is `None` here. It is in fact eligible for `render` and should handle `engine_client=None` the way `tests/plugins/vllm_add_dummy_endpoint_plugin` does before shipping it:

```python
from fastapi import FastAPI, Request


class MyAdminEndpointPlugin:
    name = "my_admin_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/plugins/my_admin_endpoint_plugin/scheduler_config")
        async def scheduler_config(raw_request: Request):
            engine_client = raw_request.app.state.my_engine_client
            results = await engine_client.collective_rpc("get_scheduler_config")
            return {"scheduler_config": results}

    async def init_state(self, engine_client, state, args) -> None:
        state.my_engine_client = engine_client
```

A complete and tested version of this example is in-repo as `tests/plugins/vllm_add_dummy_endpoint_plugin` and is exercised e2e (including a real HTTP request) in `tests/plugins_tests/test_endpoint_plugins.py`.

## Optional lifecycle methods

A plugin that owns long-lived resources, such as streaming sessions or a listener of its own, can also implement any of these methods. They are optional and are not members of the `EndpointPlugin` protocol, so plugins without them keep working unchanged.

| Method | Called from | Contract |
| --- | --- | --- |
| `async start(state)` | Server lifespan startup, before connections are accepted | Called in load order. If it raises, the plugin cleans up its own partial resources; plugins that already started are stopped in reverse order and startup is aborted. |
| `ready(state) -> bool` | `/health` | Cheap, synchronous and non-blocking. `False` or an exception makes `/health` return 503. Absent means ready. Also used by the render server. |
| `async drain(state, timeout)` | Graceful shutdown, before the engine shuts down | Receives the remaining `--shutdown-timeout` budget in seconds. Close the plugin's listener and finish or abort open work. Errors are logged; a drain still running at the deadline, or when the server exits early (e.g. after engine death), is cancelled and awaited, so it must honor cancellation. |
| `async stop(state)` | Server lifespan shutdown, after HTTP stops | Called exactly once per started plugin, in reverse order, including when the server exits without a graceful shutdown (e.g. after engine death). Must return promptly and must not wait for in-flight work, which is `drain`'s job. Like the rest of lifespan shutdown it is not time-bounded; the process manager's forced termination is the backstop. Errors are logged and do not stop other plugins. |

On SIGINT or SIGTERM the API server shuts down in this order, against one deadline set by `--shutdown-timeout`:

1. `app.state.draining` is set before anything is awaited, so `/health` returns 503. Every plugin admission path (e.g. a listener's connection handler) must check it and refuse new work; core does not close plugin listeners.
2. All plugin `drain` methods run concurrently, bounded by the deadline. With `--shutdown-timeout 0`, no `drain` is called.
3. The engine shuts down with the time left.
4. HTTP stops, then plugins `stop`.

A plugin can start the same graceful shutdown on a fatal error (e.g. its listener died) by calling `app.state.request_shutdown()` from the event loop. It only affects the current API server process.

A plugin that serves a non-HTTP listener (e.g. gRPC) must declare it with the class attribute `owns_non_http_listener = True`. The server then refuses to start with `--api-server-count` other than 1, because each API server process would run its own copy of the listener. That check does not cover data parallel external load balancing (including `--data-parallel-multi-port-external-lb`, whose supervisor runs one API server per rank): there the listener must use a distinct port per rank (e.g. a base port offset by the rank), as the supervisor already gives each rank its own HTTP port. The supervisor refuses `--grpc` for the same port collision. Such listeners are not protected by `--api-key`; see [Security](../usage/security.md#endpoint-plugins).

This example serves a TCP echo listener next to the HTTP server:

```python
import asyncio

from fastapi import FastAPI


class MyListenerEndpointPlugin:
    name = "my_listener_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None
    owns_non_http_listener = True

    def attach_router(self, app: FastAPI) -> None:
        pass

    async def init_state(self, engine_client, state, args) -> None:
        self.state = state
        self.sessions: set[asyncio.Task] = set()

    async def start(self, state) -> None:
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 9000)

    def ready(self, state) -> bool:
        return self.server.is_serving()

    async def drain(self, state, timeout: float) -> None:
        self.server.close()  # stop accepting connections
        await asyncio.gather(*self.sessions, return_exceptions=True)

    async def stop(self, state) -> None:
        for session in self.sessions:
            session.cancel()
        self.server.close()
        await self.server.wait_closed()

    async def handle(self, reader, writer) -> None:
        if self.state.draining:  # refuse new sessions
            writer.close()
            return
        self.sessions.add(session := asyncio.current_task())
        try:
            while data := await reader.read(1024):
                writer.write(data)
                await writer.drain()
        finally:
            self.sessions.discard(session)
            writer.close()
```

`tests/plugins_tests/test_endpoint_plugins.py` exercises these methods through `serve_http`, including drain overruns, engine death during a drain and `--shutdown-timeout 0`, and startup failure through the server lifespan.

## Registering the entry point

Register a zero argument factory (a class or function) under the `vllm.endpoint_plugins` group. The factory must return an object satisfying `EndpointPlugin`:

```toml
# pyproject.toml
[project.entry-points."vllm.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"
```

```python
# setup.py equivalent
setup(
    name="my_pkg",
    entry_points={
        "vllm.endpoint_plugins": [
            "my_admin_api = my_pkg.endpoints:MyAdminEndpointPlugin"
        ]
    },
)
```

The entry point name (`my_admin_api` above) is independent of the plugin's `name` attribute. `VLLM_PLUGINS` allowlisting matches on the **entry point name** following the same convention as `vllm.general_plugins` (see [Plugin System](plugin_system.md)).

## Gating: `VLLM_PLUGINS` and `required_tasks`

Endpoint plugins are discovered and gated by [`load_endpoint_plugins`][vllm.plugins.load_endpoint_plugins] which is stricter than the loader used for other plugin groups:

- **Nothing loads unless `VLLM_PLUGINS` is set and names the plugin.** Other plugin groups load everything unless `VLLM_PLUGINS` narrows the set. Endpoint plugins invert that default because they add network exposed surface. See [Security](../usage/security.md#endpoint-plugins).
- **`required_tasks` must intersect the server's supported tasks** unless it is `None`. Use this to keep a plugin from attaching routes on a server that can't service them (e.g. a pooling only deployment).
- A factory that raises an issue during instantiation is logged and skipped. It does not abort server startup.

Only the front end API server process loads endpoint plugins. There is no need to guard for worker or engine core processes.

## Pairing with `vllm.general_plugins`

Endpoint plugins run in the API server process only. If a plugin also needs new engine side behavior (a new worker-side RPC method, a custom stat) that half ships separately through the existing `vllm.general_plugins` group which loads in worker processes (see [Plugin System](plugin_system.md)). The two entry points are registered and loaded **independently**. Neither implies the other. The recommended distribution shape is a single package exposing both:

```toml
[project.entry-points."vllm.general_plugins"]
my_admin_engine = "my_pkg.engine:register"      # adds the worker side method

[project.entry-points."vllm.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"  # adds the HTTP route
```

Do not expect a single endpoint plugin to also mutate engine/worker state. If your route needs a worker side method that doesn't already exist then add it via a paired `general_plugins` entry point.

## Path-prefix convention

Plugin hooks run after the core routers have been registered. HTTP routes added by plugins replace earlier routes with the same path template and HTTP method, including their OpenAPI description. Methods not overridden by the plugin remain available. When multiple plugins register the same operation, the last registered route takes precedence. Different path templates are not treated as equivalent.

There is currently no rejection of route conflicts (tracked as a follow-up to RFC [#46565](https://github.com/vllm-project/vllm/issues/46565)). To avoid surprising operators:

- Namespace your routes under a distinct prefix, e.g. `/plugins/<plugin-name>/...`, rather than reusing `/v1/...` or other core prefixes
- Only register routes under a core prefix (like the worked example's `/v1/admin/scheduler_config`) if you specifically intend to override or extend existing behavior and document that clearly for operators allowlisting your plugin

## Compatibility

`state`/serving handler internals (e.g. the shape of in-tree `OpenAIServing*` classes) are not a stable public contract yet. Treat them as use-at-your-own-risk and expect them to change between vLLM versions. `FastAPI`, `EngineClient` and the `EndpointPlugin` protocol itself are the supported surface.
