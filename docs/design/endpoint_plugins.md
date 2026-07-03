# Endpoint Plugins

Endpoint plugins let out-of-tree packages add HTTP routes to the OpenAI compatible API server without editing `vllm/entrypoints/openai/api_server.py`. Their scope is
the **HTTP surface only** registering routes and optionally per app state used by those routes. A plugin reaches the engine the same way an in-tree serving handler does, through the `EngineClient` it is handed at startup (e.g. `engine_client.collective_rpc(...)`). No new engine access path is introduced.

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

A complete, tested version of this example ships in-repo as `tests/plugins/vllm_add_dummy_endpoint_plugin` and is exercised e2e (including a real HTTP request) in `tests/plugins_tests/test_endpoint_plugins.py`.

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

Endpoint plugins cover the HTTP surface only. If a plugin also needs new engine side behavior (a new worker-side RPC method, a custom stat) that half ships separately through the existing `vllm.general_plugins` group which loads in worker processes (see [Plugin System](plugin_system.md)). The two entry points are registered and loaded **independently**. Neither implies the other. The recommended distribution shape is a single package exposing both:

```toml
[project.entry-points."vllm.general_plugins"]
my_admin_engine = "my_pkg.engine:register"      # adds the worker side method

[project.entry-points."vllm.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"  # adds the HTTP route
```

Do not expect a single endpoint plugin to also mutate engine/worker state. If your route needs a worker side method that doesn't already exist then add it via a paired `general_plugins` entry point.

## Path-prefix convention

There is currently no route conflict enforcement (tracked as a follow-up to RFC [#46565](https://github.com/vllm-project/vllm/issues/46565)). A plugin's `attach_router` can register a path that collides with a core route and routes attached later win. To avoid surprising operators:

- Namespace your routes under a distinct prefix, e.g. `/plugins/<plugin-name>/...`, rather than reusing `/v1/...` or other core prefixes
- Only register routes under a core prefix (like the worked example's `/v1/admin/scheduler_config`) if you specifically intend to override or extend existing behavior and document that clearly for operators allowlisting your plugin

## Compatibility

`state`/serving handler internals (e.g. the shape of in-tree `OpenAIServing*` classes) are not a stable public contract yet. Treat them as use-at-your-own-risk and expect them to change between vLLM versions. `FastAPI`, `EngineClient` and the `EndpointPlugin` protocol itself are the supported surface.
