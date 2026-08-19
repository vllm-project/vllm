# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for `vllm.endpoint_plugins` entry points.

An endpoint plugin adds HTTP routes to the OpenAI compatible API server.
Its scope is HTTP surface only. It registers routes and optionally
per app state used by those routes. It must not open new paths into the
engine by reaching the engine the same way an in-tree serving handler does
via `EngineClient` (e.g. `engine_client.collective_rpc(...)`).

If a plugin also needs engine side behavior (a new worker side RPC method,
a custom stat, etc.) pair this entry point with one registered under
`vllm.general_plugins` (see `vllm/plugins/__init__.py`). The
`general_plugins` entry installs the engine side method and the
`endpoint_plugins` entry exposes it over HTTP. The two are registered and
loaded independently where neither implies the other.

Plugins are opt-in. See `load_endpoint_plugins` in `vllm/plugins/__init__.py`
for the loading/gating rules and `docs/usage/security.md` for the security
posture of exposing plugin defined routes.

The CPU only render server (see `build_and_serve_renderer` in
`vllm/entrypoints/launchers/render`) has no `EngineClient`. A plugin
eligible for the `render` task (`required_tasks` is `None` or includes
`"render"`) still gets `attach_router` called but `init_state` receives
`engine_client=None`. Plugins that cannot function without an engine should
either exclude `"render"` from `required_tasks` or check for `None` in
`init_state`/their route handlers and degrade gracefully.
"""

from argparse import Namespace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import FastAPI
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask


@runtime_checkable
class EndpointPlugin(Protocol):
    """Protocol implemented by `vllm.endpoint_plugins` entry point factories.

    An entry point registered under the `vllm.endpoint_plugins` group must
    resolve to a zero argument callable (a class or factory function) that
    returns an object satisfying this protocol.
    """

    name: str
    """Unique plugin name used in logs and for `VLLM_PLUGINS` allowlisting."""

    required_tasks: "tuple[SupportedTask, ...] | None"
    """Tasks the server must support for this plugin to be loaded.

    The plugin is loaded only if this set intersects the server's
    `supported_tasks`. `None` means the plugin has no task requirement and
    is always eligible (subject to the `VLLM_PLUGINS` allowlist).
    """

    def attach_router(self, app: FastAPI) -> None:
        """Register this plugin's routes on `app`.

        Called once during `build_app()` after all core routers have been
        attached. Routes attached here can shadow core routes with the same
        path. There is currently no conflict enforcement (see RFC #46565 follow ups).
        """
        ...

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        """Initialize per app state consumed by this plugin's routes.

        Called once during `init_app_state()` after core state has been
        initialized. Use `engine_client` (e.g. `collective_rpc`) to reach
        the engine. Do not open new engine access paths.

        `engine_client` is `None` on the CPU only render server which has
        no engine. This only happens for plugins eligible for the `render`
        task (`required_tasks` is `None` or includes `"render"`). Handle
        `None` explicitly (e.g. skip engine dependent setup, or have route
        handlers return an error) if the plugin is loadable for `render` but
        cannot function without an engine.
        """
        ...


def attach_endpoint_plugins(
    app: FastAPI, supported_tasks: tuple["SupportedTask", ...]
) -> None:
    """Phase A of endpoint plugin wiring: discover, gate and attach routes.

    Attached last after all core routers. This is so endpoint plugin routes can
    shadow core routes with the same path (see `EndpointPlugin.attach_router`
    docstring). No-ops when no plugins are discovered/allowlisted.
    """
    from vllm.plugins import load_endpoint_plugins

    endpoint_plugins = load_endpoint_plugins(supported_tasks)
    for plugin in endpoint_plugins:
        plugin.attach_router(app)
    app.state.endpoint_plugins = endpoint_plugins


async def init_endpoint_plugins_state(
    engine_client: EngineClient | None, state: State, args: Namespace
) -> None:
    """Phase B of endpoint plugin wiring: initialize per app plugin state.

    `state.endpoint_plugins` is set by `_attach_endpoint_plugins` (Phase A)
    in `build_app`. Some `init_app_state` callers (e.g. `run_batch.py`)
    build their own bare `State` without going through `build_app`. As a result
    `endpoint_plugins` may be absent and are treated that the same as "none attached".

    `engine_client` is `None` for the CPU only render server which has no
    engine (see `init_render_app_state`). Plugins must handle a `None`
    `engine_client` themselves (see `EndpointPlugin.init_state`).
    """
    for plugin in getattr(state, "endpoint_plugins", []):
        await plugin.init_state(engine_client, state, args)
