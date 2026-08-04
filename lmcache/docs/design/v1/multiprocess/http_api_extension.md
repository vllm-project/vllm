# HTTP API Extension Design

## Overview

The **HTTP API Extension** framework allows developers to add new HTTP
endpoints to the LMCache multiprocess HTTP server **without modifying
any existing source code**. A new endpoint is simply a Python module
placed in the `http_apis/` directory that exposes a FastAPI `APIRouter`.

This follows the same **zero-modification extension** principle used by
the L2 Adapter plugin system and the Internal API Server.

---

## Key Components

### `HTTPAPIRegistry`

Located at `lmcache/v1/multiprocess/http_api_registry.py`.

Responsible for auto-discovering and registering all API modules:

1. Scans the `http_apis/` directory using `pkgutil.iter_modules`.
2. Imports every module whose name ends with `_api`.
3. Checks for a module-level `router` attribute of type `APIRouter`.
4. Includes the router into the FastAPI application.

### `http_apis/` Directory

Located at `lmcache/v1/multiprocess/http_apis/`.

Each file in this directory that matches the `*_api.py` naming
convention is automatically discovered and registered. Existing
modules:

| Module | Endpoint | Method | Description |
|---|---|---|---|
| `info_api.py` | `/` | GET | Basic liveness check |
| `info_api.py` | `/healthcheck` | GET | K8s probe endpoint |
| `info_api.py` | `/status` | GET | Internal status report |
| `config_api.py` | `/config` | GET | Server config dump |
| `cache_api.py` | `/cache/clear` | POST | Force-clear L1 cache |

### `http_server.py`

The main server module creates the `FastAPI` app and delegates all
route registration to `HTTPAPIRegistry`:

```python
app = FastAPI(
    title="LMCache HTTP API",
    version="1.0.0",
    lifespan=lifespan,
)

registry = HTTPAPIRegistry(app)
registry.register_all_apis()
```

---

## Auto-Discovery Flow

```
http_server.py
  │
  ▼
HTTPAPIRegistry(app)
  │
  ▼
register_all_apis()
  │
  ├─ pkgutil.iter_modules("http_apis/")
  │    ├─ info_api        → has router? ✓ → include
  │    ├─ config_api      → has router? ✓ → include
  │    ├─ cache_api       → has router? ✓ → include
  │    ├─ quota_api       → has router? ✓ → include
  │    └─ my_new_api      → has router? ✓ → include
  │
  └─ app.include_router(collected_router)
```

No changes to `http_server.py` or `http_api_registry.py` are needed
when adding new endpoint modules.

---

## Adding a New Endpoint

### Step 1: Create the Module

Create a new file in `lmcache/v1/multiprocess/http_apis/` with the
`*_api.py` naming convention:

```python
# lmcache/v1/multiprocess/http_apis/metrics_api.py
# SPDX-License-Identifier: Apache-2.0
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request):
    """Return cache hit/miss metrics."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return {"hits": 42, "misses": 7}
```

### Step 2: Done

That's it. The `HTTPAPIRegistry` will automatically discover and
register the new endpoint on the next server startup. No other files
need to be modified.

---

## API Module Contract

An API module **must**:

1. Be placed in `lmcache/v1/multiprocess/http_apis/`.
2. Have a filename ending with `_api.py`.
3. Expose a module-level variable named `router` of type
   `fastapi.APIRouter`.

An API module **should**:

1. Guard against uninitialized engine state by checking
   `request.app.state.engine` and returning 503 if `None`.
2. Use `lmcache.logging.init_logger(__name__)` for logging.
3. Follow the project's import ordering convention
   (Standard → Third Party → First Party).

An API module **must not**:

1. Directly import or modify the `app` object from `http_server.py`.
2. Perform blocking I/O in endpoint handlers (use `async` properly).

---

## Accessing Shared State

The FastAPI `app.state` object is the shared context between the
server lifecycle and all endpoint handlers. Available attributes
(set during the `lifespan` startup phase):

| Attribute | Type | Description |
|---|---|---|
| `app.state.engine` | Cache engine instance | Main cache engine for KV operations |
| `app.state.zmq_server` | ZMQ server instance | Underlying multiprocess ZMQ server |

Access these via the `Request` object in your handler:

```python
@router.get("/my-endpoint")
async def my_endpoint(request: Request):
    engine = request.app.state.engine
    # ... use engine ...
```
