# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import resources
from urllib.parse import unquote
import argparse
import asyncio
import json
import os
import threading
import time

# Third Party
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, PlainTextResponse
import httpx
import uvicorn

_PACKAGE = "lmcache.lmcache_frontend"


def _package_resource_path(relative: str) -> str:
    """Return absolute filesystem path for a file shipped inside the package.

    Replacement for the deprecated ``pkg_resources.resource_filename``;
    works for regular (non-zipped) installs, which is how LMCache ships.
    """
    return str(resources.files(_PACKAGE).joinpath(relative))


try:
    # Local
    from .heartbeat import HeartbeatService  # import as module
except ImportError:
    # Third Party
    from heartbeat import HeartbeatService  # type: ignore  # import as script


# Create router
router = APIRouter()


class _NodeRegistry:
    """Encapsulates the mutable proxy/node list used by the frontend.

    Replacing the list is done in-place via :py:meth:`replace` so that
    aliases (the module-level ``target_nodes`` reference and the list
    handed to ``HeartbeatService``) stay in sync.
    """

    def __init__(self) -> None:
        self._nodes: list[dict] = []

    @property
    def nodes(self) -> list[dict]:
        """Return the underlying list (mutated in place)."""
        return self._nodes

    def replace(self, new_nodes: list[dict]) -> None:
        """Swap the registry content in place with ``new_nodes``."""
        self._nodes[:] = new_nodes

    def is_allowed(self, host: str, port: str) -> bool:
        """Return True if ``host:port`` matches any registered node.

        Used by the SSRF guard in :func:`proxy_request` so the proxy
        only forwards to pre-registered destinations.
        """
        return self.resolve(host, port) is not None

    def resolve(self, host: str, port: str) -> tuple[str, str] | None:
        """Return the registry-owned ``(host, port)`` matching the input.

        The returned tuple is taken from the registry itself, not from
        the caller-supplied arguments.  Using this value to build the
        outbound URL breaks the SSRF taint flow for static analysers.
        """
        port = str(port)
        for proxy in self._nodes:
            p_host, p_port = str(proxy.get("host")), str(proxy.get("port"))
            if p_host == host and p_port == port:
                return p_host, p_port
            for child in proxy.get("nodes", []):
                c_host, c_port = str(child.get("host")), str(child.get("port"))
                if c_host == host and c_port == port:
                    return c_host, c_port
        return None


_node_registry = _NodeRegistry()
# ``target_nodes`` is a module-level alias to the list owned by the
# registry.  External readers and in-place mutations (append / element
# update) keep working unchanged; whole-list replacement MUST go
# through ``_node_registry.replace`` so all aliases stay in sync.
target_nodes = _node_registry.nodes

# Initialize heartbeat service with app context
heartbeat_service: HeartbeatService = HeartbeatService()

global args
args = None


async def fetch_child_nodes_from_proxy(proxy_node: dict) -> list[dict]:
    """Fetch child nodes from a single proxy node.

    Args:
        proxy_node: Proxy dict with ``name``/``host``/``port`` keys.
            ``/api/nodes`` of the proxy is queried.

    Returns:
        A list of leaf node dicts discovered from the proxy.  An
        empty list is returned when the request fails.
    """
    try:
        url = f"http://{proxy_node['host']}:{proxy_node['port']}/api/nodes"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return_nodes = []
            added_names = set()  # Track node names to avoid duplicates
            nodes = response.json().get("nodes", [])

            # Process each node: only add leaf nodes (nodes without children)
            for node in nodes:
                if node.get("children"):
                    # If node has children, add only the children (leaf nodes)
                    for child in node["children"]:
                        # Skip if already added
                        if child["name"] in added_names:
                            continue
                        child["proxy_id"] = proxy_node["name"]
                        return_nodes.append(child)
                        added_names.add(child["name"])
                else:
                    # Skip if already added
                    if node["name"] in added_names:
                        continue
                    node["proxy_id"] = proxy_node["name"]
                    return_nodes.append(node)
                    added_names.add(node["name"])
            return return_nodes
    except Exception as e:
        print(f"Failed to fetch nodes from proxy {proxy_node['name']}: {e}")
        return []


async def fetch_all_child_nodes_concurrently(
    proxy_nodes: list[dict],
) -> list[dict]:
    """Fetch child nodes from multiple proxy nodes concurrently.

    Args:
        proxy_nodes: Proxy dicts to query.  Each dict is mutated in
            place with a new ``nodes`` field.

    Returns:
        The same ``proxy_nodes`` list with ``nodes`` populated.
    """
    tasks = [fetch_child_nodes_from_proxy(proxy) for proxy in proxy_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error fetching nodes from proxy {proxy_nodes[i]['name']}: {result}")
            proxy_nodes[i]["nodes"] = []
        else:
            proxy_nodes[i]["nodes"] = result

    return proxy_nodes


async def _fetch_child_nodes_from_api_nodes(
    host: str, port: str, proxy_name: str
) -> list:
    """Try to fetch child nodes via /api/nodes from a multiProcess server.

    Returns a non-empty list of child node dicts when the target is a
    multiProcess lmcache server that exposes /api/nodes.  Returns an
    empty list for inProcess nodes (connection error or no children).
    """
    try:
        url = "http://%s:%s/api/nodes" % (host, port)
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        nodes_data = response.json().get("nodes", [])
        children = []
        for node in nodes_data:
            if node.get("children"):
                for child in node["children"]:
                    child["proxy_id"] = proxy_name
                    children.append(child)
            else:
                node["proxy_id"] = proxy_name
                children.append(node)
        return children
    except Exception:
        return []


async def fetch_nodes_from_supplier(url: str) -> list[dict]:
    """Fetch node information from node supplier.

    For each discovered node, attempt to retrieve its child nodes via
    ``/api/nodes`` (multiProcess lmcache server).  If that call
    succeeds and returns children, those children are used as the
    ``nodes`` list (multiProcess mode).  Otherwise the node itself is
    used as the sole child (inProcess / leaf mode).
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            unique_nodes: dict[str, dict] = {}
            for api_address, info in data.get("processInfos", {}).items():
                for entity in info.get("lmCacheInfoEntities", []):
                    addr = entity["apiAddress"]
                    if addr.startswith("http://"):
                        addr = addr[7:]
                    host, port = addr.split(":")
                    node_key = "%s:%s" % (host, port)
                    if node_key not in unique_nodes:
                        unique_nodes[node_key] = {
                            "host": host,
                            "port": port,
                        }

            # Fetch child nodes concurrently for all discovered nodes.
            proxy_list = [
                {
                    "name": "proxy_%s" % node_key.replace(":", "_"),
                    "host": info["host"],
                    "port": info["port"],
                }
                for node_key, info in unique_nodes.items()
            ]
            child_tasks = [
                _fetch_child_nodes_from_api_nodes(p["host"], p["port"], p["name"])
                for p in proxy_list
            ]
            child_results = await asyncio.gather(*child_tasks, return_exceptions=True)

            result = []
            for proxy, children in zip(proxy_list, child_results, strict=False):
                if isinstance(children, Exception) or not children:
                    # inProcess mode: node itself is the leaf target.
                    leaf = {
                        "name": proxy["name"],
                        "host": proxy["host"],
                        "port": proxy["port"],
                    }
                    nodes = [leaf]
                else:
                    # multiProcess mode: use discovered child nodes.
                    assert not isinstance(children, BaseException)
                    nodes = children
                result.append(
                    {
                        "name": proxy["name"],
                        "host": proxy["host"],
                        "port": proxy["port"],
                        "nodes": nodes,
                    }
                )
            return result
    except Exception as e:
        print(f"Failed to fetch nodes from supplier: {e}")
        return []


def load_config(config_path: str | None = None) -> None:
    """Load proxy/node list from a JSON config file.

    Args:
        config_path: Optional path to the JSON file.  When ``None`` the
            packaged ``config.json`` is used.
    """
    try:
        # Prioritize user-specified configuration file
        if config_path:
            with open(config_path, "r") as f:
                _node_registry.replace(json.load(f))
            print(
                f"Loaded {len(target_nodes)} target nodes from specified path: "
                f"{config_path}"
            )
        else:
            # Use package resource path as default configuration
            default_config_path = _package_resource_path("config.json")
            with open(default_config_path, "r") as f:
                _node_registry.replace(json.load(f))
            print(f"Loaded default configuration with {len(target_nodes)} target nodes")
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        _node_registry.replace([])


def validate_node(node: dict, is_proxy: bool = False) -> bool:
    """Validate a single node configuration dict.

    Args:
        node: Candidate node dict.
        is_proxy: Reserved for future proxy-specific validation.

    Returns:
        True when ``node`` has the required ``name``/``host``/``port``
        keys and (optionally) a string ``proxy_id``.
    """
    if not isinstance(node, dict):
        return False

    required_keys = {"name", "host", "port"}
    if not required_keys.issubset(node.keys()):
        return False

    if "proxy_id" in node and node["proxy_id"]:
        if not isinstance(node["proxy_id"], str):
            return False

    return True


def validate_nodes(nodes: list) -> bool:
    """Validate a list of node dicts; see :func:`validate_node`."""
    if not isinstance(nodes, list):
        return False

    return all(validate_node(node) for node in nodes)


@router.get("/api/nodes")
async def get_all_nodes() -> dict:
    """Get all nodes in tree structure (proxies with their child nodes).

    Returns:
        ``{"nodes": [...]}`` where each element is a proxy dict whose
        ``children`` list contains the leaf nodes behind it.
    """
    all_nodes = []
    for proxy in target_nodes:
        # Create proxy node with children property
        proxy_node = {
            "id": f"proxy_{proxy['name']}",
            "name": proxy["name"],
            "host": proxy["host"],
            "port": proxy["port"],
            "is_proxy": True,
            "children": [],
        }

        # Add child nodes
        for node in proxy.get("nodes", []):
            proxy_node["children"].append(
                {
                    "id": "node_%s" % node["name"],
                    "name": node["name"],
                    "host": node["host"],
                    "port": node["port"],
                    "is_proxy": False,
                    "proxy_id": proxy["name"],
                }
            )

        all_nodes.append(proxy_node)

    return {"nodes": all_nodes}


@router.get("/api/proxies/{proxy_name}/refresh")
async def refresh_proxy_nodes(proxy_name: str):
    """Refresh child nodes of a proxy"""
    proxy = next((p for p in target_nodes if p["name"] == proxy_name), None)
    if not proxy:
        raise HTTPException(status_code=404, detail="Proxy not found")

    try:
        child_nodes = await fetch_child_nodes_from_proxy(proxy)
        proxy["nodes"] = child_nodes
        return {"status": "success", "nodes": child_nodes}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/proxies")
async def get_proxies():
    """Get all proxy nodes (without child nodes)"""
    proxies = [
        {"name": proxy["name"], "host": proxy["host"], "port": proxy["port"]}
        for proxy in target_nodes
    ]
    return {"proxies": proxies}


@router.get("/api/proxies/{proxy_name}/nodes")
async def get_proxy_nodes(proxy_name: str):
    """Get all nodes under specified proxy"""
    proxy = next((p for p in target_nodes if p["name"] == proxy_name), None)
    if not proxy:
        raise HTTPException(status_code=404, detail="Proxy not found")
    return {"nodes": proxy["nodes"]}


# ==== Node Management Endpoints ====
@router.post("/api/proxies")
async def add_proxy(request: Request):
    """Add a new proxy node"""
    try:
        new_proxy = await request.json()
        if not validate_node(new_proxy, is_proxy=True):
            raise HTTPException(status_code=400, detail="Invalid proxy format")

        # Check for duplicate names
        if any(proxy["name"] == new_proxy["name"] for proxy in target_nodes):
            raise HTTPException(status_code=409, detail="Proxy name already exists")

        # Ensure nodes field exists
        if "nodes" not in new_proxy:
            new_proxy["nodes"] = []

        target_nodes.append(new_proxy)
        return {"status": "success", "message": "Proxy added"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/proxies/{proxy_name}/nodes")
async def add_node_to_proxy(proxy_name: str, request: Request):
    """Add child node to proxy"""
    try:
        new_node = await request.json()
        if not validate_node(new_node):
            raise HTTPException(status_code=400, detail="Invalid node format")

        # Find corresponding proxy
        proxy = next((p for p in target_nodes if p["name"] == proxy_name), None)
        if not proxy:
            raise HTTPException(status_code=404, detail="Proxy not found")

        # Check for duplicate names
        if any(node["name"] == new_node["name"] for node in proxy["nodes"]):
            raise HTTPException(status_code=409, detail="Node name already exists")

        proxy["nodes"].append(new_node)
        return {"status": "success", "message": "Node added to proxy"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/api/nodes/{node_name}")
async def update_node(node_name: str, request: Request):
    """Update a proxy or child node identified by ``node_name``.

    Searches top-level proxies first, then children of every proxy,
    so both direct proxies and supplier-discovered leaf nodes can be
    updated through the same endpoint (mirrors ``proxy_request_by_name``).
    """
    try:
        updated_node = await request.json()
        if not validate_node(updated_node):
            raise HTTPException(status_code=400, detail="Invalid node format")

        for i, node in enumerate(target_nodes):
            if node["name"] == node_name:
                # Preserve existing children when the caller did not
                # provide one; proxies are expected to keep ``nodes``.
                if "nodes" not in updated_node and "nodes" in node:
                    updated_node["nodes"] = node["nodes"]
                target_nodes[i] = updated_node
                return {"status": "success", "message": "Node updated"}

        for proxy in target_nodes:
            children = proxy.get("nodes", [])
            for j, child in enumerate(children):
                if child["name"] == node_name:
                    children[j] = updated_node
                    return {
                        "status": "success",
                        "message": "Child node updated",
                    }

        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/api/nodes/{node_name}")
async def delete_node(node_name: str):
    """Delete a proxy or child node by ``node_name``.

    Mirrors :func:`update_node`: matches top-level proxies first,
    falls back to scanning children of each proxy.
    """
    try:
        original_count = len(target_nodes)
        _node_registry.replace(
            [node for node in target_nodes if node["name"] != node_name]
        )

        if len(target_nodes) != original_count:
            return {"status": "success", "message": "Node deleted"}

        for proxy in target_nodes:
            children = proxy.get("nodes", [])
            for idx, child in enumerate(children):
                if child["name"] == node_name:
                    del children[idx]
                    return {
                        "status": "success",
                        "message": "Child node deleted",
                    }

        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.api_route(
    "/proxy2/{node_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_request_by_name(request: Request, node_name: str, path: str):
    """Proxy requests using node name as identifier.

    Searches top-level target_nodes first, then child nodes of every
    proxy, so both direct nodes and supplier-discovered leaf nodes are
    reachable with a single /proxy2/{name}/{path} call.
    """
    # 1. top-level proxy nodes
    node = next((n for n in target_nodes if n["name"] == node_name), None)

    # 2. child nodes of every proxy
    if not node:
        for proxy in target_nodes:
            node = next(
                (n for n in proxy.get("nodes", []) if n["name"] == node_name),
                None,
            )
            if node:
                break

    if not node:
        raise HTTPException(
            status_code=404, detail=f"Node with name '{node_name}' not found"
        )

    # Use existing proxy_request logic
    return await proxy_request(
        request, target_host=node["host"], target_port_or_socket=node["port"], path=path
    )


@router.api_route(
    "/proxy/{target_host}/{target_port_or_socket}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_request(
    request: Request, target_host: str, target_port_or_socket: str | int, path: str
):
    """Proxy requests to the specified target host and port or socket path.

    For security, non-socket targets must match a host/port already
    registered in :data:`_node_registry`; this prevents the endpoint
    from being used as an open relay (SSRF).  Socket paths are
    accepted as-is because they are local UDS endpoints.
    """
    target_port_or_socket = unquote(str(target_port_or_socket))
    # Check if target_port_or_socket is a socket path (contains '/')
    is_socket_path = "/" in target_port_or_socket

    if is_socket_path:
        # For socket paths, use UDS transport
        socket_path = target_port_or_socket
        target_url = f"http://localhost/{path}"

        # Create UDS transport
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
    else:
        port = target_port_or_socket
        # SSRF guard: resolve against the registry and reuse the
        # trusted host/port from there when building the outbound URL.
        # This keeps user-controlled values out of the URL sink.
        resolved = _node_registry.resolve(target_host, port)
        if resolved is None:
            raise HTTPException(
                status_code=403,
                detail=("Target %s:%s is not a registered node" % (target_host, port)),
            )
        safe_host, safe_port = resolved
        target_url = f"http://{safe_host}:{safe_port}/{path}"
        transport = None  # Use default transport

    headers = {}
    for key, value in request.headers.items():
        if key.lower() in [
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        ]:
            continue
        headers[key] = value

    body = await request.body()

    # Create client with appropriate transport
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
                timeout=60.0,
            )

            response_headers = {}
            for key, value in response.headers.items():
                if key.lower() in [
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                ]:
                    continue
                response_headers[key] = value

            return PlainTextResponse(
                content=response.content,
                headers=response_headers,
                media_type=response.headers.get("content-type", "text/plain"),
                status_code=response.status_code,
            )

        except httpx.ConnectError as e:
            if is_socket_path:
                detail = f"Failed to connect to socket: {socket_path}"
            else:
                detail = f"Failed to connect to target service {target_host}:{port}"
            raise HTTPException(status_code=502, detail=detail) from e
        except httpx.TimeoutException as e:
            if is_socket_path:
                detail = f"Connection to socket {socket_path} timed out"
            else:
                detail = f"Connection to target service {target_host}:{port} timed out"
            raise HTTPException(status_code=504, detail=detail) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with target service: {str(e)}",
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lmcache-monitor"}


@router.get("/api/heartbeat/status")
async def get_heartbeat_status():
    """Get heartbeat status"""
    return heartbeat_service.status()


@router.post("/api/heartbeat/start")
async def start_heartbeat_api(request: Request):
    """Start heartbeat service"""
    try:
        data = await request.json()
        heartbeat_url = data.get("heartbeat_url")
        initial_delay = data.get("initial_delay", 0)
        interval = data.get("interval", 30)

        if not heartbeat_url:
            raise HTTPException(status_code=400, detail="heartbeat_url is required")

        heartbeat_service.start(heartbeat_url, initial_delay, interval)
        return {"status": "success", "message": "Heartbeat service started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/heartbeat/stop")
async def stop_heartbeat_api():
    """Stop heartbeat service"""
    try:
        heartbeat_service.stop()
        return {"status": "success", "message": "Heartbeat service stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def load_nodes_from_supplier(node_supplier_url: str | None = None) -> bool:
    """Load node information from node supplier.

    Args:
        node_supplier_url: Discovery service URL, e.g. ``/lmcache_infos``.

    Returns:
        ``True`` when a non-empty node list was retrieved and stored.
    """
    if not node_supplier_url:
        return False

    print(f"Fetching nodes from supplier: {node_supplier_url}")
    nodes = await fetch_nodes_from_supplier(node_supplier_url)
    if nodes:
        _node_registry.replace(nodes)
        print(f"Loaded {len(target_nodes)} proxy nodes from supplier")

        # Child nodes are already populated by fetch_nodes_from_supplier;
        # no secondary /api/nodes round-trip is needed.
        for proxy in target_nodes:
            print(f"Proxy {proxy['name']} loaded {len(proxy['nodes'])} child nodes")
        return True
    else:
        print("Warning: No nodes loaded from supplier")
        return False


async def initialize_nodes(node_supplier_url: str | None = None) -> None:
    """Initialize node configuration from CLI args or supplier URL.

    Args:
        node_supplier_url: Optional discovery service URL.  When set,
            nodes are loaded via :func:`load_nodes_from_supplier`.
            Otherwise falls back to ``args.nodes`` / ``args.config``.
    """
    global args

    if args is None:
        raise ValueError("args is not initialized")

    if node_supplier_url:
        await load_nodes_from_supplier(node_supplier_url)
    elif args.nodes:
        try:
            nodes = json.loads(args.nodes)
            if validate_nodes(nodes):
                _node_registry.replace(
                    [
                        {
                            "name": "local_proxy",
                            "host": args.host,
                            "port": args.port,
                            "nodes": nodes,
                        }
                    ]
                )
                print(f"Loaded {len(nodes)} target nodes from command line arguments")
        except json.JSONDecodeError:
            print("Failed to parse nodes JSON parameter")
    elif args.config:
        load_config(args.config)


# Minimum seconds between two supplier refreshes triggered by ``/``.
# Each refresh fans out to every registered proxy's ``/api/nodes``
# endpoint, so an unthrottled ``/`` would DOS both the supplier and
# every proxy. 30s matches the default heartbeat interval.
_SUPPLIER_REFRESH_INTERVAL_SEC = 30.0
_supplier_last_refresh: float = 0.0
_supplier_refresh_lock = asyncio.Lock()


async def _maybe_refresh_from_supplier(node_supplier_url: str) -> None:
    """Refresh the node registry from supplier, at most once per interval.

    The first caller within the interval performs the refresh; other
    concurrent callers return immediately (stale-on-read). This keeps
    the homepage responsive even under high traffic.
    """
    global _supplier_last_refresh
    now = time.monotonic()
    if now - _supplier_last_refresh < _SUPPLIER_REFRESH_INTERVAL_SEC:
        return
    if _supplier_refresh_lock.locked():
        return
    async with _supplier_refresh_lock:
        now = time.monotonic()
        if now - _supplier_last_refresh < _SUPPLIER_REFRESH_INTERVAL_SEC:
            return
        await initialize_nodes(node_supplier_url)
        _supplier_last_refresh = time.monotonic()


@router.get("/")
async def serve_frontend():
    """Return frontend homepage.

    When a node supplier URL is configured, trigger a throttled
    background-style refresh so opening the homepage repeatedly does
    not hammer the supplier or every proxy's ``/api/nodes``.
    """
    if args.node_supplier_url:
        await _maybe_refresh_from_supplier(args.node_supplier_url)

    try:
        # Use package resource path
        index_path = _package_resource_path("static/index.html")
        return FileResponse(index_path)
    except Exception:
        # Development environment uses local files
        return FileResponse("static/index.html")


# Helper function to fetch metrics from a single node
async def _fetch_node_metrics(node):
    """Fetch metrics from a single node"""
    try:
        # Check if port is a socket path
        is_socket_path = "/" in node["port"]

        if is_socket_path:
            # Use UDS transport for socket paths
            transport = httpx.AsyncHTTPTransport(uds=node["port"])
            # Use localhost as host
            url = "http://localhost/metrics"
            async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        else:
            # Build URL for regular port
            url = f"http://{node['host']}:{node['port']}/metrics"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
    except Exception as e:
        return f"# ERROR: Failed to get metrics from {node['name']}: {str(e)}\n"


@router.get("/metrics")
async def aggregated_metrics():
    """Aggregate /metrics from all leaf (child) nodes across every proxy.

    Previously only nodes under the ``local_proxy`` entry were
    aggregated, which silently returned nothing in supplier mode
    (where proxy names follow the ``proxy_<host>_<port>`` pattern).
    Now every proxy's ``nodes`` list is flattened.
    """
    if not target_nodes:
        return PlainTextResponse("# No nodes configured\n", status_code=404)

    nodes: list[dict] = []
    seen: set[str] = set()
    for proxy in target_nodes:
        for child in proxy.get("nodes", []) or []:
            # De-duplicate by name so the same leaf reported via
            # multiple proxies is only scraped once.
            name = child.get("name")
            if name and name in seen:
                continue
            if name:
                seen.add(name)
            nodes.append(child)

    if not nodes:
        return PlainTextResponse(
            "# No nodes available for metrics collection\n", status_code=404
        )

    metrics_results = await asyncio.gather(
        *[_fetch_node_metrics(node) for node in nodes]
    )

    # Combine all metrics with node name as comment header
    aggregated = ""
    for i, metrics in enumerate(metrics_results):
        node = nodes[i]
        aggregated += (
            f"# Metrics from node: {node['name']} ({node['host']}:{node['port']})\n"
        )
        aggregated += metrics
        aggregated += "\n\n"

    return PlainTextResponse(aggregated)


def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Flexible Proxy Server",
        description="HTTP proxy service supporting specified target hosts and ports",
    )
    app.include_router(router)

    # Get static file path (prefer package resources)
    try:
        static_path = _package_resource_path("static")
    except Exception:
        static_path = os.path.join(os.path.dirname(__file__), "static")

    # Mount static file service
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    return app


def main():
    global args
    parser = argparse.ArgumentParser(description="LMCache Cluster Monitoring Tool")
    parser.add_argument(
        "--port", type=int, default=8000, help="Service port, default 8000"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Bind host address, default 0.0.0.0"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specify configuration file path, default uses internal config.json",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Directly specify target nodes as a JSON string. "
        'Example: \'[{"name":"node1","host":"127.0.0.1","port":8001}]\'',
    )
    parser.add_argument(
        "--heartbeat-url",
        type=str,
        default=None,
        help="Heartbeat service URL, e.g.: http://example.com/heartbeat",
    )
    parser.add_argument(
        "--report-host",
        type=str,
        default=None,
        help="Host to report in heartbeat api_address. When set, bypasses "
        "get_local_ip() auto-detection. Useful for local dev or multi-NIC "
        "hosts where the auto-detected IP is not reachable from the "
        "discovery service side.",
    )
    parser.add_argument(
        "--heartbeat-initial-delay",
        type=int,
        default=0,
        help="Initial delay before starting heartbeat (seconds), default 0",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat interval (seconds), default 30",
    )
    parser.add_argument(
        "--node-supplier-url",
        type=str,
        default=None,
        help="URL to fetch node information from, e.g.: http://example.com/lmcache_infos",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["critical", "error", "warning", "warn", "info", "debug", "trace"],
        help="Uvicorn log level, default: warn",
    )
    parser.add_argument(
        "--no-http",
        action="store_true",
        default=False,
        help="Disable HTTP server startup (heartbeat still runs)",
    )

    args = parser.parse_args()

    # Initialize node configuration
    asyncio.run(initialize_nodes(args.node_supplier_url))

    app = create_app()
    print(f"Monitoring service running at http://{args.host}:{args.port}")
    print(f"Node management: http://{args.host}:{args.port}/static/index.html")

    # Start heartbeat service if URL is configured
    if args.heartbeat_url:
        # Set application configuration for heartbeat service
        heartbeat_service.set_app_config(
            args.host, args.port, target_nodes, args.report_host
        )

        print("Starting heartbeat service...")
        print(f"Heartbeat URL: {args.heartbeat_url}")
        print(f"Initial delay: {args.heartbeat_initial_delay}s")
        print(f"Interval: {args.heartbeat_interval}s")
        reported_host = args.report_host or heartbeat_service.get_local_ip()
        print(f"API Address: http://{reported_host}:{args.port}")
        print(f"Target nodes count: {len(target_nodes)}")

        heartbeat_service.start(
            args.heartbeat_url, args.heartbeat_initial_delay, args.heartbeat_interval
        )
    else:
        print("Heartbeat URL not configured, heartbeat disabled")

    if args.no_http:
        print("HTTP server disabled (--no-http), running heartbeat only")
        try:
            stop_event = threading.Event()
            stop_event.wait()
        finally:
            print("Shutting down application...")
            heartbeat_service.stop()
        return

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        # Stop heartbeat service when app closes
        print("Shutting down application...")
        heartbeat_service.stop()


if __name__ == "__main__":
    main()
