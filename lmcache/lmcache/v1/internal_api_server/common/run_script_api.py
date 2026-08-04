# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, List
import importlib

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_allowed_imports(request: Request) -> List[str]:
    """Extract script_allowed_imports from either inProcess or mp mode."""
    adapter = getattr(request.app.state, "lmcache_adapter", None)
    if adapter is not None:
        return getattr(adapter.config, "script_allowed_imports", None) or []
    configs = getattr(request.app.state, "configs", None)
    if isinstance(configs, dict):
        mp_cfg = configs.get("mp")
        return getattr(mp_cfg, "script_allowed_imports", None) or []
    return []


@router.post("/run_script")
async def run_script(request: Request):
    form_data = await request.form()
    script_file = form_data.get("script")

    if not script_file or not hasattr(script_file, "file"):
        return PlainTextResponse("No script file provided", status_code=400)

    script_content = await script_file.read()

    try:
        allowed_imports = _get_allowed_imports(request)

        # Pre-import allowed modules
        allowed_modules = {}
        for module_name in allowed_imports:
            try:
                module = importlib.import_module(module_name)
                allowed_modules[module_name] = module
                logger.info("Imported allowed module: %s", module_name)
            except ImportError as e:
                logger.warning("Failed to import module %s: %s", module_name, e)

        # Create custom __import__ function that only allows configured modules
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in allowed_modules:
                return allowed_modules[name]
            raise ImportError(f"Import of '{name}' is not allowed")

        restricted_globals = {
            "__builtins__": {
                "print": print,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "__import__": restricted_import,
            },
            "app": request.app,
        }

        restricted_locals: dict[str, Any] = {}

        exec(script_content, restricted_globals, restricted_locals)

        result = restricted_locals.get("result", "Script executed successfully")
        return PlainTextResponse(str(result), media_type="text/plain")

    except Exception as e:
        return PlainTextResponse(f"Error executing script: {str(e)}", status_code=500)
