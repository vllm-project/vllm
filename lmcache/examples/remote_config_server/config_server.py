#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Reference implementation of a remote config server for LMCache.

This is a simple Flask-based config server that demonstrates the protocol
between LMCache workers and a centralized configuration service.

Usage:
    python config_server.py [--port PORT] [--host HOST]

The server exposes a single endpoint:
    POST /config?appId=<app_id>

See README.md for the full protocol specification.
"""

# Standard
from typing import Any
import argparse
import logging

# Third Party
from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example configuration database
# In production, this could be fetched from a database, config file, or other service
CONFIG_DATABASE: dict[str, list[dict[str, Any]]] = {
    # Default configs applied to all apps
    "default": [
        {"key": "chunk_size", "value": 512, "override": False},
        {"key": "max_local_cpu_size", "value": 1, "override": False},
    ],
    # App-specific configs
    "test-app": [
        {"key": "chunk_size", "value": 1024, "override": False},
        {"key": "max_local_cpu_size", "value": 2, "override": False},
        {"key": "lmcache_worker_heartbeat_time", "value": "14", "override": True},
        {
            "key": "extra_config",
            "value": '{"internal_api_server_access_log": true, '
            '"internal_api_server_log_level":"info", '
            '"save_only_first_rank": true, '
            '"first_rank_max_local_cpu_size": 1.1}',
            "override": True,
        },
    ],
    "production-app": [
        {"key": "chunk_size", "value": 2048, "override": True},
        {"key": "max_local_cpu_size", "value": 100, "override": True},
    ],
}


def get_configs_for_app(
    app_id: str | None,
    current_config: dict[str, Any],
    env_variables: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Determine which configs to return based on app_id and current state.

    This is where you would implement your custom logic, such as:
    - Looking up configs from a database
    - Applying rules based on environment variables
    - A/B testing different configurations
    """
    # Use dict to deduplicate by key, later entries override earlier ones
    config_map: dict[str, dict[str, Any]] = {}

    # Start with default configs
    if "default" in CONFIG_DATABASE:
        for item in CONFIG_DATABASE["default"]:
            config_map[item["key"]] = item

    # Add app-specific configs (override defaults with same keys)
    if app_id and app_id in CONFIG_DATABASE:
        for item in CONFIG_DATABASE[app_id]:
            config_map[item["key"]] = item

    return list(config_map.values())


@app.route("/config", methods=["GET", "POST"])
def get_config():
    """
    Handle config requests from LMCache workers.

    Expected request body:
    {
        "current_config": {...},
        "env_variables": {...}
    }

    Query parameters:
    - appId: Optional application identifier
    """
    try:
        # Parse request - supports both JSON body and form data
        data = request.get_json(silent=True)
        if data is None:
            # Fallback to form data or query parameters
            data = request.form.to_dict() or request.args.to_dict()

        current_config = data.get("current_config", {})
        env_variables = data.get("env_variables", {})
        app_id = request.args.get("appId") or data.get("appId")

        logger.info(
            "Config request received - app_id: %s, current_config keys: %s",
            app_id,
            list(current_config.keys()),
        )

        # Get configs for this app
        configs = get_configs_for_app(app_id, current_config, env_variables)

        response = {"configs": configs}

        logger.info("Returning %d config items for app_id: %s", len(configs), app_id)
        return jsonify(response)

    except Exception as e:
        logger.exception("Error processing config request")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


def main():
    parser = argparse.ArgumentParser(
        description="LMCache Remote Config Server Reference Implementation"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logger.info("Starting config server on %s:%d", args.host, args.port)
    logger.info("Available app configs: %s", list(CONFIG_DATABASE.keys()))

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
