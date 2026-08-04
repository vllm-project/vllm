# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List


def get_all_server_infos(config, worker_count) -> List[Dict[str, str]]:
    """
    Generate a list of server information (scheduler and workers) based on the config.

    Args:
        config: The configuration object containing server details.
        worker_count: The number of worker servers.

    Returns:
        List[Dict[str, str]]: A JSON list with server information.
    """
    servers = []
    include_index_list = getattr(config, "internal_api_server_include_index_list", None)
    socket_path_prefix = getattr(config, "internal_api_server_socket_path_prefix", None)

    # Add scheduler info (index 0)
    if include_index_list is None or 0 in include_index_list:
        port = config.internal_api_server_port_start
        server_info = {
            "name": f"{config.lmcache_instance_id}_scheduler",
            "host": config.internal_api_server_host,
            "port": f"{socket_path_prefix}_{port}" if socket_path_prefix else port,
        }
        servers.append(server_info)

    # Add workers info (index 1 to worker_count)
    for worker_id in range(worker_count):
        port_offset = 1 + worker_id
        if include_index_list is None or port_offset in include_index_list:
            port = config.internal_api_server_port_start + port_offset
            server_info = {
                "name": f"{config.lmcache_instance_id}_worker{worker_id}",
                "host": config.internal_api_server_host,
                "port": f"{socket_path_prefix}_{port}" if socket_path_prefix else port,
            }
            servers.append(server_info)

    return servers
