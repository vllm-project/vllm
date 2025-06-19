#!/usr/bin/env python3
"""
Port Allocation Utility

A small utility that generates consistent port numbers based on username and default port
to avoid port collisions during development.
"""

import argparse
import getpass
import hashlib


def allocate_port(base_port,
                  username=None,
                  project_name=None,
                  port_range=None):
    """
    Allocate a port based on username and base port.
    
    Args:
        base_port (int): The default port number for the service
        username (str, optional): Username to use for hashing. Defaults to current user.
        project_name (str, optional): Project name to make ports unique per project
        port_range (tuple, optional): Range of valid ports (min, max). Defaults to (1024, 65535).
    
    Returns:
        int: A port number derived from hashing the username and base port
    """
    if not username:
        username = getpass.getuser()

    if not port_range:
        port_range = (1024, 65535)

    min_port, max_port = port_range
    available_range = max_port - min_port

    # Create hash input from username, base_port and optional project_name
    hash_input = f"{username}:{base_port}"
    if project_name:
        hash_input = f"{project_name}:{hash_input}"

    # Create a hash and convert to an integer in our port range
    hash_obj = hashlib.md5(hash_input.encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    # Generate a port within the valid range
    port_offset = hash_int % available_range
    allocated_port = min_port + port_offset

    # Check if it's too close to the base_port (within 10)
    if abs(allocated_port - base_port) < 10:
        # Add a small offset to avoid collisions with the default port
        allocated_port = (allocated_port + 100) % available_range + min_port

    return allocated_port


def main():
    parser = argparse.ArgumentParser(
        description='Allocate a consistent port based on username and base port'
    )
    parser.add_argument('base_port',
                        type=int,
                        help='The default port number for the service')
    parser.add_argument('--username',
                        '-u',
                        help='Username to use (defaults to current user)')
    parser.add_argument('--project',
                        '-p',
                        help='Project name to make ports unique per project')
    parser.add_argument('--env-var',
                        '-e',
                        help='Output as export ENV_VAR=port')
    parser.add_argument('--min-port',
                        type=int,
                        default=1024,
                        help='Minimum port number')
    parser.add_argument('--max-port',
                        type=int,
                        default=65535,
                        help='Maximum port number')

    args = parser.parse_args()

    port = allocate_port(args.base_port,
                         username=args.username,
                         project_name=args.project,
                         port_range=(args.min_port, args.max_port))

    if args.env_var:
        print(f"export {args.env_var}={port}")
    else:
        print(port)


if __name__ == "__main__":
    main()
