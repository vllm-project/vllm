import errno
import glob
import socket

import psutil
import ray


def ray_run_on_every_nodes(func, *args, **kwargs):
    unique_ips = set(
        [node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]])
    futures = [
        ray.remote(func).options(resources={
            f"node:{ip}": 0.01
        }).remote(*args, **kwargs) for ip in unique_ips
    ]
    return ray.get(futures)



def find_node_ip(address: str = "8.8.8.8:53") -> str:
    """
    NOTE: this implementation is adapted from ray-project/ray, see:
    https://github.com/ray-project/ray/blob/aa2dede7f795d21407deebf4cefc61fd00e68e84/python/ray/_private/services.py#L637

    IP address by which the local node can be reached *from* the `address`.

    Args:
        address: The IP address and port of any known live service on the
            network you care about.

    Returns:
        The IP address by which the local node can be reached from the address.
    """
    ip_address, port = address.split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet
        # connection.
        s.connect((ip_address, int(port)))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()

    return node_ip_address


def find_free_port(address: str = "") -> str:
    """
    find one free port

    Returns:
        port
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def find_interface_by_ip(ip_address):
    """
    Find the network interface name associated with the given IP address.

    Args:
        ip_address (str): The IP address to look up (e.g., "192.168.1.100").

    Returns:
        str: The name of the matching network interface (e.g., "eth0" or "wlan0"), or None if not found.
    """
    interfaces = psutil.net_if_addrs()

    for interface_name, addresses in interfaces.items():
        for address in addresses:
            if address.family == socket.AF_INET and address.address == ip_address:
                return interface_name

    # Return None if no match is found
    return None


def find_ip_by_interface(interface_name: str):
    """
    Find the IP address associated with the given network interface name.

    Args:
        interface_name (str): The name of the network interface (e.g., "eth0", "wlan0").

    Returns:
        str: The IP address associated with the interface, or None if not found.
    """
    # Get all network interfaces and their addresses
    interfaces = psutil.net_if_addrs()

    # Check if the interface exists
    if interface_name not in interfaces:
        return None

    # Determine the address family (IPv4 or IPv6)
    family = socket.AF_INET  # IPv6: 10 (AF_INET6), IPv4: 2 (AF_INET)

    # Iterate through the addresses of the specified interface
    for address in interfaces[interface_name]:
        if address.family == family:
            return address.address

    # Return None if no matching IP address is found
    return None
