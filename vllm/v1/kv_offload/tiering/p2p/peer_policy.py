# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IP-based admission policy for P2P control connections."""

import ipaddress
import socket

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def _resolve_host(host: str) -> list[IPAddress]:
    if not host or host != host.strip() or any(c in host for c in "%/\\\x00"):
        raise ValueError(f"Invalid P2P host: {host!r}")
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        if any(c in host for c in ":[]"):
            raise ValueError(f"Invalid P2P host: {host!r}") from None
    try:
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot resolve P2P host: {host!r}") from exc
    if not addresses:
        raise ValueError(f"No addresses for P2P host: {host!r}")
    # Match libzmq's default preference for IPv4 hostname listeners.
    return sorted(addresses, key=lambda addr: (addr.version, int(addr)))


class PeerAllowlist:
    def __init__(self, allowed_peers: list[str] | None = None) -> None:
        if allowed_peers is None:
            allowed_peers = ["127.0.0.0/8", "::1"]
        if not isinstance(allowed_peers, list) or not all(
            isinstance(peer, str) and peer for peer in allowed_peers
        ):
            raise ValueError("allowed_peers must be a list of IPs, hostnames or CIDRs")
        self._networks: list[IPNetwork] = []
        for peer in allowed_peers:
            try:
                if "%" in peer:
                    raise ValueError("Scoped addresses are not supported")
                try:
                    network = ipaddress.ip_network(peer)
                except ValueError:
                    self._networks.extend(
                        ipaddress.ip_network(addr) for addr in _resolve_host(peer)
                    )
                else:
                    self._networks.append(network)
            except ValueError as exc:
                raise ValueError(f"Invalid allowed_peers entry: {peer!r}") from exc

    def allows_address(self, address: str) -> bool:
        """Check a transport-supplied numeric source address without DNS."""
        try:
            addr = ipaddress.ip_address(address)
        except ValueError:
            return False
        mapped = addr.ipv4_mapped if isinstance(addr, ipaddress.IPv6Address) else None
        return any(
            addr in network or (mapped is not None and mapped in network)
            for network in self._networks
        )

    def resolve_peer(self, peer_id: str) -> tuple[str, int]:
        """Authorize every DNS answer and return a numeric endpoint to dial.

        Configured hostnames are resolved at startup. Request hostnames are
        resolved at admission; dialing the returned IP prevents DNS rebinding
        between authorization and the transport's asynchronous connect.
        """
        host, port_str = peer_id.rsplit(":", 1)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not port_str.isascii() or not port_str.isdecimal():
            raise ValueError(f"Invalid P2P port: {port_str!r}")
        port = int(port_str)
        if not 0 < port < 65536:
            raise ValueError(f"Invalid P2P port: {port}")
        addresses = _resolve_host(host)
        if not all(self.allows_address(str(addr)) for addr in addresses):
            raise ValueError(f"P2P peer is not in allowed_peers: {peer_id!r}")
        return str(addresses[0]), port
