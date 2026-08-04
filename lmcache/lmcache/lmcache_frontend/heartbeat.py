# SPDX-License-Identifier: Apache-2.0
"""Heartbeat service module for periodic health reporting"""

# Standard
from datetime import datetime
from typing import Optional
import asyncio
import json
import os
import socket
import threading

# Third Party
import httpx


class HeartbeatService:
    """Periodic heartbeat service to report system status"""

    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.startup_time = datetime.now()
        self.app_host = "0.0.0.0"
        self.app_port = 8000
        self.target_nodes = []
        # When set, ``send_heartbeat`` reports this host verbatim instead
        # of relying on ``get_local_ip()`` auto-detection. Useful when the
        # detected IP is unreachable from the discovery service side
        # (e.g. local dev on macOS with VPN, multi-NIC hosts, etc.).
        self.report_host = None

    def get_local_ip(self) -> str:
        """Get the local IP address of the machine using multiple methods.

        Returns:
            A non-loopback IPv4 string when discoverable, otherwise
            ``"127.0.0.1"`` as the final fallback.
        """
        # Method 1: Try UDP socket connection (works when network is available)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            # Use a non-routable address - doesn't actually send packets
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127."):
                return ip
        except Exception:
            pass

        # Method 2: Try hostname resolution
        try:
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]
            for ip in ips:
                if not ip.startswith("127."):
                    return ip
        except Exception:
            pass

        # Method 3: Iterate through network interfaces (platform-specific)
        try:
            # Standard
            import platform
            import subprocess

            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["ipconfig", "getifaddr", "en0"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["hostname", "-I"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split()[0]
        except Exception:
            pass

        print("Failed to get local IP address. Falling back to loopback address.")
        return "127.0.0.1"

    def set_app_config(
        self,
        host: str,
        port: int,
        target_nodes: list,
        report_host: Optional[str] = None,
    ):
        """Set application configuration for heartbeat reporting.

        Args:
            host: Bind host of the local HTTP server (informational).
            port: Bind port of the local HTTP server; used as the port
                part of the reported ``api_address``.
            target_nodes: Downstream node list used for version probing.
            report_host: If provided (non-empty), used verbatim as the
                host part of the reported ``api_address``, bypassing
                the ``get_local_ip()`` auto-detection.
        """
        self.app_host = host
        self.app_port = port
        self.target_nodes = target_nodes
        self.report_host = report_host or None

    async def send_heartbeat(self, heartbeat_url: str):
        """Send heartbeat request"""
        try:
            host = self.report_host or self.get_local_ip()
            api_address = f"http://{host}:{self.app_port}"
            version = await self._get_version_from_nodes()
            if version:
                print(f"Got version from target nodes: {version}")

            # Calculate total children nodes across all proxies
            total_children = sum(
                len(proxy_node["nodes"]) for proxy_node in self.target_nodes
            )
            params = {
                "pid": os.getpid(),
                "api_address": api_address,
                "version": version or "1.0.0",
                "other_info": json.dumps(
                    {
                        "startup_time": self.startup_time.isoformat(),
                        "nodes_count": total_children,
                    }
                ),
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(heartbeat_url, params=params)
                response.raise_for_status()
                print(
                    f"Heartbeat sent successfully: "
                    f"{heartbeat_url} - Status: {response.status_code}"
                )
                return True
        except Exception as e:
            print(f"Heartbeat send failed: {heartbeat_url} - Error: {str(e)}")
            return False

    async def _get_version_from_nodes(self):
        """Get version from target nodes by querying each node directly."""
        if not self.target_nodes:
            return None

        for proxy_node in self.target_nodes:
            for node in proxy_node["nodes"]:
                try:
                    url = "http://%s:%s/version" % (
                        node["host"],
                        node["port"],
                    )
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(url)

                    if response.status_code == 200 and response.content:
                        content = response.content.decode("utf-8").strip()
                        if (content.startswith('"') and content.endswith('"')) or (
                            content.startswith("'") and content.endswith("'")
                        ):
                            content = content[1:-1]
                        return content

                except Exception as e:
                    print(
                        "Failed to get version from node %s: %s"
                        % (node["name"], str(e))
                    )
                    continue

        return None

    def worker(self, heartbeat_url: str, initial_delay: int, interval: int):
        """Heartbeat background thread worker function"""
        local_ip = self.get_local_ip()
        print(
            f"Heartbeat thread started - Local IP: {local_ip}, "
            f"Service URL: {heartbeat_url}"
        )
        print(f"Initial delay: {initial_delay}s, Interval: {interval}s")

        if initial_delay > 0:
            print(f"Waiting initial delay {initial_delay}s...")
            if self.stop_event.wait(initial_delay):
                print("Heartbeat thread stopped during initial delay")
                return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self.stop_event.is_set():
                try:
                    loop.run_until_complete(self.send_heartbeat(heartbeat_url))
                except Exception as e:
                    print(f"Heartbeat send exception: {str(e)}")

                if self.stop_event.wait(interval):
                    break
        finally:
            loop.close()
            print("Heartbeat thread stopped")

    def start(self, heartbeat_url: str, initial_delay: int = 0, interval: int = 30):
        """Start heartbeat thread"""
        if self.thread and self.thread.is_alive():
            print("Heartbeat thread is already running")
            return

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.worker,
            args=(heartbeat_url, initial_delay, interval),
            daemon=True,
        )
        self.thread.start()
        print("Heartbeat thread started")

    def stop(self):
        """Stop heartbeat thread"""
        if self.thread and self.thread.is_alive():
            print("Stopping heartbeat thread...")
            self.stop_event.set()
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                print("Warning: Heartbeat thread didn't stop within 5 seconds")
            else:
                print("Heartbeat thread stopped successfully")
        else:
            print("Heartbeat thread is not running")

    def status(self) -> dict:
        """Return a snapshot of the heartbeat thread state.

        Returns:
            Dict containing ``running`` (bool), ``local_ip``,
            ``startup_time`` and ``current_time`` (ISO-8601 strings).
        """
        is_running = self.thread and self.thread.is_alive()
        return {
            "running": is_running,
            "local_ip": self.get_local_ip(),
            "startup_time": self.startup_time.isoformat(),
            "current_time": datetime.now().isoformat(),
        }
