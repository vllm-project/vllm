# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import atexit
import os
import shutil
import subprocess
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class RuntimePluginLauncher:
    def __init__(
        self,
        config,
        role: str | None,
        worker_count: int,
        worker_id: int,
    ):
        self.config = config
        self.role = role
        self.worker_count = worker_count
        self.worker_id = worker_id
        self.plugin_processes: list[subprocess.Popen] = []
        # Register cleanup handler
        atexit.register(self.stop_plugins)

    def launch_plugins(self):
        """Launch all configured plugins"""
        if not self.config.runtime_plugin_locations:
            return

        for loc in self.config.runtime_plugin_locations:
            self._launch_plugins(loc)

    def _launch_plugins(self, loc: str):
        """Launch plugins from specified location"""
        path = Path(loc)
        if not path.exists():
            logger.warning("Runtime plugin location %s does not exist", loc)
            return

        files = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            # Recursively find all .py and .sh files
            for ext in ["*.py", "*.sh"]:
                files.extend(path.rglob(ext))

        for file in files:
            self._launch_plugin(file)

    def _should_skip_plugin(self, file: Path, parts: list[str]) -> bool:
        """Determine if plugin should be skipped based on role/worker ID.

        The naming convention is ROLE_WORKERID_NAME (e.g. server_0_foo.py).
        When role is None (MP mode), this convention does not apply,
        so both role and worker ID filtering are skipped entirely.
        """
        if len(parts) < 2:
            return False

        # When role is None, skip all role/worker-based filtering
        # (e.g. MP mode has no role concept)
        if self.role is None:
            return False

        plugin_role = parts[0].upper()
        if plugin_role != "ALL" and plugin_role != self.role.upper():
            logger.info(
                "Skipping %s: requires role %s",
                file,
                plugin_role,
            )
            return True

        # Check worker ID match (parts[1] in ROLE_WORKERID_NAME)
        if len(parts) > 2 and parts[1].isdigit():
            plugin_worker_id = int(parts[1])
            if plugin_worker_id != self.worker_id:
                logger.info(
                    "worker %d is skipping plugin %s, which is only for worker ID %d",
                    self.worker_id,
                    file,
                    plugin_worker_id,
                )
                return True

        return False

    def _launch_plugin(self, file: Path):
        """Launch a plugin"""
        try:
            filename = file.stem.lower()
            parts = filename.split("_")

            if self._should_skip_plugin(file, parts):
                return

            # Get interpreter from first line (shebang)
            interpreter = self._get_interpreter(file)

            # Pass role and config as environment variables
            env = os.environ.copy()
            role_str = self.role or ""
            env["LMCACHE_RUNTIME_PLUGIN_ROLE"] = role_str
            env["LMCACHE_RUNTIME_PLUGIN_CONFIG"] = self.config.to_json()
            env["LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT"] = str(self.worker_count)
            env["LMCACHE_RUNTIME_PLUGIN_WORKER_ID"] = str(self.worker_id)

            # TODO: For backwards compatibility, remove when applicable
            env["LMCACHE_PLUGIN_ROLE"] = role_str
            env["LMCACHE_PLUGIN_CONFIG"] = self.config.to_json()
            env["LMCACHE_PLUGIN_WORKER_COUNT"] = str(self.worker_count)
            env["LMCACHE_PLUGIN_WORKER_ID"] = str(self.worker_id)

            # Force line-buffered stdout for Python sub-processes
            # so that output is captured in real-time rather than
            # being held in the block buffer until exit.
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                [interpreter, str(file)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.plugin_processes.append(proc)
            logger.info(
                "Launched runtime plugin: %s with %s",
                file,
                interpreter,
            )

            # Start thread to capture output continuously
            threading.Thread(
                target=self._capture_plugin_output, args=(proc, str(file)), daemon=True
            ).start()
        except Exception as e:
            logger.error("Failed to launch plugin %s: %s", file, e)

    def _get_interpreter(self, file: Path) -> str:
        """Get interpreter from first line comment"""
        interpreters = []
        try:
            with open(file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    # Extract interpreter
                    interpreter_str = first_line[2:].strip()
                    interpreters.append(interpreter_str)
        except Exception as e:
            logger.error(
                "Error reading interpreter from runtime plugin file %s - "
                "using default interpreters: %s",
                file,
                e,
            )
            pass

        # Fallback to default interpreters
        if file.suffix == ".py":
            interpreters.append("python")
            interpreters.append("python3")
        elif file.suffix == ".sh":
            interpreters.append("bash")
        else:
            raise ValueError(f"Plugin type {file.suffix} not supported ")

        # Try each interpreter until we find one that exists
        for interpreter in interpreters:
            interpreter = interpreter.strip()
            resolved_interpreter = shutil.which(interpreter)
            if resolved_interpreter:
                return resolved_interpreter

        raise ValueError(f"No valid interpreter found for {file} from {interpreters}")

    def _capture_plugin_output(self, proc: subprocess.Popen, plugin_name: str):
        """Continuously capture and log plugin output"""
        try:
            assert proc.stdout is not None, (
                "The runtime plugin subprocess does not have stdout"
            )
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                logger.info("[%s] %s", plugin_name, line.strip())

            proc.wait()
            logger.info(
                "Runtime plugin %s exited with code %d",
                plugin_name,
                proc.returncode,
            )
        except Exception as e:
            logger.error("Error capturing output for %s: %s", plugin_name, e)

    def stop_plugins(self):
        """Terminate all plugin processes.

        Note: This method may be called via atexit during interpreter shutdown,
        when the logging system may already be closed. We avoid using logger
        here to prevent "I/O operation on closed file" errors.
        """
        for proc in self.plugin_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                # Silently ignore errors during shutdown
                pass
