# SPDX-License-Identifier: Apache-2.0
import os

from entrypoints.entrypoint_base import EntrypointBase
from entrypoints.script_generator import ScriptGenerator


class EntrypointBenchmark(EntrypointBase):

    def _update_benchmark_envs_from_user_vars(self):
        env_file = "benchmark/benchmark_user.env"
        env_vars = []

        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _ = line.split("=", 1)
                        key = key.strip()
                    else:
                        key = line
                    if key:
                        env_vars.append(key)
        except FileNotFoundError:
            print(f"[WARNING] .env file '{env_file}' not found. "
                  "No user-defined variables loaded from .env.")

        for param in env_vars:
            if os.environ.get(param) is not None:
                try:
                    self.config_envs[param] = eval(os.environ[param])
                except Exception:
                    self.config_envs[param] = os.environ[param]
                print(
                    f"[INFO] Overwriting {param} with value from environment: "
                    f"{self.config_envs[param]}")
        if not env_vars:
            print(f"[WARNING] No variables loaded from '{env_file}'.")

    def run(self):
        self._prepare_conf("benchmark/benchmark_defaults.yaml")

        print("[INFO] Starting container in benchmark mode.")
        self._update_benchmark_envs_from_user_vars()
        ScriptGenerator(
            template_script_path="templates/template_vllm_benchmark.sh",
            output_script_path="vllm_benchmark.sh",
            variables=self.config_envs,
            log_dir="logs",
        ).create_and_run()
