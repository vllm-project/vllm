# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys

import yaml
from entrypoints.script_generator import ScriptGenerator
from server_autoconfig.vllm_autocalc import VarsGenerator


class EntrypointMain:
    """
    EntrypointMain dispatches to the appropriate Entrypoint mode:
    - 'server': runs Entrypoint in server mode (default)
    - 'benchmark': runs Entrypoint in benchmark mode (stub for extension)
    - 'test': keeps the container alive for debugging
    Optional: --config_file and --config_name must be both provided or neither.
    """

    def __init__(self, mode="server", config_file=None, config_name=None):
        self.mode = mode
        self.config_file = config_file
        self.config_name = config_name
        self.config_envs = {}
        if (self.config_file is not None) ^ (self.config_name is not None):
            print(
                "[ERROR] Both --config-file and --config-name must be "
                "provided together, or neither.",
                file=sys.stderr)
            sys.exit(1)

    def _load_env_from_config(self):
        try:
            with open(self.config_file) as f:
                config = yaml.safe_load(f)
                section = config.get(self.config_name)
                if section is None:
                    print(
                        f"[ERROR] Section '{self.config_name}' not found in "
                        f"'{self.config_file}'.",
                        file=sys.stderr)
                    sys.exit(1)
                if not isinstance(section, dict):
                    print(
                        f"[ERROR] Section '{self.config_name}' is not a "
                        f"dictionary in '{self.config_file}'.",
                        file=sys.stderr)
                    sys.exit(1)
                self.config_envs = section
                print(f"[INFO] Loaded configuration from file: "
                      f"{self.config_file}, section: {self.config_name}")
                print("[INFO] The following parameters and values were loaded "
                      "from the config file:")
                for key, value in self.config_envs.items():
                    print(f"    {key}: {value}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
            sys.exit(1)

    def run(self):
        if self.config_file and self.config_name:
            self._load_env_from_config()

        if self.mode == "server":
            print("[INFO] Starting container in server mode.")
            # VarsGenerator will read variables from the environment
            for key, value in self.config_envs.items():
                os.environ[str(key)] = str(value)
            variables = VarsGenerator(
                defaults_path="server_autoconfig/defaults.yaml",
                varlist_conf_path="server_autoconfig/varlist_conf.yaml",
                model_def_settings_path=("server_autoconfig/settings_vllm.csv"
                                         )).calculate_variables()
            ScriptGenerator(
                template_script_path="templates/template_vllm_server.sh",
                output_script_path="vllm_server.sh",
                variables=variables,
                log_dir="logs").create_and_run()
        elif self.mode == "benchmark":
            print("[INFO] Starting container in benchmark mode.")
            ScriptGenerator(
                template_script_path="templates/template_vllm_benchmark.sh",
                output_script_path="vllm_benchmark.sh",
                variables=self.config_envs,
                log_dir="logs").create_and_run()
        elif self.mode == "test":
            print("[INFO] Test mode: keeping container active. "
                  "Press Ctrl+C to exit.")
            try:
                while True:
                    import time
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Exiting test mode.")
                sys.exit(0)
        else:
            print(f"[ERROR] Unknown mode '{self.mode}'. Use 'server', "
                  "'benchmark' or 'test'.")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EntrypointMain for vllm docker")
    parser.add_argument("mode",
                        nargs="?",
                        default="server",
                        choices=["server", "benchmark", "test"],
                        help="Mode to run: server, benchmark, or test")
    parser.add_argument("--config-file", type=str, help="Path to config file")
    parser.add_argument("--config-name",
                        type=str,
                        help="Config name in the config file")
    args = parser.parse_args()

    EntrypointMain(mode=args.mode,
                   config_file=args.config_file,
                   config_name=args.config_name).run()
