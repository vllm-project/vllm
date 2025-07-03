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

    def _load_env_from_defaults(self):
        """
        Loads default environment variables from a YAML file based on the mode.
        For each section starting with 'defaults_', if model is in the section's
        'MODELS' list, loads the environment variables from that section.
        If no section matches, loads nothing.
        If the file does not exist, it returns an empty dictionary.
        """
        defaults_file = ("server_configurations/defaults.yaml"
                         if self.mode == "server" else
                         "benchmark_configurations/defaults.yaml")
        try:
            with open(defaults_file) as f:
                config = yaml.safe_load(f)
                found = False
                for section_name, section in config.items():
                    if section_name.startswith("defaults_") and isinstance(
                            section, dict):
                        models = section.get("MODELS", [])
                        if (isinstance(models, list)
                                and self.config_envs.get("MODEL") in models):
                            env_vars = {
                                k: v
                                for k, v in section.items() if k != "MODELS"
                            }
                            self.config_envs.update(env_vars)
                            print(
                                f"[INFO] Loaded default configuration section "
                                f"'{section_name}' for model "
                                f"'{self.config_envs.get('MODEL')}' from file: "
                                f"{defaults_file}")
                            for key, value in env_vars.items():
                                print(f"    {key}: {value}")
                            found = True
                if not found:
                    print(f"[WARNING] No defaults section found for model "
                          f"'{self.config_envs.get('MODEL')}' in "
                          f"'{defaults_file}'.")
        except FileNotFoundError:
            print(f"[WARNING] Defaults file '{defaults_file}' not found. "
                  "No defaults loaded.")
        except Exception as e:
            print(
                f"[ERROR] Failed to load defaults: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    def _load_env_from_config_file(self):
        """
        Loads a specific configuration section from a YAML file and updates the
        current environment configuration with the values from that section.
        If a key already exists (e.g., from defaults), it will be overwritten
        by the value from the file. Exits the program with an error message if
        the section is missing or invalid, or if the file cannot be read.

        Raises:
            SystemExit: If the configuration file or section is missing,
            invalid, or cannot be loaded.
        """
        try:
            with open(self.config_file) as f:
                config = yaml.safe_load(f)
                section = config.get(self.config_name)
                if section is None or not isinstance(section, dict):
                    print(
                        f"[ERROR] Section '{self.config_name}' not found or "
                        f"is not a dictionary in '{self.config_file}'.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(f"[INFO] Loaded configuration section "
                      f"'{self.config_name}' from file: {self.config_file}")
                for key, value in section.items():
                    print(f"    {key}: {value}")
                return section
        except Exception as e:
            print(
                f"[ERROR] Failed to load config: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    def _update_benchmark_envs_from_user_vars(self):
        """
        Loads a list of variable names from a YAML file and, for each variable
        present in the current environment, updates the internal configuration
        dictionary with the environment value. If the YAML file is missing or
        empty, no variables are updated.
        """
        user_vars_file = "benchmark_configurations/user_vars.yaml"
        try:
            with open(user_vars_file) as f:
                user_vars = yaml.safe_load(f)
                if user_vars and isinstance(user_vars, dict):
                    variables = user_vars.get("variables", [])
                    for var in variables:
                        if var in os.environ:
                            self.config_envs[var] = os.environ[var]
                            print(f"[INFO] Overwriting {var} with value from "
                                  f"environment: {self.config_envs[var]}")
                else:
                    print(f"[WARNING] No user-defined variables found in "
                          f"'{user_vars_file}'.")
        except FileNotFoundError:
            print(
                f"[WARNING] User variables file '{user_vars_file}' not found. "
                "No user-defined variables loaded.")
        except Exception as e:
            print(
                f"[ERROR] Failed to load user-defined variables: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    def run(self):

        if self.mode == "test":
            print("[INFO] Test mode: keeping container active. "
                  "Press Ctrl+C to exit.")
            try:
                while True:
                    import time
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Exiting test mode.")
                sys.exit(0)

        model_conf = {}
        if self.config_file and self.config_name:
            model_conf = self._load_env_from_config_file()
            if "MODEL" in model_conf:
                self.config_envs["MODEL"] = model_conf["MODEL"]

        env_model = os.environ.get("MODEL")
        if env_model:
            self.config_envs["MODEL"] = env_model

        if not self.config_envs.get("MODEL"):
            print("[ERROR] MODEL is not set. Exiting.", file=sys.stderr)
            sys.exit(1)

        self._load_env_from_defaults()

        if model_conf:
            self.config_envs.update(model_conf)

        if self.mode == "server":
            print("[INFO] Starting container in server mode.")
            for key, value in self.config_envs.items():
                os.environ[str(key)] = str(value)
            variables = VarsGenerator(
                defaults_path="server_autoconfig/defaults.yaml",
                varlist_conf_path="server_autoconfig/varlist_conf.yaml",
                model_def_settings_path=(
                    "server_autoconfig/settings_vllm.csv"),
            ).calculate_variables()
            ScriptGenerator(
                template_script_path="templates/template_vllm_server.sh",
                output_script_path="vllm_server.sh",
                variables=variables,
                log_dir="logs",
            ).create_and_run()
        elif self.mode == "benchmark":
            print("[INFO] Starting container in benchmark mode.")
            self._update_benchmark_envs_from_user_vars()
            ScriptGenerator(
                template_script_path="templates/template_vllm_benchmark.sh",
                output_script_path="vllm_benchmark.sh",
                variables=self.config_envs,
                log_dir="logs",
            ).create_and_run()
        else:
            print(f"[ERROR] Unknown mode '{self.mode}'. Use 'server', "
                  "'benchmark' or 'test'.")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EntrypointMain for vllm docker")
    parser.add_argument(
        "mode",
        nargs="?",
        default="server",
        choices=["server", "benchmark", "test"],
        help="Mode to run: server, benchmark, or test",
    )
    parser.add_argument("--config-file", type=str, help="Path to config file")
    parser.add_argument("--config-name",
                        type=str,
                        help="Config name in the config file")
    args = parser.parse_args()

    EntrypointMain(
        mode=args.mode,
        config_file=args.config_file,
        config_name=args.config_name,
    ).run()
