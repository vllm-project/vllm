# SPDX-License-Identifier: Apache-2.0
import os
import sys
from abc import ABC, abstractmethod

import yaml


class EntrypointBase(ABC):
    """
    Abstract base class for Entrypoint implementations.
    """

    def __init__(self, config_file=None, config_name=None):
        self.config_file = config_file
        self.config_name = config_name
        self.config_envs = {}
        if (self.config_file is not None) ^ (self.config_name is not None):
            print(
                "[ERROR] Both --config-file and --config-name must be "
                "provided together, or neither.",
                file=sys.stderr)
            sys.exit(1)

    @abstractmethod
    def run(self):
        pass

    def _load_env_from_defaults(self, defaults_file):
        try:
            with open(defaults_file) as f:
                config = yaml.safe_load(f)
                found = False
                for section_name, section in config.items():
                    if section_name.startswith("model_") and isinstance(
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

    def _prepare_conf(self, defaults_path):
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

        self._load_env_from_defaults(defaults_path)

        if model_conf:
            self.config_envs.update(model_conf)
        return model_conf
