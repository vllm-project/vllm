# SPDX-License-Identifier: Apache-2.0
import os

from entrypoints.entrypoint_base import EntrypointBase
from entrypoints.script_generator import ScriptGenerator
from server.vllm_autocalc import VarsGenerator


class EntrypointServer(EntrypointBase):

    def run(self):
        self._prepare_conf("server/server_defaults.yaml")

        print("[INFO] Starting container in server mode.")
        for key, value in self.config_envs.items():
            os.environ[str(key)] = str(value)
        variables = VarsGenerator(
            defaults_path="server/server_defaults.yaml",
            varlist_conf_path="server/server_user.env",
            model_def_settings_path="server/settings_vllm.csv",
        ).calculate_variables()
        ScriptGenerator(
            template_script_path="templates/template_vllm_server.sh",
            output_script_path="vllm_server.sh",
            variables=variables,
            log_dir="logs",
        ).create_and_run()
