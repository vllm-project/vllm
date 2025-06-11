#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

from vllm_autocalc import VarsGenerator


class Entrypoint:

    def __init__(self):
        # Paths to config files
        self.defaults_path = 'settings/defaults.yaml'
        self.varlist_conf_path = 'settings/varlist_conf.yaml'
        self.model_def_settings_path = 'settings/settings_vllm.csv'
        self.template_server_path = 'settings/template_vllm_server.sh'
        self.output_server_path = 'vllm_server.sh'
        self.log_dir = 'logs'
        self.log_file = os.path.join(self.log_dir, 'vllm_server.log')

    def generate_server_script(self, template_path, output_path, vars_dict):
        """
        Generate the server script from a template, 
        replacing placeholders with environment variables.
        """
        with open(template_path) as f:
            template = f.read()
        export_lines = "\n".join(
            [f"export {k}={v}" for k, v in vars_dict.items()])
        script_content = template.replace("#@VARS", export_lines)
        with open(output_path, 'w') as f:
            f.write(script_content)

    def make_scripts_executable(self):
        """
        Make all .sh scripts in the current directory executable.
        """
        for fname in os.listdir('.'):
            if fname.endswith('.sh'):
                os.chmod(fname, 0o755)

    def print_server_script(self):
        """
        Print the generated server script for debugging.
        """
        print("\n===== Generated vllm_server.sh =====")
        with open(self.output_server_path) as f:
            print(f.read())
        print("====================================\n")

    def prepare_log_file(self):
        """
        Ensure log directory and file exist.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file, 'a'):
            os.utime(self.log_file, None)

    def run(self):
        # Calculate variables using VarsGenerator
        vg = VarsGenerator(
            defaults_path=self.defaults_path,
            varlist_conf_path=self.varlist_conf_path,
            model_def_settings_path=self.model_def_settings_path)
        # Calculate variables based on the configuration
        variables = vg.calculate_variables()

        # Generate the server script from template
        self.generate_server_script(self.template_server_path,
                                    self.output_server_path, variables)

        # Make all .sh scripts executable
        self.make_scripts_executable()

        # Prepare log file
        self.prepare_log_file()

        # If the first argument is 'test', keep the container active
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("Test mode: keeping container active. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Exiting test mode.")
                sys.exit(0)

        # Print generated server script for debugging
        self.print_server_script()

        # Run the server script and redirect output to log file
        print(f"Starting server, logging to {self.log_file}")
        os.execvp("bash", ["bash", self.output_server_path])


if __name__ == "__main__":
    Entrypoint().run()
