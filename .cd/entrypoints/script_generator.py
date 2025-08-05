# SPDX-License-Identifier: Apache-2.0
import os


class ScriptGenerator:

    def __init__(self,
                 template_script_path,
                 output_script_path,
                 variables,
                 log_dir="logs"):
        self.template_script_path = template_script_path
        self.output_script_path = output_script_path
        self.variables = variables
        self.log_dir = log_dir
        self.log_file = os.path.join(
            self.log_dir,
            f"{os.path.splitext(os.path.basename(self.output_script_path))[0]}.log"
        )

    def generate_script(self, vars_dict):
        """
        Generate the script from a template, 
        replacing placeholders with environment variables.
        """
        with open(self.template_script_path) as f:
            template = f.read()
        export_lines = "\n".join(
            [f"export {k}={v}" for k, v in vars_dict.items()])
        script_content = template.replace("#@VARS", export_lines)
        with open(self.output_script_path, 'w') as f:
            f.write(script_content)

    def make_script_executable(self):
        """
        Make the output script executable.
        """
        os.chmod(self.output_script_path, 0o755)

    def print_script(self):
        """
        Print the generated script for debugging.
        """
        print(f"\n===== Generated {self.output_script_path} =====")
        with open(self.output_script_path) as f:
            print(f.read())
        print("====================================\n")

    def create_and_run(self):
        self.generate_script(self.variables)
        self.make_script_executable()
        self.print_script()

        # Run the generated script and redirect output to log file
        print(f"Starting script, logging to {self.log_file}")
        os.makedirs(self.log_dir, exist_ok=True)
        os.execvp("bash", ["bash", self.output_script_path])
