#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os

import pandas as pd
import yaml
from server_autoconfig.vllm_autocalc_rules import PARAM_CALC_FUNCS


class VarsGenerator:

    def __init__(self,
                 defaults_path,
                 varlist_conf_path,
                 model_def_settings_path,
                 file_output_vars='server_vars.txt'):
        """
        Initialize VarsGenerator by 
        opening all config files and storing their contents.
        """
        with open(defaults_path) as f:
            self.defaults = yaml.safe_load(f)
        with open(varlist_conf_path) as f:
            self.varlist_conf = yaml.safe_load(f)
        self.model_def_settings = pd.read_csv(model_def_settings_path)
        self.file_output_vars = file_output_vars
        self.context = {}
        self.build_context()

    def get_device_name(self):
        import habana_frameworks.torch.hpu as hthpu
        os.environ["LOG_LEVEL_ALL"] = "6"
        device_name = hthpu.get_device_name()
        return device_name

    def get_model_from_csv(self):
        """
        Reads the model settings CSV and 
        returns a dictionary for the selected model.
        """
        filtered = self.model_def_settings[self.model_def_settings['MODEL'] ==
                                           self.context['MODEL']]

        if filtered.empty:
            raise ValueError(
                f"No matching rows found for model '{self.context['MODEL']}'")

        return filtered.iloc[0].to_dict()

    def build_context(self):
        """
        Build context dictionary for autocalc rules and server configuration.
        Loads defaults from self.defaults (from settings/defaults.yaml).
        """
        self.context['MODEL'] = os.environ.get('MODEL')
        if not self.context['MODEL']:
            print('Error: no model. Provide model name in env var "MODEL"')
            exit(-1)
        defaults = self.defaults.get('defaults', {})
        self.context['HPU_MEM'] = defaults.get('HPU_MEM', {})
        self.context['DTYPE'] = defaults.get('DTYPE', "bfloat16")
        self.context['DEVICE_NAME'] = defaults.get(
            'DEVICE_NAME') or self.get_device_name()
        server_conf = self.get_model_from_csv()
        self.context.update(server_conf)

    def overwrite_params(self):
        """
        Overwrite default values with user provided ones before auto_calc.
        Uses the 'user_update_vars' section from self.varlist_conf.
        """
        user_update_vars = self.varlist_conf.get('user_variable', [])
        for param in user_update_vars:
            if os.environ.get(param) is not None:
                try:
                    self.context[param] = eval(os.environ[param])
                except Exception:
                    self.context[param] = os.environ[param]
                print(f"Adding or updating {param} to {self.context[param]}")
        return self.context

    def auto_calc_all(self):
        for param, func in PARAM_CALC_FUNCS.items():
            self.context[param] = func(self.context)

    def return_dict(self):
        """
        Returns a dictionary with variables listed in 'output_vars' 
        from varlist_conf.yaml.
        """
        output_vars = self.varlist_conf.get('output_vars', [])
        return {
            key: self.context[key]
            for key in output_vars if key in self.context
        }

    def calculate_variables(self):
        """
        Main execution method.
        """
        self.overwrite_params()
        try:
            self.auto_calc_all()
        except ValueError as e:
            print("Error:", e)
            exit(-1)
        return self.return_dict()
