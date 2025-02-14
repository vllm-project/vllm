# SPDX-License-Identifier: Apache-2.0

from setuptools import setup

setup(
    name='vllm_add_dummy_scheduler',
    version='0.1',
    packages=['vllm_add_dummy_scheduler'],
    entry_points={
        'vllm.platform_plugins': [
            "dummy_scheduler_plugin = vllm_add_dummy_scheduler:dummy_scheduler_plugin"  # noqa
        ]
    })
