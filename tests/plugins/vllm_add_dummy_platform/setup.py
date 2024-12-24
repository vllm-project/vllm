from setuptools import setup

setup(name='vllm_add_dummy_platform',
      version='0.1',
      packages=['vllm_add_dummy_platform'],
      entry_points={
          'vllm.general_plugins':
          ["register_dummy_model = vllm_add_dummy_platform:register"]
      })
