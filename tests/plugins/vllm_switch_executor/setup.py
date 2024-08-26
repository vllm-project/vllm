from setuptools import setup

setup(name='vllm_switch_executor',
      version='0.1',
      packages=['vllm_switch_executor'],
      entry_points={
          'vllm.general_plugins':
          ["switch_executor = vllm_switch_executor:switch_executor"]
      })
