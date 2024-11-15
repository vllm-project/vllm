vLLM's Plugin System
=======================

It is a common ask from the community to extend vLLM with custom features. To make this process easier, vLLM has a plugin system that allows users to extend vLLM with custom features without modifying the vLLM codebase. This document explains how plugins work in vLLM, and to create a plugin for vLLM.

How Plugins Work in vLLM
-------------------------

Plugins, in essence, are user-registered code that wants to be executed by vLLM. Given the architecture of vLLM, there might be several processes involved, epsecially when we use distributed inference with various parallelism techniques. To successfully enable plugins, every process created by vLLM needs to load the plugin. This is done by the `load_general_plugins <https://github.com/vllm-project/vllm/blob/c76ac49d266e27aa3fea84ef2df1f813d24c91c7/vllm/plugins/__init__.py#L16>`__ function in the ``vllm.plugins`` module. The function will be called for every process created by vLLM before it starts to do any work.

How vLLM discovers plugins
--------------------------

vLLM's plugin system uses the standard Python ``entry_points`` mechanism. This mechanism allows developers to register functions in their Python packages to be used for other packages. An example of plugin:

.. code-block:: python

    # inside `setup.py` file
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })
    
    # inside `vllm_add_dummy_model.py` file
    def register():
        from vllm import ModelRegistry

        def register():
            if "MyLlava" not in ModelRegistry.get_supported_archs():
                ModelRegistry.register_model("MyLlava",
                                            "vllm_add_dummy_model.my_llava:MyLlava")


Please check the `official documentation <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`__ for more information about how to add entry points to your package.

Every plugin has three parts:

1. **Plugin group**: The name of the entry point group. vLLM uses the entry point group ``vllm.general_plugins`` to register general plugins. This is the key of ``entry_points`` in the ``setup.py`` file. Please always use ``vllm.general_plugins`` for vLLM's general plugins.

2. **Plugin name**: The name of the plugin. This is in the value of the dictionary in the ``entry_points`` dictionary. In the example above, the plugin name is ``register_dummy_model``. We filter plugins by their names, using ``VLLM_PLUGINS`` environment variables. If you want to load only a specific plugin, you can set ``VLLM_PLUGINS`` to the plugin name.

3. **Plugin value**: The qualified full name of the function that you want to register into the plugin system. In the example above, the plugin value is ``vllm_add_dummy_model:register``. It is a function named ``register`` in the ``vllm_add_dummy_model`` module in this case.

What can plugins do?
---------------------

Currently, the main use case for plugins is to register custom, out-of-the-tree models into vLLM. This is done by calling ``ModelRegistry.register_model`` to register the model. In the future, we can extend the plugin system to support more features, such as swapping in some classes into vLLM with custom implementations.

Guidelines for writing plugins
-------------------------------

- **Being re-entrant**: the function specified in the entry point should be re-entrant. This means that the function should be able to be called multiple times without causing any issues. This is because the function might be called multiple times in some processes.

Compatability Guarantee
------------------------

vLLM will always guarantee the interface of documented plugins, for example, ``ModelRegistry.register_model`` will always be available for plugins to register models. However, it is the responsibilities of the plugin developers to ensure that their plugins are compatible with the version of vLLM they are targeting. For example, ``"vllm_add_dummy_model.my_llava:MyLlava"`` should be compatible with the version of vLLM that the plugin is targeting. The interface for the model can be subject to change during the development of vLLM.
