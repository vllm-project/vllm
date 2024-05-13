.. Logits Processor Plugins:

Logits Processor Plugins
========================

vLLM supports using custom logits processors through plugins.
This means you can use custom logits processors, and even create your own without having to change vLLM.

Installing a logits processor plugin
------------------------------------

To install a logits processor plugin,
all you have to do it install the Python package containing the plugin in the same environment as vLLM.


Using the installed plugin
--------------------------------

To use the logits processor plugins you installed,
you can use the :code:`logits_processors` field in the generation request body as such:

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "...",
    $         "prompt": "Hello!",
    $         "max_tokens": 32,
    $         "temperature": 0,
    $         "logits_processors": {
    $           "my_logits_processor": {
    $               "my_param": 100
    $           }
    $         }
    $     }'

.. note::
    This is only an example, in reality each logits processor plugin has a name which is the key in the :code:`logits_processors` dictionary.
    The value is a dictionary of parameters passed to the plugin's implementation.


Creating your own logits processor plugin
-----------------------------------------

Advanced users might want to build their custom logits processors and publish them as plugins.
This can be done by simply creating a Python package with your implementation.

Here is an example :code:`main.py` for a logits processor plugin implementation,
that takes a token ID and multiplies it's logit by 100:

.. code-block:: python

    from pydantic import BaseModel


    class MyParameters(BaseModel):
        token_id: int


    class MyLogitsProcessor:
        def __init__(self, tokenizer, parameters: MyParameters):
            self.tokenizer = tokenizer
            self.parameters = parameters

        def __call__(self, token_ids, logits):
            new_logits = logits.clone()
            new_logits[self.parameters.token_id] *= 100
            return new_logits


    LOGITS_PROCESSOR_PLUGIN = {
        'logits_processor_class': MyLogitsProcessor,
        'parameters_model': MyParameters
    }


The :code:`setup.py` file for the plugin package should look something like this:

.. code-block:: python

    from setuptools import setup

    setup(name='example_logits_processor',
          version='0.1',
          install_requires=[
                "pydantic>=1.8.2"
          ],
          entry_points={
                'vllm.logits_processors': ['example_plugin=example_plugin.main:LOGITS_PROCESSOR_PLUGIN']
          }
   )

After installing the plugin package in the same environment as vLLM,
you can run vLLM and use you custom logits processor as such:

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "...",
    $         "prompt": "Hello!",
    $         "max_tokens": 32,
    $         "temperature": 0,
    $         "logits_processors": {
    $           "example_plugin": {
    $               "token_id": 10
    $           }
    $         }
    $     }'
