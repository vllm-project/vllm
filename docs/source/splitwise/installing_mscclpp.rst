.. _installing_mscclpp:

Installing MSCCL++
============================

`MSCCL++ <https://github.com/microsoft/mscclpp>`_ is a GPU-driven communication stack for scalable AI applications.
It is used to implement KV cache communication in Splitwise.

To install MSCCL++, please follow the instructions at  `MSCCL++ Quickstart <https://github.com/microsoft/mscclpp/blob/main/docs/quickstart.md>`_ or follow the steps below to install it from source:

.. code-block:: console

    $ git clone https://github.com/microsoft/mscclpp;
    $ mkdir mscclpp/build; cd mscclpp/build; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j;
    $ conda install -c conda-forge mpi4py
    $ cd ../python; pip install -r requirements_c12.txt;
    $ cd ..; pip install -e .
