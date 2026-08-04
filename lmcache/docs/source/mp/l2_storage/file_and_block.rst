File & Block
============

File-system and block-device backends that persist KV cache to local or
networked storage. These are conceptually similar -- they write KV objects to
files or fixed-size slots on a disk or block device.

.. toctree::
   :maxdepth: 1

   fs
   fs_native
   raw_block
