# Compile-time layer variant; mono/variants.py loads this package once per variant with its own copy of this module.
NOPE_LAYER = False  # ML4 NoPE layers (i % 4 == 3): q is 128 nope dims, zero rope key, no indexer RoPE
KV_SHARE = False  # ML4 KV-sharing layers (39, 43, ..., 71): read layer 35's latent cache, never write it
