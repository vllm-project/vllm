# LiteTopK JIT sources

The SM100 scoring pipeline in `dsa_litetopk.cu` and
`sm100_dsa_litetopk.cuh` is derived from DeepSeek DeepGEMM, commit
`891d57b4db1071624b5c8fa0d1e51cb317fa709f`, and remains available under
the MIT license in `LICENSE.deepseek-deepgemm`.

The sources live inside the `vllm` Python package because
`litetopk_indexer.py` hashes and JIT-compiles them at runtime. Keep the
package-data entries in `setup.py` synchronized with this directory.
