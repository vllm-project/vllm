This directory contains tuned configurations for different settings of the fused_moe kernel.
For different settings of
- E (number of experts)
- N (intermediate size)
- device_name (torch.cuda.get_device_name())
the JSON file contains a mapping from M (batch size) to the chosen configuration.

The example configurations provided are for the Mixtral model for TP2 on H100
and TP4 on A100. Mixtral has intermediate size N = 14336, i.e. for TP2 we have
N = 7168 and for TP4 we have N = 3584.

See `benchmark/kernels/benchmark_moe.py` on how to generate these config files.
