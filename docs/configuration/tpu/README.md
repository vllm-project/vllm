# TPU Optimization Tips

This doc serves as a collection of handy tips for optimizing your vLLM on TPU workload.

### TPU workload sizing 
- [link to easy HBM calculator colab]

### Optimize based on your data
- max model len
- most model len
- padding

### If possible, use the precision supported by the chip
- v5e has bf16 hardware acceleration
- v6e has int8/int4 hardware acceleration

### Don't set TP to be less than the number of chips on the host
- If you need 1 or 4 chips, just create an instance with 1 or 4 chips, don't try to fragment 2 different workloads across 8 chips.

### Tune your workloads!
- Although we try to have great default configs, we strongly recommend you checkout our auto-tuner and optimize for your workload[LINK]. 
