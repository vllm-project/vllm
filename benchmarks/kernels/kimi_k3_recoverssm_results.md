# Kimi K3 RecoverSSM Commit Results

Configuration: B300, TP8 (12 local heads), 69 KDA layers, head dimension 128,
T=8, and five accepted tokens. Values are mean CUPTI spans in microseconds
from the commit plan through both commit kernels; reset and L2 flush are
captured but excluded.

## Tuned implementation

The optimized path uses the forward recurrence, immediate boundary-state
write, and KDA-first PDL overlap. The compact kernel is the PDL-dependent
consumer.

| Mode | B1 | B32 | B256 |
| --- | ---: | ---: | ---: |
| Unaligned | 31.72 | 755.81 | 6128.89 |
| Aligned, no boundary | 40.01 | 971.00 | 7921.03 |
| Aligned, boundary at accepted token 2 | 44.79 | 1157.97 | 9392.93 |

| Mode | KDA commit | Conv compact |
| --- | --- | --- |
| Unaligned | value slice 16, 1 warp | channel tile 128, 1 warp |
| Aligned | value slice 32, 1 warp | channel tile 64, 1 warp |

The aligned configuration is used for both aligned cases; it does not need a
separate compile-time choice for whether this iteration crosses a boundary.

## Original kernels

These were rerun on the same B300 with the preserved original benchmark
revision (`314990ee`): backward recurrence, retained boundary state, original
256-channel/4-warp compact launch, and no PDL.

| Mode | B1 | B32 | B256 |
| --- | ---: | ---: | ---: |
| Unaligned | 64.67 | 1615.72 | 13092.84 |
| Aligned, no boundary | 84.67 | 2281.48 | 18976.55 |
| Aligned, boundary at accepted token 2 | 88.53 | 2393.79 | 20051.55 |
