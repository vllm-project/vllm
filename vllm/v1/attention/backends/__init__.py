# TPU v6e Adaptive Attention Backend Registration
from .tpu_v6_adaptive_pallas import (
    TPUv6AdaptiveAttentionBackend,
    TPUv6AdaptiveAttentionBackendImpl,
    create_tpu_v6_adaptive_backend,
    tpu_detector
)