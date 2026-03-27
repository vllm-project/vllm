from vllm.config.utils import config

@config
class ScoreEncoderCacheConfig:
    """
    Configuration class for controlling the behavior of the CHIME system.

    This configuration defines a score-based encoder cache management policy,
    mainly used to control cache entry clock decay, promotion strategy,
    and cache management thresholds.
    """
    def __init__(self,
                score_encoder_cache_config: dict):
        """
        Initialize ScoreEncoderCacheConfig.

        Args:
            score_encoder_cache_config (dict):
                A dictionary containing configuration parameters
                for the encoder cache scoring policy.
        """

        # Whether to enable the score-based encoder cache management policy
        self.enabled = score_encoder_cache_config.get("enabled", False)

        # Maximum number of encoder cache slots available on the CPU side
        self.cpu_cache_slots = score_encoder_cache_config.get("cpu_cache_slots", 100000)

        # Maximum clock value used by the clock mechanism,
        # representing the highest activity or freshness level of a cache entry
        self.max_clock = score_encoder_cache_config.get("max_clock", 15)

        # Number of operations between clock decay steps.
        # Clock decay gradually decreases the score of cache entries
        # that have not been accessed for a long time.
        self.clock_decay_every = score_encoder_cache_config.get("clock_decay_every", 64)

        # Cache watermark threshold. When eviction is triggered,
        # cache entries will be continuously removed until the cache
        # usage ratio drops below this threshold.
        self.watermark = score_encoder_cache_config.get("watermark", 0.2)

        # Promotion percentile threshold.
        # If the score of a cache entry exceeds this percentile
        # in the overall score distribution, the entry can be promoted.
        self.promote_percentile = score_encoder_cache_config.get("promote_percentile", 0.2)