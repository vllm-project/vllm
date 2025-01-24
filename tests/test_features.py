from vllm.features import *


def test_incompatible_features():
    usage = FeatureUsage()
    # no features
    assert usage.compatible() is True

    # one feature
    usage.add(FEATURE_STRUCTURED_OUTPUT)
    assert usage.compatible() is True

    # a second compatible feature
    usage.add(FEATURE_BEST_OF)
    assert usage.compatible() is True

    # two features are incompatible with each other
    usage.add(FEATURE_SPEC_DECODE)
    assert usage.compatible() is False


def test_incompatible_features_with_base():
    base = FeatureUsage()
    base.add(FEATURE_STRUCTURED_OUTPUT)
    base.add(FEATURE_BEST_OF)

    usage = FeatureUsage(base)
    usage.add(FEATURE_SPEC_DECODE)
    assert usage.compatible() is False
