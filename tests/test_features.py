from vllm.features import FEATURES, INCOMPATIBILITY_MATRIX, FeatureUsage


def test_incompatible_features():
    usage = FeatureUsage()
    # no features
    assert usage.compatible() is True

    # one feature
    usage.add(FEATURES.STRUCTURED_OUTPUT)
    assert usage.compatible() is True

    # a second compatible feature
    usage.add(FEATURES.BEST_OF)
    assert usage.compatible() is True

    # two features are incompatible with each other
    usage.add(FEATURES.SPEC_DECODE)
    assert usage.compatible() is False


def test_incompatible_features_with_base():
    base = FeatureUsage()
    base.add(FEATURES.STRUCTURED_OUTPUT)
    base.add(FEATURES.BEST_OF)

    usage = FeatureUsage(base)
    usage.add(FEATURES.SPEC_DECODE)
    assert usage.compatible() is False


def test_incompat_matrix_consistency():
    # Features can not be incompatible with themselves
    for feature, incompatible in INCOMPATIBILITY_MATRIX.items():
        assert (feature & incompatible) == 0

    # Features listed as incompatible should also be listed as incompatible
    # the other way around
    for feature, incompatible in INCOMPATIBILITY_MATRIX.items():
        for other_feature in INCOMPATIBILITY_MATRIX:
            if feature == other_feature:
                continue
            if incompatible & other_feature:
                assert INCOMPATIBILITY_MATRIX[other_feature] & feature
