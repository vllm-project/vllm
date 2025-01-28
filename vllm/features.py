from typing import Dict, Optional

#
# Instructions:
#
# To add a new feature:
#
# 1) Reserve a bit by defining a new constant, FEATURE_<NAME>, with a value of
#    1 << n, where n is the next available bit in the range 0-63.
#
# 2) Add a string representation of the feature to FEATURE_STR.
#
# 3) Add the feature to INCOMPATIBILITY_MATRIX. The key should be the feature
#    and the value should be a bit mask of all the features that are
#    incompatible with it. The feature should be added to the value of any other
#    feature that it is incompatible with, as well.
#
# To change the compatibility matrix for existing features,
# update INCOMPATIBILITY_MATRIX.
#
# This should match what in the feature compatibility matrix in the docs:
# https://docs.vllm.ai/en/latest/features/compatibility_matrix.html
#

# This range must not go above 63 as they represent the bits in the feature
# usage bit field. The implementation will have to change if we need to
# calculate compatibility between more than 64 features.
FEATURE_SPEC_DECODE: int = 1 << 0
FEATURE_STRUCTURED_OUTPUT: int = 1 << 1
FEATURE_BEST_OF: int = 1 << 2

FEATURE_STR: Dict[int, str] = {
    FEATURE_SPEC_DECODE: 'SPEC_DECODE',
    FEATURE_STRUCTURED_OUTPUT: 'STRUCTURED_OUTPUT',
    FEATURE_BEST_OF: 'BEST_OF',
}

INCOMPATIBILITY_MATRIX: Dict[int, int] = {
    FEATURE_SPEC_DECODE: FEATURE_STRUCTURED_OUTPUT | FEATURE_BEST_OF,
    FEATURE_STRUCTURED_OUTPUT: FEATURE_SPEC_DECODE,
    FEATURE_BEST_OF: FEATURE_SPEC_DECODE,
}


class FeatureUsage:
    '''A class to manage the usage of features and check for compatibility.

    This class is used to manage the usage of features and check for
    compatibility between them. It uses a bit field to store the features in use
    and checks for compatibility in O(n) time.
    '''

    _features: int

    def __init__(self, base: Optional['FeatureUsage'] = None):
        self._features = base._features if base else 0

    def add(self, feature: int):
        '''Add a feature to the current usage.

        Args:
            feature (int): The feature to add to the current usage.
        '''
        self._features |= feature

    def compatible(self) -> bool:
        '''Check if the current feature usage is compatible.

        Given a set of features in use (stored as bits in self._features),
        do a check in O(n) time to see if any of the features are incompatible
        with each other.

        Returns:
            bool: True if the features are compatible, False otherwise.
        '''
        for feature, incompatible in INCOMPATIBILITY_MATRIX.items():
            # If `feature` is in use and something in its incompatibility mask
            # is also in use
            if self._features & feature and self._features & incompatible:
                return False
        return True

    def conflicts(self) -> str:
        '''Return a string representation of the incompatible features.'''
        conflicts = []
        for feature, incompatible in INCOMPATIBILITY_MATRIX.items():
            if self._features & feature and self._features & incompatible:
                conflicts.append(FEATURE_STR[feature])
        return 'Conflicting features: ' + ', '.join(conflicts)


class FeaturesIncompatible(ValueError):
    '''An exception to raise when incompatible features are used.

    Inherit from ValueError because ValueError is used throughout vllm
    as a more generic exception that will get converted to a 400 HTTP
    response in the API. This way, if a code path doesn't explicitly
    handle FeaturesIncompatible, it can still handle it as a ValueError.
    '''
