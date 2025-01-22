from dataclasses import dataclass
from typing import List


@dataclass
class ComputedTokenRange:
    """
    (start, end]
    """
    start: int
    end: int


ComputedTokens = List[ComputedTokenRange]


def intersect_two_ranges(
        a: List[ComputedTokenRange],
        b: List[ComputedTokenRange]) -> List[ComputedTokenRange]:
    """
    Intersect two sorted lists of ComputedTokenRange intervals.
    
    Args:
        a: List of intervals
        b: List of intervals
    Returns:
        List of intervals that are intersections of a and b
    """
    i, j = 0, 0
    result = []

    while i < len(a) and j < len(b):
        overlap_start = max(a[i].start, b[j].start)
        overlap_end = min(a[i].end, b[j].end)

        if overlap_start <= overlap_end:
            result.append(ComputedTokenRange(overlap_start, overlap_end))

        if a[i].end < b[j].end:
            i += 1
        else:
            j += 1

    return result


def intersect_ranges(
        ranges: List[List[ComputedTokenRange]]) -> List[ComputedTokenRange]:
    """
    Intersect multiple lists of ComputedTokenRange intervals, each is sorted.
    
    Args:
        ranges: A list of lists of intervals 
    Returns:
        A list of intervals representing the intersection of all ranges
    """
    if not ranges:
        return []

    current_intersection = ranges[0]
    for i in range(1, len(ranges)):
        current_intersection = intersect_two_ranges(current_intersection,
                                                    ranges[i])
        if not current_intersection:
            break

    return current_intersection
