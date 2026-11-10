// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! CPython's `heapq` algorithms (`Modules/_heapqmodule.c`), reproduced
//! step by step so the heap list order matches Python element by element
//! for any `<` relation, including ones whose keys changed while queued.
//!
//! `lt(a, b)` is `a < b`, the only comparison `heapq` makes.

/// Follow the path to the root, moving parents down until finding a place
/// `heap[pos]` fits.
fn siftdown<T>(
    heap: &mut [T],
    startpos: usize,
    mut pos: usize,
    lt: &mut impl FnMut(&T, &T) -> bool,
) {
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        if !lt(&heap[pos], &heap[parentpos]) {
            break;
        }
        heap.swap(pos, parentpos);
        pos = parentpos;
    }
}

/// Bubble the smaller child up until hitting a leaf, then sift the item
/// that was at `pos` down from there.
fn siftup<T>(heap: &mut [T], mut pos: usize, lt: &mut impl FnMut(&T, &T) -> bool) {
    let endpos = heap.len();
    let startpos = pos;
    let limit = endpos >> 1;
    while pos < limit {
        // Set childpos to index of smaller child.
        let mut childpos = 2 * pos + 1;
        if childpos + 1 < endpos && !lt(&heap[childpos], &heap[childpos + 1]) {
            childpos += 1;
        }
        // Move the smaller child up.
        heap.swap(pos, childpos);
        pos = childpos;
    }
    // Bubble it up to its final resting place (by sifting its parents down).
    siftdown(heap, startpos, pos, lt);
}

/// Push item onto heap, maintaining the heap invariant.
pub(crate) fn heappush<T>(heap: &mut Vec<T>, item: T, mut lt: impl FnMut(&T, &T) -> bool) {
    heap.push(item);
    let pos = heap.len() - 1;
    siftdown(heap, 0, pos, &mut lt);
}

/// Pop the smallest item off the heap, maintaining the heap invariant.
pub(crate) fn heappop<T>(heap: &mut Vec<T>, mut lt: impl FnMut(&T, &T) -> bool) -> Option<T> {
    let lastelt = heap.pop()?;
    if heap.is_empty() {
        return Some(lastelt);
    }
    let returnitem = std::mem::replace(&mut heap[0], lastelt);
    siftup(heap, 0, &mut lt);
    Some(returnitem)
}

/// Transform list into a heap, in-place, in O(len(heap)) time.
///
/// CPython uses a cache-friendly traversal for more than 2500 items; it
/// sifts the same subtrees in a different order and produces exactly the
/// same heap.
pub(crate) fn heapify<T>(heap: &mut [T], mut lt: impl FnMut(&T, &T) -> bool) {
    for i in (0..heap.len() / 2).rev() {
        siftup(heap, i, &mut lt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_heapq() {
        // heap = []
        // for x in [5, 3, 8, 1, 9, 2, 7]: heapq.heappush(heap, x)
        let mut heap = Vec::new();
        for x in [5, 3, 8, 1, 9, 2, 7] {
            heappush(&mut heap, x, |a, b| a < b);
        }
        assert_eq!(heap, [1, 3, 2, 5, 9, 8, 7]);
        // heapq.heappop(heap)
        assert_eq!(heappop(&mut heap, |a, b| a < b), Some(1));
        assert_eq!(heap, [2, 3, 7, 5, 9, 8]);
        // h = [9, 8, 7, 6, 5, 4, 3, 2, 1]; heapq.heapify(h)
        let mut h = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
        heapify(&mut h, |a, b| a < b);
        assert_eq!(h, [1, 2, 3, 6, 5, 4, 7, 8, 9]);
        let mut empty: Vec<i32> = Vec::new();
        assert_eq!(heappop(&mut empty, |a, b| a < b), None);
    }
}
