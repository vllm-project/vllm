// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler-visible parts of `vllm.multimodal.inputs.PlaceholderRange`
//! and `MultiModalFeatureSpec`, and `vllm.multimodal.utils`'s
//! `get_mm_features_in_window`.

/// Placeholder location information for multi-modal data.
///
/// Example:
/// Prompt: `AAAA BBBB What is in these images?`
///
/// Images A and B will have:
///
/// ```text
/// A: PlaceholderRange(offset=0, length=4)
/// B: PlaceholderRange(offset=5, length=4)
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaceholderRange {
    /// The start index of the placeholder in the prompt.
    pub offset: usize,
    /// The length of the placeholder.
    pub length: usize,
    /// Inclusive prefix sums of the `is_embed` mask: a boolean mask of shape
    /// `(length,)` indicating which positions between `offset` and
    /// `offset + length` to assign embeddings to. `None` when every position
    /// is an embedding.
    ///
    /// Python caches this lazily as `embeds_cumsum`; it is computed eagerly
    /// here since the range is immutable.
    embeds_cumsum: Option<Vec<usize>>,
}

impl PlaceholderRange {
    pub fn new(offset: usize, length: usize, is_embed: Option<&[bool]>) -> Self {
        let embeds_cumsum = is_embed.map(|mask| {
            mask.iter()
                .scan(0, |sum, &is_embed| {
                    *sum += usize::from(is_embed);
                    Some(*sum)
                })
                .collect()
        });
        Self {
            offset,
            length,
            embeds_cumsum,
        }
    }

    /// This range moved to `offset`, as `dataclasses.replace(r, offset=...)`.
    pub fn with_offset(&self, offset: usize) -> Self {
        Self {
            offset,
            ..self.clone()
        }
    }

    /// Whether an `is_embed` mask is set.
    pub fn has_is_embed(&self) -> bool {
        self.embeds_cumsum.is_some()
    }

    pub fn get_num_embeds(&self) -> usize {
        match &self.embeds_cumsum {
            None => self.length,
            Some(cumsum) => cumsum.last().copied().unwrap_or(0),
        }
    }

    /// Returns the starting and ending indices of the embeddings of encoder outputs
    /// in the range of `[start_idx, end_idx)` in the placeholders.
    ///
    /// For example, given:
    /// `PlaceholderRange(offset=2, length=5, is_embed=[False, True, False, True, True])`
    ///
    /// If `start_idx=3` and `end_idx=5`, the output is `(1, 3)` because we want to get
    /// the second and the third embeddings from the encoder output.
    pub fn get_embeds_indices_in_range(&self, start_idx: usize, end_idx: usize) -> (usize, usize) {
        let Some(cumsum) = &self.embeds_cumsum else {
            return (start_idx, end_idx);
        };
        let at = |idx: usize| if idx > 0 { cumsum[idx - 1] } else { 0 };
        (at(start_idx), at(end_idx))
    }

    /// Extract the start and end indices of the embedded region in prompt.
    ///
    /// For example, given `PlaceholderRange(offset=2, length=5)` and
    /// `is_embed = [False, True, False, True, True]`, the output is
    /// `[(1 + offset, 1 + offset), (3 + offset, 4 + offset)]`.
    ///
    /// Returns:
    ///     A tuple `(start, end)` representing the start and end
    ///     indices (inclusive) of the embedded region.
    ///     Returns full placeholder range if `is_embed` is `None`.
    pub fn extract_embeds_range(&self) -> Vec<(usize, usize)> {
        let Some(cumsum) = &self.embeds_cumsum else {
            // Python computes `offset + length - 1`, i.e. `offset - 1` for an
            // empty range; empty placeholders do not reach this.
            return vec![(self.offset, self.offset + self.length - 1)];
        };
        let mut ranges = Vec::new();
        let mut start = None;
        let mut prev = 0;
        for (i, &sum) in cumsum.iter().enumerate() {
            let is_embed = sum > prev;
            prev = sum;
            match (is_embed, start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    ranges.push((s + self.offset, i - 1 + self.offset));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            ranges.push((s + self.offset, cumsum.len() - 1 + self.offset));
        }
        ranges
    }
}

/// Represents a single multimodal input with its processed data and metadata.
///
/// Used to track multimodal data through processing and caching.
/// A request containing multiple multimodal items will have one
/// `MultiModalFeatureSpec` per item.
///
/// The processed data (`data`, a `MultiModalKwargsItem`) is an opaque Python
/// object; the binding layer keeps it keyed by request id and item index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiModalFeatureSpec {
    /// The input modality, e.g., `"image"`, `"audio"`, `"video"`.
    pub modality: String,
    /// The hash for caching encoder outputs (with LoRA prefix if applicable).
    pub identifier: String,
    /// The location of the `modality` tokens corresponding to this item
    /// in the prompt, e.g., `PlaceholderRange(offset=2, length=336)`.
    pub mm_position: PlaceholderRange,
    /// The hash for caching processor outputs (without LoRA prefix).
    pub mm_hash: Option<String>,
    /// What `data` holds; the payload itself stays with the binding layer.
    pub data: MultiModalData,
}

/// The kind of a feature's `data` (`MultiModalKwargsItem | None`), which
/// decides whether the scheduler can release and restore its CPU payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MultiModalData {
    /// `data is None`.
    #[default]
    Missing,
    /// A shared-memory descriptor (`"address" in data`), which follows the
    /// processor cache's own ACK protocol.
    ShmAddress,
    /// A tensor payload.
    Payload,
}

/// Return `(lo, hi)` indices for features overlapping `[start, end)`.
///
/// Assumes `mm_features` are sorted by offset and non-overlapping, so
/// `offset + length` is also sorted.
pub fn get_mm_features_in_window(
    mm_features: &[MultiModalFeatureSpec],
    start: usize,
    end: usize,
) -> (usize, usize) {
    // `bisect.bisect_left(a, x, key=k)` is the first index with `k(a[i]) >= x`.
    let lo =
        mm_features.partition_point(|f| f.mm_position.offset + f.mm_position.length < start + 1);
    let hi = mm_features.partition_point(|f| f.mm_position.offset < end);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_with_mask() {
        let r = PlaceholderRange::new(2, 5, Some(&[false, true, false, true, true]));
        assert_eq!(r.get_num_embeds(), 3);
        assert_eq!(r.get_embeds_indices_in_range(3, 5), (1, 3));
        assert_eq!(r.get_embeds_indices_in_range(0, 0), (0, 0));
        assert_eq!(r.extract_embeds_range(), vec![(3, 3), (5, 6)]);

        let empty = PlaceholderRange::new(0, 0, Some(&[]));
        assert_eq!(empty.get_num_embeds(), 0);
        assert_eq!(empty.extract_embeds_range(), vec![]);
    }

    #[test]
    fn embeds_without_mask() {
        let r = PlaceholderRange::new(2, 5, None);
        assert_eq!(r.get_num_embeds(), 5);
        assert_eq!(r.get_embeds_indices_in_range(1, 4), (1, 4));
        assert_eq!(r.extract_embeds_range(), vec![(2, 6)]);
    }

    #[test]
    fn features_in_window() {
        let feature = |offset, length| MultiModalFeatureSpec {
            modality: "image".into(),
            identifier: String::new(),
            mm_position: PlaceholderRange::new(offset, length, None),
            mm_hash: None,
            data: MultiModalData::Payload,
        };
        let features = [feature(0, 4), feature(5, 4), feature(10, 2)];
        assert_eq!(get_mm_features_in_window(&features, 0, 1), (0, 1));
        assert_eq!(get_mm_features_in_window(&features, 4, 5), (1, 1));
        assert_eq!(get_mm_features_in_window(&features, 3, 6), (0, 2));
        assert_eq!(get_mm_features_in_window(&features, 9, 100), (2, 3));
        assert_eq!(get_mm_features_in_window(&features, 12, 100), (3, 3));
    }
}
