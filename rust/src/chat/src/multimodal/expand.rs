//! Prompt placeholder expansion shared across modalities.

use std::collections::{HashMap, VecDeque};

use llm_multimodal::{Modality, PromptReplacement};
use vllm_engine_core_client::protocol::multimodal::PlaceholderRange;
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::PreparedMedia;
use crate::error::{Error, Result, bail_multimodal};

/// One modality's queue of pending placeholder replacements for prompt
/// expansion.
struct ExpansionLane<'a> {
    modality: Modality,
    marker_token_id: u32,
    embed_token_id: u32,
    placeholder_token: String,
    replacements: VecDeque<&'a PromptReplacement>,
}

impl<'a> ExpansionLane<'a> {
    fn from_prepared(media: &'a PreparedMedia) -> Option<Self> {
        if media.replacements.is_empty() {
            return None;
        }

        Some(Self {
            modality: media.modality,
            marker_token_id: media.placeholder.marker_token_id,
            embed_token_id: media.placeholder.embed_token_id,
            placeholder_token: media.placeholder.token.clone(),
            replacements: media.replacements.iter().collect(),
        })
    }
}

/// Replace rendered placeholder markers with model-specific replacement
/// tokens across all modalities in one left-to-right pass.
///
/// Each prepared modality consumes its own marker occurrences in order,
/// matching the original media-part order within that modality; markers of
/// different modalities may interleave freely.
///
/// The returned ranges point into the already-expanded prompt, grouped per
/// modality in item order.
pub(super) fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    prepared: &[PreparedMedia],
) -> Result<HashMap<Modality, Vec<PlaceholderRange>>> {
    let mut lanes = prepared.iter().filter_map(ExpansionLane::from_prepared).collect::<Vec<_>>();
    if lanes.is_empty() {
        return Ok(HashMap::new());
    }

    let replacement_growth = lanes
        .iter()
        .flat_map(|lane| lane.replacements.iter())
        .fold(0usize, |total, replacement| {
            total.saturating_add(replacement.tokens.len().saturating_sub(1))
        });
    let expanded_len = prompt_token_ids.len().saturating_add(replacement_growth);

    let mut expanded = Vec::with_capacity(expanded_len);
    let mut ranges = HashMap::<Modality, Vec<PlaceholderRange>>::new();

    for &token in prompt_token_ids.iter() {
        let lane = lanes
            .iter_mut()
            .find(|lane| lane.marker_token_id == token && !lane.replacements.is_empty());
        let Some(lane) = lane else {
            expanded.push(token);
            continue;
        };

        let replacement = lane.replacements.pop_front().expect("lane queue is non-empty");
        debug_assert_eq!(replacement.modality, lane.modality);
        if replacement.tokens.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` expanded to no tokens",
                lane.placeholder_token
            );
        }

        let replacement_len = replacement.tokens.len();
        let is_embed = {
            let mask = replacement
                .tokens
                .iter()
                .map(|&token| token as u32 == lane.embed_token_id)
                .collect::<Vec<_>>();
            WireTensor::from_bool(vec![replacement_len], mask).map_err(Error::Multimodal)?
        };

        let expanded_offset = expanded.len();
        expanded.extend(replacement.tokens.iter().map(|&token| token as u32));
        ranges.entry(lane.modality).or_default().push(PlaceholderRange {
            offset: expanded_offset,
            length: replacement_len,
            is_embed: Some(is_embed),
        });
    }

    for lane in &lanes {
        if !lane.replacements.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` was not found in tokenized prompt for {} remaining `{}` item(s)",
                lane.placeholder_token,
                lane.replacements.len(),
                lane.modality
            );
        }
    }

    *prompt_token_ids = expanded;

    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use llm_multimodal::TokenId;
    use vllm_engine_core_client::protocol::tensor::WireArrayData;

    use super::super::tests::{
        LLAMA4_IMAGE_END_ID, LLAMA4_IMAGE_ID, LLAMA4_IMAGE_START_ID, LLAMA4_PATCH_ID,
        LLAMA4_TILE_X_SEPARATOR_ID, LLAMA4_TILE_Y_SEPARATOR_ID, QWEN3_IMAGE_PAD_ID,
        QWEN3_VIDEO_PAD_ID,
    };
    use super::super::{PreparedMedia, ResolvedPlaceholder};
    use super::*;

    /// Build prepared media directly from placeholder token IDs.
    fn prepared_media(
        modality: Modality,
        placeholder_token: &str,
        marker_token_id: u32,
        embed_token_id: u32,
        replacements: Vec<PromptReplacement>,
    ) -> PreparedMedia {
        PreparedMedia {
            modality,
            placeholder: ResolvedPlaceholder {
                token: placeholder_token.to_string(),
                marker_token_id,
                embed_token_id,
            },
            replacements,
            items: Vec::new(),
        }
    }

    /// Llama4 image prepared media: the `<|image|>` marker expands to
    /// sequences whose embed positions are the `<|patch|>` tokens.
    fn llama4_prepared(replacements: Vec<PromptReplacement>) -> PreparedMedia {
        prepared_media(
            Modality::Image,
            "<|image|>",
            LLAMA4_IMAGE_ID,
            LLAMA4_PATCH_ID,
            replacements,
        )
    }

    fn qwen3_image_prepared(replacements: Vec<PromptReplacement>) -> PreparedMedia {
        prepared_media(
            Modality::Image,
            "<|image_pad|>",
            QWEN3_IMAGE_PAD_ID,
            QWEN3_IMAGE_PAD_ID,
            replacements,
        )
    }

    fn qwen3_video_prepared(replacements: Vec<PromptReplacement>) -> PreparedMedia {
        prepared_media(
            Modality::Video,
            "<|video_pad|>",
            QWEN3_VIDEO_PAD_ID,
            QWEN3_VIDEO_PAD_ID,
            replacements,
        )
    }

    fn llama4_single_tile_replacement() -> PromptReplacement {
        PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            vec![
                LLAMA4_IMAGE_START_ID as TokenId,
                LLAMA4_IMAGE_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_IMAGE_END_ID as TokenId,
            ],
        )
    }

    fn llama4_multi_tile_replacement() -> PromptReplacement {
        PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            vec![
                LLAMA4_IMAGE_START_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_TILE_X_SEPARATOR_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_TILE_Y_SEPARATOR_ID as TokenId,
                LLAMA4_IMAGE_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_IMAGE_END_ID as TokenId,
            ],
        )
    }

    fn assert_bool_mask(range: &PlaceholderRange, expected: &[bool]) {
        let tensor = range.is_embed.as_ref().expect("is_embed mask");
        assert_eq!(tensor.dtype, "bool");
        assert_eq!(tensor.shape, vec![expected.len()]);
        assert_eq!(
            tensor.data,
            WireArrayData::RawView(expected.iter().map(|value| u8::from(*value)).collect())
        );
    }

    #[test]
    fn expand_prompt_tokens_marks_only_llama4_patch_tokens_as_embed() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let prepared = vec![llama4_prepared(vec![llama4_multi_tile_replacement()])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_TILE_X_SEPARATOR_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_TILE_Y_SEPARATOR_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                2,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 8);
        assert_bool_mask(
            &ranges[0],
            &[false, true, false, true, false, false, true, false],
        );
    }

    #[test]
    fn expand_prompt_tokens_errors_when_placeholder_missing() {
        let mut prompt_token_ids = vec![1, 2, 3];
        let prepared = vec![llama4_prepared(vec![llama4_single_tile_replacement()])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_ignores_empty_replacements() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let prepared = vec![llama4_prepared(Vec::new())];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();

        assert!(ranges.is_empty());
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_leaves_prompt_unchanged_when_later_placeholder_missing() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let prepared = vec![llama4_prepared(vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let prepared = vec![llama4_prepared(vec![PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            Vec::new(),
        )])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let prepared = vec![llama4_prepared(vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                2,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                3,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 5);
        assert_bool_mask(&ranges[0], &[false, false, true, true, false]);
        assert_eq!(ranges[1].offset, 7);
        assert_eq!(ranges[1].length, 5);
        assert_bool_mask(&ranges[1], &[false, false, true, true, false]);
    }

    #[test]
    fn expand_prompt_tokens_interleaves_image_and_video_prepared_media() {
        let mut prompt_token_ids = vec![
            1,
            QWEN3_IMAGE_PAD_ID,
            2,
            QWEN3_VIDEO_PAD_ID,
            3,
            QWEN3_IMAGE_PAD_ID,
            4,
        ];
        let prepared = vec![
            qwen3_image_prepared(vec![
                PromptReplacement::repeated(
                    Modality::Image,
                    "<|image_pad|>",
                    QWEN3_IMAGE_PAD_ID as TokenId,
                    2,
                ),
                PromptReplacement::repeated(
                    Modality::Image,
                    "<|image_pad|>",
                    QWEN3_IMAGE_PAD_ID as TokenId,
                    3,
                ),
            ]),
            qwen3_video_prepared(vec![PromptReplacement::repeated(
                Modality::Video,
                "<|video_pad|>",
                QWEN3_VIDEO_PAD_ID as TokenId,
                4,
            )]),
        ];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                QWEN3_IMAGE_PAD_ID,
                QWEN3_IMAGE_PAD_ID,
                2,
                QWEN3_VIDEO_PAD_ID,
                QWEN3_VIDEO_PAD_ID,
                QWEN3_VIDEO_PAD_ID,
                QWEN3_VIDEO_PAD_ID,
                3,
                QWEN3_IMAGE_PAD_ID,
                QWEN3_IMAGE_PAD_ID,
                QWEN3_IMAGE_PAD_ID,
                4,
            ]
        );

        let image_ranges = &ranges[&Modality::Image];
        assert_eq!(image_ranges[0].offset, 1);
        assert_eq!(image_ranges[0].length, 2);
        assert_bool_mask(&image_ranges[0], &[true, true]);
        assert_eq!(image_ranges[1].offset, 9);
        assert_eq!(image_ranges[1].length, 3);
        assert_bool_mask(&image_ranges[1], &[true, true, true]);

        let video_ranges = &ranges[&Modality::Video];
        assert_eq!(video_ranges[0].offset, 4);
        assert_eq!(video_ranges[0].length, 4);
        assert_bool_mask(&video_ranges[0], &[true, true, true, true]);
    }

    #[test]
    fn expand_prompt_tokens_error_names_modality_with_leftover_replacements() {
        let mut prompt_token_ids = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let prepared = vec![
            qwen3_image_prepared(vec![PromptReplacement::repeated(
                Modality::Image,
                "<|image_pad|>",
                QWEN3_IMAGE_PAD_ID as TokenId,
                2,
            )]),
            qwen3_video_prepared(vec![PromptReplacement::repeated(
                Modality::Video,
                "<|video_pad|>",
                QWEN3_VIDEO_PAD_ID as TokenId,
                4,
            )]),
        ];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap_err();

        assert!(matches!(
            error,
            Error::Multimodal(message)
                if message.contains("<|video_pad|>") && message.contains("`video`")
        ));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }
}
