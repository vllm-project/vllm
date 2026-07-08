//! Prompt placeholder expansion shared across modalities.

use std::collections::{HashMap, VecDeque};

use llm_multimodal::{Modality, PromptReplacement};
use vllm_engine_core_client::protocol::multimodal::PlaceholderRange;
use vllm_engine_core_client::protocol::tensor::WireTensor;

use crate::error::{Error, Result, bail_multimodal};

/// One modality's queue of pending placeholder replacements for prompt
/// expansion.
pub(super) struct ExpansionLane {
    pub(super) modality: Modality,
    pub(super) marker_token_id: u32,
    pub(super) embed_token_id: u32,
    pub(super) placeholder_token: String,
    pub(super) replacements: VecDeque<PromptReplacement>,
}

/// Replace rendered placeholder markers with model-specific replacement
/// tokens across all modalities in one left-to-right pass.
///
/// Each lane consumes its own marker occurrences in order, matching the
/// original media-part order within that modality; markers of different
/// modalities may interleave freely. Only the original prompt is scanned, so
/// marker tokens inside replacement sequences are never re-matched. The
/// returned ranges point into the already-expanded prompt, grouped per
/// modality in item order.
///
/// On error the prompt is left unchanged.
pub(super) fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    mut lanes: Vec<ExpansionLane>,
) -> Result<HashMap<Modality, Vec<PlaceholderRange>>> {
    lanes.retain(|lane| !lane.replacements.is_empty());
    if lanes.is_empty() {
        return Ok(HashMap::new());
    }

    let replacement_growth = lanes
        .iter()
        .flat_map(|lane| lane.replacements.iter())
        .fold(0usize, |total, replacement| {
            total.saturating_add(replacement.tokens.len().saturating_sub(1))
        });
    let mut expanded =
        Vec::with_capacity(prompt_token_ids.len().saturating_add(replacement_growth));
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
        expanded.extend(replacement.tokens.into_iter().map(|token| token as u32));
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
    use super::*;

    /// Build an expansion lane directly from placeholder token IDs.
    fn lane(
        modality: Modality,
        placeholder_token: &str,
        marker_token_id: u32,
        embed_token_id: u32,
        replacements: Vec<PromptReplacement>,
    ) -> ExpansionLane {
        ExpansionLane {
            modality,
            marker_token_id,
            embed_token_id,
            placeholder_token: placeholder_token.to_string(),
            replacements: replacements.into(),
        }
    }

    /// The llama4 image lane: the `<|image|>` marker expands to sequences
    /// whose embed positions are the `<|patch|>` tokens.
    fn llama4_lane(replacements: Vec<PromptReplacement>) -> ExpansionLane {
        lane(
            Modality::Image,
            "<|image|>",
            LLAMA4_IMAGE_ID,
            LLAMA4_PATCH_ID,
            replacements,
        )
    }

    fn qwen3_image_lane(replacements: Vec<PromptReplacement>) -> ExpansionLane {
        lane(
            Modality::Image,
            "<|image_pad|>",
            QWEN3_IMAGE_PAD_ID,
            QWEN3_IMAGE_PAD_ID,
            replacements,
        )
    }

    fn qwen3_video_lane(replacements: Vec<PromptReplacement>) -> ExpansionLane {
        lane(
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
        let lanes = vec![llama4_lane(vec![llama4_multi_tile_replacement()])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap();
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
        let lanes = vec![llama4_lane(vec![llama4_single_tile_replacement()])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_ignores_empty_replacements() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![llama4_lane(Vec::new())];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap();

        assert!(ranges.is_empty());
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_leaves_prompt_unchanged_when_later_placeholder_missing() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![llama4_lane(vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![llama4_lane(vec![PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            Vec::new(),
        )])];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let lanes = vec![llama4_lane(vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap();
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
    fn expand_prompt_tokens_interleaves_image_and_video_lanes() {
        let mut prompt_token_ids = vec![
            1,
            QWEN3_IMAGE_PAD_ID,
            2,
            QWEN3_VIDEO_PAD_ID,
            3,
            QWEN3_IMAGE_PAD_ID,
            4,
        ];
        let lanes = vec![
            qwen3_image_lane(vec![
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
            qwen3_video_lane(vec![PromptReplacement::repeated(
                Modality::Video,
                "<|video_pad|>",
                QWEN3_VIDEO_PAD_ID as TokenId,
                4,
            )]),
        ];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap();

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
        let lanes = vec![
            qwen3_image_lane(vec![PromptReplacement::repeated(
                Modality::Image,
                "<|image_pad|>",
                QWEN3_IMAGE_PAD_ID as TokenId,
                2,
            )]),
            qwen3_video_lane(vec![PromptReplacement::repeated(
                Modality::Video,
                "<|video_pad|>",
                QWEN3_VIDEO_PAD_ID as TokenId,
                4,
            )]),
        ];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(
            error,
            Error::Multimodal(message)
                if message.contains("<|video_pad|>") && message.contains("`video`")
        ));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }
}
