// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prompt placeholder expansion shared across modalities.

use std::collections::{HashMap, VecDeque};

use llm_multimodal::{Modality, PromptReplacement};
use vllm_engine_core_client::protocol::multimodal::PlaceholderRange;
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::PreparedMedia;
use super::timing::MM_STAGE_TARGET;
use crate::error::{Error, Result, bail_multimodal};

/// One modality's queue of pending placeholder replacements for prompt
/// expansion.
struct ExpansionLane<'a> {
    modality: Modality,
    marker_token_ids: &'a [u32],
    embed_token_id: u32,
    placeholder_token: &'a str,
    replacements: VecDeque<&'a PromptReplacement>,
}

impl<'a> ExpansionLane<'a> {
    fn from_prepared(media: &'a PreparedMedia) -> Option<Self> {
        if media.replacements.is_empty() {
            return None;
        }

        Some(Self {
            modality: media.modality,
            marker_token_ids: &media.placeholder.marker_token_ids,
            embed_token_id: media.placeholder.embed_token_id,
            placeholder_token: &media.placeholder.token,
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
/// Replacements may carry an [`AlignmentPad`](llm_multimodal::AlignmentPad)
/// (e.g. DeepSeek-V4.1's compressor-alignment pad): `period - 1 - offset %
/// period` pad tokens are
/// prepended at splice time, where `offset` is the position the replacement
/// lands at in the expanded prompt. The pads are part of the reported range
/// and count as embed positions only if the pad token id equals the
/// modality's embed token id.
///
/// The returned ranges point into the already-expanded prompt, grouped per
/// modality in item order.
#[tracing::instrument(
    name = "mm_stage",
    target = MM_STAGE_TARGET,
    skip_all,
    fields(stage = "prompt_expansion")
)]
pub(super) fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    prepared: &[PreparedMedia],
) -> Result<HashMap<Modality, Vec<PlaceholderRange>>> {
    let mut lanes = prepared.iter().filter_map(ExpansionLane::from_prepared).collect::<Vec<_>>();
    if lanes.is_empty() {
        return Ok(HashMap::new());
    }

    let replacement_growth = lanes.iter().fold(0usize, |total, lane| {
        lane.replacements.iter().fold(total, |total, replacement| {
            // Alignment pads are position-dependent; reserve the worst case.
            let max_pads = replacement.alignment_pad.map_or(0, |pad| pad.period - 1);
            total.saturating_add(
                max_pads + replacement.tokens.len().saturating_sub(lane.marker_token_ids.len()),
            )
        })
    });
    let expanded_len = prompt_token_ids.len().saturating_add(replacement_growth);

    let mut expanded = Vec::with_capacity(expanded_len);
    let mut ranges = HashMap::<Modality, Vec<PlaceholderRange>>::new();

    let mut prompt_index = 0;
    while prompt_index < prompt_token_ids.len() {
        let lane = lanes.iter_mut().find(|lane| {
            !lane.replacements.is_empty()
                && prompt_token_ids[prompt_index..].starts_with(lane.marker_token_ids)
        });
        let Some(lane) = lane else {
            expanded.push(prompt_token_ids[prompt_index]);
            prompt_index += 1;
            continue;
        };

        prompt_index += lane.marker_token_ids.len();
        let replacement = lane.replacements.pop_front().expect("lane queue is non-empty");
        debug_assert_eq!(replacement.modality, lane.modality);
        if replacement.tokens.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` expanded to no tokens",
                lane.placeholder_token
            );
        }

        let replacement_len = replacement.tokens.len();
        let structural_prefix = replacement.structural_prefix;
        if structural_prefix > expanded.len() {
            bail_multimodal!(
                "placeholder token `{}` declares a structural prefix of {} token(s), but only {} token(s) precede it",
                lane.placeholder_token,
                structural_prefix,
                expanded.len()
            );
        }
        // Position-dependent alignment padding, computed from the offset the
        // replacement block lands at in the expanded prompt (mirroring
        // vLLM's `_apply_token_matches_with_placeholders` splice-time pad).
        let pad_count = replacement.alignment_pad.map_or(0, |pad| pad.count_at(expanded.len()));
        let range_len = structural_prefix + pad_count + replacement_len;
        let is_embed = {
            let mut mask = Vec::with_capacity(range_len);
            mask.extend(std::iter::repeat_n(false, structural_prefix));
            if let Some(pad) = &replacement.alignment_pad {
                mask.extend(std::iter::repeat_n(
                    pad.token_id as u32 == lane.embed_token_id,
                    pad_count,
                ));
            }
            mask.extend(
                replacement.tokens.iter().map(|&token| token as u32 == lane.embed_token_id),
            );
            WireTensor::from_bool(vec![range_len], mask).map_err(Error::Multimodal)?
        };

        let expanded_offset = expanded.len() - structural_prefix;
        if let Some(pad) = &replacement.alignment_pad {
            expanded.extend(std::iter::repeat_n(pad.token_id as u32, pad_count));
        }
        expanded.extend(replacement.tokens.iter().map(|&token| token as u32));
        ranges.entry(lane.modality).or_default().push(PlaceholderRange {
            offset: expanded_offset,
            length: range_len,
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
        QWEN3_VIDEO_PAD_ID, STEP3_IM_END_ID, STEP3_IM_PATCH_ID, STEP3_IM_START_ID,
        STEP3_PATCH_END_ID, STEP3_PATCH_NEWLINE_ID, STEP3_PATCH_START_ID, step3_vl_tokenizer,
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
                marker_token_ids: vec![marker_token_id],
                embed_token_id,
            },
            replacements,
            items: Vec::new(),
        }
    }

    #[test]
    fn expand_prompt_tokens_matches_multi_token_placeholder() {
        let marker_token_ids = vec![40, 41, 42];
        let replacement = PromptReplacement::repeated(
            Modality::Image,
            "<image-placeholder>",
            QWEN3_IMAGE_PAD_ID as TokenId,
            2,
        );
        let prepared = vec![PreparedMedia {
            modality: Modality::Image,
            placeholder: ResolvedPlaceholder {
                token: "<image-placeholder>".to_string(),
                marker_token_ids: marker_token_ids.clone(),
                embed_token_id: QWEN3_IMAGE_PAD_ID,
            },
            replacements: vec![replacement],
            items: Vec::new(),
        }];
        let mut prompt_token_ids = vec![1, 40, 41, 42, 2];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![1, QWEN3_IMAGE_PAD_ID, QWEN3_IMAGE_PAD_ID, 2]
        );
        assert_eq!(ranges[&Modality::Image][0].offset, 1);
        assert_eq!(ranges[&Modality::Image][0].length, 2);
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
        assert_eq!(tensor.dtype.as_str(), "bool");
        assert_eq!(tensor.shape, vec![expected.len()]);
        assert_eq!(
            tensor.data,
            WireArrayData::RawView(
                expected.iter().map(|value| u8::from(*value)).collect::<Vec<_>>().into(),
            )
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
    fn expand_prompt_tokens_prepends_alignment_pad_at_splice_offset() {
        const PAD_ID: u32 = 7;

        let mut prompt_token_ids = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let prepared = vec![qwen3_image_prepared(vec![
            PromptReplacement::repeated(
                Modality::Image,
                "<|image_pad|>",
                QWEN3_IMAGE_PAD_ID as TokenId,
                2,
            )
            .with_alignment_pad(PAD_ID as TokenId, 4),
        ])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        // The marker lands at offset 1, so the period-4 pad contributes
        // `4 - 1 - 1 % 4 = 2` tokens ahead of the replacement.
        assert_eq!(
            prompt_token_ids,
            vec![1, PAD_ID, PAD_ID, QWEN3_IMAGE_PAD_ID, QWEN3_IMAGE_PAD_ID, 2]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 4);
        assert_bool_mask(&ranges[0], &[false, false, true, true]);
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
    fn expand_prompt_tokens_includes_structural_prefix_in_video_range() {
        const VISION_START_ID: u32 = 151652;
        const VISION_END_ID: u32 = 151653;

        let mut prompt_token_ids = vec![1, VISION_START_ID, QWEN3_VIDEO_PAD_ID, VISION_END_ID, 2];
        let replacement = PromptReplacement::repeated(
            Modality::Video,
            "<|video_pad|>",
            QWEN3_VIDEO_PAD_ID as TokenId,
            3,
        )
        .with_structural_prefix(1);
        let prepared = vec![qwen3_video_prepared(vec![replacement])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();

        let range = &ranges[&Modality::Video][0];
        assert_eq!(range.offset, 1);
        assert_eq!(range.length, 4);
        assert_bool_mask(range, &[false, true, true, true]);
    }

    const QWEN3_AUDIO_PAD_ID: u32 = 151_676;
    fn qwen3_audio_prepared(replacements: Vec<PromptReplacement>) -> PreparedMedia {
        prepared_media(
            Modality::Audio,
            "<|audio_pad|>",
            QWEN3_AUDIO_PAD_ID,
            QWEN3_AUDIO_PAD_ID,
            replacements,
        )
    }

    fn qwen3_audio_replacement(tokens: usize) -> PromptReplacement {
        PromptReplacement::repeated(
            Modality::Audio,
            "<|audio_pad|>",
            QWEN3_AUDIO_PAD_ID as TokenId,
            tokens,
        )
    }

    #[test]
    fn expand_prompt_tokens_interleaves_audio_and_image_prepared_media() {
        // Audio marker precedes the image marker: the image range offset must
        // account for the earlier audio expansion growth.
        let mut prompt_token_ids = vec![1, QWEN3_AUDIO_PAD_ID, 2, LLAMA4_IMAGE_ID, 3];
        let prepared = vec![
            llama4_prepared(vec![llama4_single_tile_replacement()]),
            qwen3_audio_prepared(vec![qwen3_audio_replacement(3)]),
        ];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                QWEN3_AUDIO_PAD_ID,
                QWEN3_AUDIO_PAD_ID,
                QWEN3_AUDIO_PAD_ID,
                2,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                3,
            ]
        );

        let audio_ranges = &ranges[&Modality::Audio];
        assert_eq!(audio_ranges[0].offset, 1);
        assert_eq!(audio_ranges[0].length, 3);
        assert_bool_mask(&audio_ranges[0], &[true, true, true]);

        let image_ranges = &ranges[&Modality::Image];
        assert_eq!(image_ranges[0].offset, 5);
        assert_eq!(image_ranges[0].length, 5);
        assert_bool_mask(&image_ranges[0], &[false, false, true, true, false]);
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

    /// Step3-VL/Step5-VL image prepared media: `<im_patch>` is both the prompt
    /// marker and the embed token; the other five special tokens are
    /// structural.
    fn step3_vl_prepared(replacements: Vec<PromptReplacement>) -> PreparedMedia {
        prepared_media(
            Modality::Image,
            "<im_patch>",
            STEP3_IM_PATCH_ID,
            STEP3_IM_PATCH_ID,
            replacements,
        )
    }

    /// The replacement layout `Step3VLVisionSpec` builds for one image with
    /// two patches (newline flag on the first) plus the global-view span.
    fn step3_vl_expanded_tokens() -> Vec<u32> {
        let mut tokens = vec![STEP3_PATCH_START_ID];
        tokens.extend(std::iter::repeat_n(STEP3_IM_PATCH_ID, 81));
        tokens.push(STEP3_PATCH_END_ID);
        tokens.push(STEP3_PATCH_NEWLINE_ID);
        tokens.push(STEP3_PATCH_START_ID);
        tokens.extend(std::iter::repeat_n(STEP3_IM_PATCH_ID, 81));
        tokens.push(STEP3_PATCH_END_ID);
        tokens.push(STEP3_IM_START_ID);
        tokens.extend(std::iter::repeat_n(STEP3_IM_PATCH_ID, 169));
        tokens.push(STEP3_IM_END_ID);
        tokens
    }

    #[test]
    fn expand_prompt_tokens_marks_only_step3_vl_im_patch_tokens_as_embed() {
        use vllm_text::tokenizer::Tokenizer as _;

        let tokenizer = step3_vl_tokenizer();
        let mut prompt_token_ids = tokenizer.encode("a <im_patch> b", false).unwrap();
        let replacement = step3_vl_expanded_tokens();
        let prepared = vec![step3_vl_prepared(vec![PromptReplacement::sequence(
            Modality::Image,
            "<im_patch>",
            replacement.iter().map(|&token| token as TokenId).collect(),
        )])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        let mut expected = vec![u32::from(b'a'), u32::from(b' ')];
        expected.extend(&replacement);
        expected.extend([u32::from(b' '), u32::from(b'b')]);
        assert_eq!(prompt_token_ids, expected);

        // Two text bytes precede the marker; the span covers both patch spans
        // and the global-view span.
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 2);
        assert_eq!(ranges[0].length, replacement.len());

        let expected_mask =
            replacement.iter().map(|&token| token == STEP3_IM_PATCH_ID).collect::<Vec<_>>();
        assert_bool_mask(&ranges[0], &expected_mask);
    }

    // DeepSeek-V4.1 image spans: every span position carries the image token
    // id; a compressor-alignment pad (id 129265, period 2) is prepended at
    // splice time so spans always start at the same compressor phase.
    const DS_IMAGE_ID: u32 = 129_264;
    const DS_IMAGE_PAD_ID: u32 = 129_265;

    fn deepseek_prepared(span_lens: Vec<usize>) -> PreparedMedia {
        prepared_media(
            Modality::Image,
            "<｜deepseek_image｜>",
            DS_IMAGE_ID,
            DS_IMAGE_ID,
            span_lens
                .into_iter()
                .map(|len| {
                    PromptReplacement::repeated(
                        Modality::Image,
                        "<｜deepseek_image｜>",
                        DS_IMAGE_ID as TokenId,
                        len,
                    )
                    .with_alignment_pad(DS_IMAGE_PAD_ID as TokenId, 2)
                })
                .collect(),
        )
    }

    #[test]
    fn expand_prompt_tokens_skips_alignment_pad_at_odd_offset() {
        // Odd block start: already at the target compressor phase, no pad.
        let mut prompt_token_ids = vec![1, DS_IMAGE_ID, 2];
        let prepared = vec![deepseek_prepared(vec![3])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        assert_eq!(
            prompt_token_ids,
            vec![1, DS_IMAGE_ID, DS_IMAGE_ID, DS_IMAGE_ID, 2]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
        assert_bool_mask(&ranges[0], &[true, true, true]);
    }

    #[test]
    fn expand_prompt_tokens_prepends_alignment_pad_at_even_offset() {
        // Even block start: one pad token aligns the span to phase 1.
        let mut prompt_token_ids = vec![1, 3, DS_IMAGE_ID, 2];
        let prepared = vec![deepseek_prepared(vec![3])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                3,
                DS_IMAGE_PAD_ID,
                DS_IMAGE_ID,
                DS_IMAGE_ID,
                DS_IMAGE_ID,
                2
            ]
        );
        // The range starts at the pad; pads are not embed positions.
        assert_eq!(ranges[0].offset, 2);
        assert_eq!(ranges[0].length, 4);
        assert_bool_mask(&ranges[0], &[false, true, true, true]);
    }

    #[test]
    fn expand_prompt_tokens_alignment_pad_realigns_each_span() {
        // Every image span must start at phase 1 (odd offset) regardless of
        // where the previous span landed.
        let mut prompt_token_ids = vec![1, DS_IMAGE_ID, 2, DS_IMAGE_ID, 3];
        let prepared = vec![deepseek_prepared(vec![2, 2])];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, &prepared).unwrap();
        let ranges = &ranges[&Modality::Image];

        // First span at odd offset 1: no pad (block [1,3)).
        // Second marker originally at index 3; after the first expansion the
        // block starts at offset 4 (even): one pad, span at odd offset 5.
        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                DS_IMAGE_ID,
                DS_IMAGE_ID,
                2,
                DS_IMAGE_PAD_ID,
                DS_IMAGE_ID,
                DS_IMAGE_ID,
                3,
            ]
        );
        assert_eq!((ranges[0].offset, ranges[0].length), (1, 2));
        assert_bool_mask(&ranges[0], &[true, true]);
        assert_eq!((ranges[1].offset, ranges[1].length), (4, 3));
        assert_bool_mask(&ranges[1], &[false, true, true]);
        // Both spans (past the pad) start at odd offsets.
        assert_eq!((ranges[1].offset + 1) % 2, 1);
    }
}
