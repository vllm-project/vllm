// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use expect_test::expect;
use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use llm_multimodal::MediaContentPart;
use serde_json::json;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::MmKwargValue;
use vllm_tokenizer::test_utils::TestTokenizer;

use super::*;
use crate::multimodal::tests::test_info;

const IMAGE_MARKER_ID: u32 = 200005;
const PAD: u32 = IMAGE_PAD_ID as u32;

/// FNV-1a over little-endian bf16 bits.
fn fnv1a_bf16(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|&value| half::bf16::from_f32(value).to_bits().to_le_bytes())
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

/// Patches match `Inkling2ImageProcessor.preprocess` bit-for-bit in bf16
/// (RGBA checkerboard compositing, grid padding, grayscale conversion).
///
/// Reference digest generated with the Python processor on the same images.
#[test]
fn patches_match_python_processor() {
    let rgba = RgbaImage::from_fn(45, 83, |x, y| {
        Rgba([x * 7 + y * 3, x * y, x + 10 * y, x * 11 + y * 13].map(|v| (v % 256) as u8))
    });
    let rgb = RgbImage::from_fn(81, 40, |x, y| {
        Rgb([x * 5, y * 9, (x ^ y) * 3].map(|v| (v % 256) as u8))
    });
    let gray = GrayImage::from_pixel(1, 1, Luma([77]));
    let images = [rgba.into(), rgb.into(), gray.into()];

    let output = Inkling2ImageProcessor.preprocess(&images).unwrap();

    assert_eq!(output.encoder_input.shape(), [10, 40, 40, 3]);
    assert_eq!(output.feature_token_counts, [6, 3, 1]);
    let ModelSpecificValue::IntTensor { data, shape } = &output.model_specific[VISION_GRID_THW_KEY]
    else {
        panic!("{VISION_GRID_THW_KEY} is not an int tensor");
    };
    assert_eq!(
        (data.as_slice(), shape.as_slice()),
        ([1, 3, 2, 1, 1, 3, 1, 1, 1].as_slice(), [3, 3].as_slice())
    );
    let patches = output.encoder_input.as_f32();
    let patches = patches.as_slice().unwrap();
    assert_eq!(fnv1a_bf16(patches), 0x167dd634e06ac81d);
}

/// Engine inputs follow `Inkling2MultiModalProcessor`: marker kept, pad tokens
/// embedded, patches flat by grid size, and grid batched on CPU.
#[tokio::test]
async fn prepares_engine_inputs_like_python_processor() {
    let tokenizer = TestTokenizer::new().with_regular_token(IMAGE_PLACEHOLDER, IMAGE_MARKER_ID);
    let info = test_info(MODEL_TYPE, json!({"model_type": MODEL_TYPE}), tokenizer);
    // 41x2 RGBA PNG.
    let image = MediaContentPart::ImageUrl {
        url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAACCAYAAADFC/UDAAAAF0lEQVR4nGM8wcDAwMDAwDaYMRPDEAAA6bUBvJDJE7sAAAAASUVORK5CYII=".to_string(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    };
    let mut tokens = vec![11, IMAGE_MARKER_ID, 12, IMAGE_MARKER_ID];

    let features = info
        .prepare_multimodal(
            vec![image.clone(), image],
            &mut tokens,
            ModelDtype::BFloat16,
            None,
        )
        .await
        .unwrap();

    let summary = features
        .iter()
        .map(|feature| {
            let position = &feature.mm_position;
            let is_embed = position.is_embed.as_ref().unwrap().data.as_raw_view().unwrap();
            let is_embed = is_embed.iter().map(|&byte| byte != 0).collect::<Vec<_>>();
            let fields = feature
                .data
                .as_ref()
                .unwrap()
                .iter()
                .map(|(key, elem)| {
                    let Some(MmKwargValue::Tensor(tensor)) = &elem.data else {
                        panic!("{key} is not a tensor");
                    };
                    let values = (key == VISION_GRID_THW_KEY).then(|| {
                        let bytes = tensor.data.as_raw_view().unwrap();
                        bytes
                            .chunks_exact(8)
                            .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
                            .collect::<Vec<_>>()
                    });
                    let summary = (&elem.field, tensor.dtype, tensor.shape.clone(), values);
                    (key.clone(), format!("{summary:?}"))
                })
                .collect::<BTreeMap<_, _>>();
            (position.offset, position.length, is_embed, fields)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        tokens,
        [11, IMAGE_MARKER_ID, PAD, PAD, 12, IMAGE_MARKER_ID, PAD, PAD]
    );
    expect![[r#"
        [
            (
                1,
                3,
                [
                    false,
                    true,
                    true,
                ],
                {
                    "vision_grid_thw": "(Batched(MmBatchedField { keep_on_cpu: true }), I64, [3], Some([1, 1, 2]))",
                    "vision_pixels": "(Flat(MmFlatField { slices: [Slice(SliceSpec { start: Some(0), stop: Some(2), step: None })], dim: 0, keep_on_cpu: false }), Bf16, [2, 40, 40, 3], None)",
                },
            ),
            (
                5,
                3,
                [
                    false,
                    true,
                    true,
                ],
                {
                    "vision_grid_thw": "(Batched(MmBatchedField { keep_on_cpu: true }), I64, [3], Some([1, 1, 2]))",
                    "vision_pixels": "(Flat(MmFlatField { slices: [Slice(SliceSpec { start: Some(0), stop: Some(2), step: None })], dim: 0, keep_on_cpu: false }), Bf16, [2, 40, 40, 3], None)",
                },
            ),
        ]
    "#]].assert_debug_eq(&summary);
}
