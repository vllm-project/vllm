//! Video-modality preparation: per-clip preprocessing, config resolution,
//! and per-item feature build.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{
    FieldLayout, Modality, PreProcessorConfig, PreprocessedEncoderInputs, VideoClip,
};
use thiserror_ext::AsReport as _;
use tracing::warn;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargsItem, MmSharedField, MmSlice,
    SliceSpec,
};

use super::{ModalitySupport, MultimodalModelInfo, PreparedItem, PreparedMedia, tensor};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary video encoder input.
///
/// Video-capable vLLM models read `pixel_values_videos` alongside
/// `video_grid_thw`, mirroring the HF processor output naming.
const VIDEO_PRIMARY_KEY: &str = "pixel_values_videos";

/// Load the video preprocessor config from its dedicated file, falling back
/// to the `video_processor` section of the combined processor config.
///
/// Returns `Ok(None)` when neither source provides one; callers then reuse
/// the image preprocessor config, mirroring HF processor behavior.
pub(super) fn load_video_preprocessor_config(
    video_preprocessor_config_path: Option<&Path>,
    processor_config_path: Option<&Path>,
) -> Result<Option<PreProcessorConfig>> {
    if let Some(path) = video_preprocessor_config_path {
        let text = fs::read_to_string(path).map_err(|error| {
            multimodal!("failed to read video_preprocessor_config.json: {error}")
        })?;
        let config = PreProcessorConfig::from_json(&text).map_err(|error| {
            multimodal!("failed to parse video_preprocessor_config.json: {error}")
        })?;
        return Ok(Some(config));
    }

    if let Some(path) = processor_config_path {
        let text = fs::read_to_string(path)
            .map_err(|error| multimodal!("failed to read processor_config.json: {error}"))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| multimodal!("failed to parse processor_config.json: {error}"))?;
        if let Some(video_processor) = value.get("video_processor") {
            let config =
                PreProcessorConfig::from_value(video_processor.clone()).map_err(|error| {
                    multimodal!(
                        "failed to parse video_processor from processor_config.json: {error}"
                    )
                })?;
            return Ok(Some(config));
        }
    }

    Ok(None)
}

impl MultimodalModelInfo {
    /// Preprocess fetched video clips one at a time and build per-item
    /// features.
    ///
    /// Unlike images, each clip runs through the preprocessor independently
    /// (a batch of one), so its tensors are complete per item and need no
    /// cross-item slicing.
    pub(super) async fn prepare_videos(
        &self,
        clips: Vec<Arc<VideoClip>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.video.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Video.to_string(),
        })?;
        let mut replacements = Vec::with_capacity(clips.len());
        let mut items = Vec::with_capacity(clips.len());

        for (clip, uuid) in izip!(&clips, uuids) {
            let preprocessed = self.preprocess_video_clip(support, Arc::clone(clip)).await?;
            let mut clip_replacements =
                self.spec
                    .prompt_replacements_for(&self.context, &preprocessed, Modality::Video)?;
            if clip_replacements.len() != 1 {
                bail_multimodal!(
                    "expected exactly one prompt replacement per video clip, got {}",
                    clip_replacements.len()
                );
            }
            replacements.push(clip_replacements.pop().unwrap());
            items.push(self.build_video_item(
                preprocessed,
                clip.hash.clone(),
                uuid,
                model_dtype,
            )?);
        }

        Ok(PreparedMedia {
            modality: Modality::Video,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess one decoded video clip with the model's resolved vision
    /// processor.
    async fn preprocess_video_clip(
        &self,
        support: &ModalitySupport,
        clip: Arc<VideoClip>,
    ) -> Result<PreprocessedEncoderInputs> {
        let config = support.config.clone();
        let processor = support.processor;

        tokio::task::spawn_blocking(move || {
            // Prefer the borrowed-RGB fast path, which avoids materializing a
            // `DynamicImage` per sampled frame after media decode.
            if let Some(rgb_video) = clip.rgb_video() {
                match rgb_video.frame_refs() {
                    Ok(frame_refs) => match processor.preprocess_video_rgb(&frame_refs, &config) {
                        Ok(preprocessed) => return Ok(preprocessed),
                        Err(error) => warn!(
                            error = %error.as_report(),
                            "RGB video preprocessing fast path failed; falling back to materialized frames"
                        ),
                    },
                    Err(error) => warn!(
                        error,
                        "RGB video frame refs are invalid; falling back to materialized frames"
                    ),
                }
            }

            let frames = clip.materialized_frames().map_err(|error| multimodal!("{error}"))?;
            Ok(processor.preprocess_video(&frames, &config)?)
        })
        .await
        .map_err(|error| multimodal!("video preprocessing task failed: {error}"))?
    }

    /// Convert one preprocessed video clip into engine kwargs.
    ///
    /// The clip is a batch of one, so no per-item slicing is required: the
    /// primary tensor ships as a full-range flat field (the engine re-batches
    /// flat fields by concatenating along the declared dim, matching vLLM's
    /// `flat_from_sizes` treatment of video patches), and batched metadata
    /// tensors drop their singleton batch axis.
    fn build_video_item(
        &self,
        preprocessed: PreprocessedEncoderInputs,
        hash: String,
        uuid: Option<String>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedItem> {
        let tensors = tensor::collect_tensors(preprocessed, VIDEO_PRIMARY_KEY, model_dtype)?;

        let mut data = MmKwargsItem::new();
        for (key, tensor) in tensors {
            let keep_on_cpu = self.spec.keep_on_cpu_keys.contains(&key);
            let (value, field) = if key == VIDEO_PRIMARY_KEY {
                let len = tensor
                    .first_dim()
                    .ok_or_else(|| multimodal!("video encoder input `{key}` is not a tensor"))?;
                (
                    tensor,
                    MmField::Flat(MmFlatField {
                        slices: vec![MmSlice::Slice(SliceSpec {
                            start: Some(0),
                            stop: Some(len as isize),
                            step: None,
                        })],
                        dim: 0,
                        keep_on_cpu,
                    }),
                )
            } else if matches!(
                self.spec.field_layouts.get(&key),
                Some(FieldLayout::Batched)
            ) {
                (
                    tensor.batched_value_at(0)?,
                    MmField::Batched(MmBatchedField { keep_on_cpu }),
                )
            } else {
                (
                    tensor,
                    MmField::Shared(MmSharedField {
                        batch_size: 1,
                        keep_on_cpu,
                    }),
                )
            };

            data.insert(
                key,
                MmFieldElem {
                    data: Some(value.try_into()?),
                    field,
                },
            );
        }

        Ok(PreparedItem { data, hash, uuid })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use llm_multimodal::ModelSpecificValue;
    use ndarray::ArrayD;
    use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

    use super::super::tests::{
        QWEN3_IMAGE_PAD_ID, QWEN3_VIDEO_PAD_ID, qwen3_vl_info, qwen3_vl_tokenizer,
    };
    use super::super::{MultimodalConfigFiles, MultimodalModelInfo};
    use super::*;

    #[test]
    fn from_paths_resolves_video_config_from_dedicated_file_or_processor_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            serde_json::json!({
                "model_type": "qwen3_vl",
                "image_token_id": QWEN3_IMAGE_PAD_ID,
                "video_token_id": QWEN3_VIDEO_PAD_ID,
            })
            .to_string(),
        )
        .unwrap();

        let info_for = |files: MultimodalConfigFiles<'_>| {
            MultimodalModelInfo::from_paths(
                "qwen3-vl-test".to_string(),
                Some("qwen3_vl".to_string()),
                files,
                Arc::new(qwen3_vl_tokenizer()),
            )
        };

        // Dedicated video preprocessor config file.
        let video_config_path = dir.path().join("video_preprocessor_config.json");
        std::fs::write(&video_config_path, r#"{"size":{"shortest_edge":128}}"#).unwrap();
        let info = info_for(MultimodalConfigFiles {
            config: Some(&config_path),
            video_preprocessor_config: Some(&video_config_path),
            ..Default::default()
        })
        .unwrap()
        .unwrap();
        assert!(info.video.is_some());

        // `video_processor` section of the combined processor config.
        let processor_config_path = dir.path().join("processor_config.json");
        std::fs::write(
            &processor_config_path,
            r#"{"video_processor":{"size":{"shortest_edge":128}}}"#,
        )
        .unwrap();
        let info = info_for(MultimodalConfigFiles {
            config: Some(&config_path),
            processor_config: Some(&processor_config_path),
            ..Default::default()
        })
        .unwrap()
        .unwrap();
        assert!(info.video.is_some());

        // Neither source: video support still resolves on the image config.
        let info = info_for(MultimodalConfigFiles {
            config: Some(&config_path),
            ..Default::default()
        })
        .unwrap()
        .unwrap();
        assert!(info.video.is_some());

        // Malformed dedicated file is a real error, not a silent fallback.
        std::fs::write(&video_config_path, r#"{"size""#).unwrap();
        let error = match info_for(MultimodalConfigFiles {
            config: Some(&config_path),
            video_preprocessor_config: Some(&video_config_path),
            ..Default::default()
        }) {
            Err(error) => error,
            Ok(_) => panic!("malformed video preprocessor config should fail"),
        };
        assert!(matches!(
            error,
            Error::Multimodal(message)
                if message.contains("failed to parse video_preprocessor_config.json")
        ));
    }

    #[test]
    fn build_video_item_names_primary_tensor_and_layouts() {
        let info = qwen3_vl_info();
        // One clip flattened to 6 patches with 4 features each.
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::zeros(vec![6, 4]),
            feature_token_counts: vec![6],
            item_sizes: vec![(32, 32)],
            model_specific: HashMap::from([
                (
                    "video_grid_thw".to_string(),
                    ModelSpecificValue::int_2d(vec![1, 2, 3], 1, 3),
                ),
                (
                    "patches_per_video".to_string(),
                    ModelSpecificValue::int_1d(vec![6]),
                ),
            ]),
        };

        let item = info
            .build_video_item(
                preprocessed,
                "<hash>".to_string(),
                None,
                ModelDtype::Float32,
            )
            .unwrap();

        let primary = &item.data[VIDEO_PRIMARY_KEY];
        assert!(matches!(
            &primary.field,
            MmField::Flat(MmFlatField { slices, dim: 0, .. })
                if matches!(
                    slices.as_slice(),
                    [MmSlice::Slice(SliceSpec { start: Some(0), stop: Some(6), step: None })]
                )
        ));

        // Batched metadata drops its singleton batch axis per item.
        let grid = &item.data["video_grid_thw"];
        assert!(matches!(&grid.field, MmField::Batched(_)));
        let MmKwargValue::Tensor(grid_tensor) = grid.data.as_ref().unwrap() else {
            panic!("expected tensor value for video_grid_thw");
        };
        assert_eq!(grid_tensor.shape, vec![3]);

        assert_eq!(item.hash, "<hash>");
    }
}
