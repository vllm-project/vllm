// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{
    ImageFrame, Modality, PreprocessedEncoderInputs, PromptReplacement, VisionPreprocessingContext,
};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::cache::{CacheLookup, MmProcessorCache, merge_slots, put_or_inline_and_track};
use super::timing::MM_STAGE_TARGET;
use super::{
    ItemSource, MultimodalModelInfo, PreparedItem, PreparedMedia, VisionModalitySupport, item,
};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary image encoder input.
pub(super) const IMAGE_PRIMARY_KEY: &str = "pixel_values";

impl MultimodalModelInfo {
    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    ///
    /// With the SHM processor cache enabled, cached frames skip preprocessing
    /// entirely: their prompt replacements come from the cache shadow and
    /// their feature data is the shm address item. Cache misses are
    /// preprocessed as one batch. Budget-sensitive processors bypass the
    /// cache (`cache` is `None` for them), so they always preprocess inline.
    ///
    /// A frame carrying a client UUID is keyed by that UUID instead of its
    /// content hash, and its `mm_hash` is the UUID.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "preprocess_image")
    )]
    pub(super) async fn prepare_images(
        &self,
        sources: Vec<ItemSource<Arc<ImageFrame>>>,
        cache: Option<&MmProcessorCache>,
        model_dtype: ModelDtype,
        preprocessing_context: VisionPreprocessingContext,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;

        let mut slots = Vec::with_capacity(sources.len());
        let mut miss_frames = Vec::new();
        let mut miss_uuids = Vec::new();
        let mut miss_hashes = Vec::new();
        for source in sources {
            let (frame, uuid) = match source {
                ItemSource::Cached(item, replacement) => {
                    slots.push(Some((item, replacement)));
                    continue;
                }
                ItemSource::Fetch(frame, uuid) => (frame, uuid),
            };
            // A client UUID keys the item when the cache is enabled;
            // otherwise the item is content-addressed.
            let mm_hash = match (cache, &uuid) {
                (Some(_), Some(uuid)) => uuid.clone(),
                _ => frame.hash.clone(),
            };
            let lookup = match cache {
                Some(cache) => cache.lookup(&mm_hash, uuid),
                None => CacheLookup::Miss(uuid),
            };
            match lookup {
                CacheLookup::Hit(item, replacement) => slots.push(Some((item, replacement))),
                CacheLookup::Miss(uuid) => {
                    slots.push(None);
                    miss_hashes.push(mm_hash);
                    miss_frames.push(frame);
                    miss_uuids.push(uuid);
                }
            }
        }

        let mut misses = Vec::new();
        if !miss_frames.is_empty() {
            let (replacements, mut items) = self
                .preprocess_image_batch(
                    support,
                    miss_frames,
                    miss_uuids,
                    model_dtype,
                    preprocessing_context,
                )
                .await?;
            for (item, replacement, hash) in izip!(&mut items, &replacements, &miss_hashes) {
                item.data = put_or_inline_and_track(
                    cache,
                    hash,
                    replacement,
                    std::mem::take(&mut item.data),
                );
                // UUID-keyed items report the UUID as `mm_hash` (Python
                // parity: `hashes.append(uuid_item)`).
                item.hash = hash.clone();
            }
            misses = izip!(items, replacements).collect();
        }
        let (replacements, items) = merge_slots(slots, misses);

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched image frames as one batch and build per-item
    /// replacements and engine kwargs.
    async fn preprocess_image_batch(
        &self,
        support: &VisionModalitySupport,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
        preprocessing_context: VisionPreprocessingContext,
    ) -> Result<(Vec<PromptReplacement>, Vec<PreparedItem>)> {
        let preprocessed = self.preprocess_images(support, &frames, preprocessing_context).await?;
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let hashes = frames.iter().map(|frame| frame.hash.clone()).collect();
        let items =
            item::build_batched_items(&support.spec, preprocessed, hashes, uuids, model_dtype)?;
        Ok((replacements, items))
    }

    /// Preprocess fetched image frames with the model's resolved vision
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        support: &VisionModalitySupport,
        image_frames: &[Arc<ImageFrame>],
        preprocessing_context: VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs> {
        let processor = Arc::clone(&support.processor);
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            Ok(processor.preprocess_with_context(&images, &preprocessing_context)?)
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}
