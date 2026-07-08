//! Chat-layer multimodal media preparation.
//!
//! This module owns the multimodal path for chat requests: it extracts media
//! parts from structured chat messages, fetches and preprocesses them through
//! `llm-multimodal` one modality at a time, expands rendered prompt
//! placeholders after tokenization, and builds the engine-facing
//! `MmFeatures` payload.
//!
//! Raw media stays above `vllm-text`; this module lowers it into token IDs and
//! opaque tensor payloads before the request is handed to text generation.
//!
//! Modalities differ in payload types and preprocessing shape (images are
//! preprocessed as one batch, videos one clip at a time), so each supported
//! modality resolves into its own [`ModalitySupport`] and produces a common
//! [`PreparedMedia`] intermediate. Everything downstream of that contract —
//! placeholder expansion and feature assembly — is shared across modalities.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use itertools::izip;
use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, ImageFrame, MediaConnector, MediaConnectorConfig,
    MediaContentPart, Modality, ModelMetadata, ModelProcessorSpec, ModelRegistry,
    PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, Tokenizer as TokenResolver,
    TrackedMedia, VideoClip, VisionPreProcessor, VisionProcessorRegistry,
};
use thiserror_ext::AsReport as _;
use tracing::warn;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmFeatureSpec, MmFeatures, MmField, MmFieldElem, MmFlatField, MmKwargsItem,
    MmSharedField, MmSlice, PlaceholderRange, SliceSpec,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use crate::error::{Error, Result, bail_multimodal, multimodal};
use crate::renderer::RenderedPrompt;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

mod tensor;

/// Forward-kwargs name of the primary video encoder input.
///
/// Video-capable vLLM models read `pixel_values_videos` alongside
/// `video_grid_thw`, mirroring the HF processor output naming.
const VIDEO_PRIMARY_KEY: &str = "pixel_values_videos";

/// Resolved multimodal support for one loaded model.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    spec: ResolvedMultimodalSpec,
    image: Option<ModalitySupport>,
    video: Option<ModalitySupport>,
    media_connector: Arc<MediaConnector>,
}

/// Model metadata and tokenizer access shared by all multimodal specs.
#[derive(Clone)]
struct MultimodalModelContext {
    model_id: String,
    model_type: Option<String>,
    config: serde_json::Value,
    tokenizer: TokenizerResolver,
}

impl MultimodalModelContext {
    fn metadata(&self) -> ModelMetadata<'_> {
        ModelMetadata {
            model_id: &self.model_id,
            tokenizer: &self.tokenizer,
            config: &self.config,
        }
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.0.as_ref()
    }

    /// Resolve a static model processor spec for one loaded model.
    fn resolve_model_spec(&self) -> Option<&'static dyn ModelProcessorSpec> {
        static REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);
        REGISTRY.lookup(&self.metadata())
    }

    /// Resolve a static vision preprocessor for one loaded model.
    ///
    /// The vision preprocessor serves both the image and video modalities.
    fn resolve_vision_processor(&self) -> Option<&'static dyn VisionPreProcessor> {
        static REGISTRY: LazyLock<VisionProcessorRegistry> =
            LazyLock::new(VisionProcessorRegistry::with_defaults);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }
}

/// Static model-specific tensor-layout behavior shared across modalities.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    field_layouts: HashMap<String, FieldLayout>,
    keep_on_cpu_keys: HashSet<String>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec) -> Self {
        Self {
            raw,
            field_layouts: raw.field_layouts(),
            keep_on_cpu_keys: raw.keep_on_cpu_keys().into_iter().collect(),
        }
    }

    fn prompt_replacements_for(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> Result<Vec<PromptReplacement>> {
        self.raw
            .prompt_replacements_for(&context.metadata(), preprocessed, modality)
            .map_err(|error| multimodal!("{error}"))
    }
}

/// Resolved placeholder tokens for one modality.
#[derive(Clone)]
struct ResolvedPlaceholder {
    token: String,
    /// The token ID emitted for `token` in the rendered prompt.
    marker_token_id: u32,
    /// The model-declared embed token ID marked in `is_embed` masks.
    embed_token_id: u32,
}

impl ResolvedPlaceholder {
    fn resolve(
        raw: &'static dyn ModelProcessorSpec,
        context: &MultimodalModelContext,
        modality: Modality,
    ) -> Result<Self> {
        let metadata = context.metadata();
        let token = raw
            .placeholder_token_for(&metadata, modality)
            .map_err(|error| multimodal!("{error}"))?;
        // This is the rendered prompt marker, so resolve it from the token
        // string itself. Do not use `ModelProcessorSpec::placeholder_token_id_for()`:
        // for some specs that ID is the replacement vision/patch token,
        // not necessarily the token ID of the placeholder token.
        let marker_token_id = context.tokenizer().token_to_id(&token).ok_or_else(|| {
            multimodal!("placeholder token `{token}` is not in the tokenizer vocabulary")
        })?;
        let embed_token_id = raw
            .placeholder_token_id_for(&metadata, modality)
            .map_err(|error| multimodal!("{error}"))? as u32;

        Ok(Self {
            token,
            marker_token_id,
            embed_token_id,
        })
    }
}

/// Static per-modality vision preprocessor plus its loaded config and
/// resolved placeholder tokens.
#[derive(Clone)]
struct ModalitySupport {
    placeholder: ResolvedPlaceholder,
    processor: &'static dyn VisionPreProcessor,
    config: PreProcessorConfig,
}

/// Model-repo config file locations consumed by multimodal support.
#[derive(Debug, Default, Clone, Copy)]
pub struct MultimodalConfigFiles<'a> {
    pub config: Option<&'a Path>,
    pub preprocessor_config: Option<&'a Path>,
    /// Video-specific preprocessor config (`video_preprocessor_config.json`).
    pub video_preprocessor_config: Option<&'a Path>,
    /// Combined processor config (`processor_config.json`), whose
    /// `video_processor` section is the fallback video config source.
    pub processor_config: Option<&'a Path>,
}

/// Request-scoped fetched media, split per modality with tracker UUID
/// metadata preserved in request order.
struct FetchedMedia {
    images: Vec<Arc<ImageFrame>>,
    image_uuids: Vec<Option<String>>,
    videos: Vec<Arc<VideoClip>>,
    video_uuids: Vec<Option<String>>,
}

/// One modality's preprocessed output, ready for the shared expansion and
/// feature-assembly tail.
struct PreparedMedia {
    modality: Modality,
    placeholder: ResolvedPlaceholder,
    /// One replacement per media item, in request order.
    replacements: Vec<PromptReplacement>,
    /// One entry per media item, aligned with `replacements`.
    items: Vec<PreparedItem>,
}

/// One media item's complete engine kwargs plus identity metadata.
struct PreparedItem {
    data: MmKwargsItem,
    hash: String,
    uuid: Option<String>,
}

impl PreparedMedia {
    /// Detach this modality's pending replacements as an expansion lane.
    fn expansion_lane(&mut self) -> ExpansionLane {
        ExpansionLane {
            modality: self.modality,
            marker_token_id: self.placeholder.marker_token_id,
            embed_token_id: self.placeholder.embed_token_id,
            placeholder_token: self.placeholder.token.clone(),
            replacements: std::mem::take(&mut self.replacements).into(),
        }
    }
}

impl MultimodalModelInfo {
    /// Load and resolve multimodal support from model files.
    ///
    /// Returns `Ok(Some(_))` only when the model spec is registered and at
    /// least one modality resolves. File read/parse failures are real errors;
    /// unsupported model families are logged and returned as `Ok(None)`.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        files: MultimodalConfigFiles<'_>,
        tokenizer: DynTokenizer,
    ) -> Result<Option<Self>> {
        let config = match files.config {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .map_err(|error| multimodal!("failed to read config.json: {error}"))?;
                serde_json::from_str(&text)
                    .map_err(|error| multimodal!("failed to parse config.json: {error}"))?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let preprocessor_config = match files.preprocessor_config {
            Some(path) => {
                let text = fs::read_to_string(path).map_err(|error| {
                    multimodal!("failed to read preprocessor_config.json: {error}")
                })?;
                PreProcessorConfig::from_json(&text).map_err(|error| {
                    multimodal!("failed to parse preprocessor_config.json: {error}")
                })?
            }
            None => PreProcessorConfig::default(),
        };
        let video_preprocessor_config = load_video_preprocessor_config(
            files.video_preprocessor_config,
            files.processor_config,
        )?;

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        Self::from_loaded(context, preprocessor_config, video_preprocessor_config)
    }

    /// Resolve multimodal support from an assembled context and parsed
    /// preprocessor configs.
    fn from_loaded(
        context: MultimodalModelContext,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: Option<PreProcessorConfig>,
    ) -> Result<Option<Self>> {
        let Some(raw_spec) = context.resolve_model_spec() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "multimodal model spec is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        let Some(processor) = context.resolve_vision_processor() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "vision processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        // The spec's modality limits double as its capability declaration:
        // a modality key is present iff this model (with this config) can
        // consume that modality.
        let limits = raw_spec
            .modality_limits(&context.metadata())
            .map_err(|error| multimodal!("{error}"))?;

        let image = if limits.contains_key(&Modality::Image) {
            Some(ModalitySupport {
                placeholder: ResolvedPlaceholder::resolve(raw_spec, &context, Modality::Image)?,
                processor,
                config: preprocessor_config.clone(),
            })
        } else {
            None
        };

        let video = if limits.contains_key(&Modality::Video) {
            let placeholder = ResolvedPlaceholder::resolve(raw_spec, &context, Modality::Video)?;
            // Placeholder expansion attributes markers to modalities by token
            // ID, so a marker shared with the image modality is ambiguous.
            let image_marker = image.as_ref().map(|image| image.placeholder.marker_token_id);
            if image_marker == Some(placeholder.marker_token_id) {
                warn!(
                    model_id = context.model_id,
                    token = placeholder.token,
                    "video placeholder token collides with the image placeholder; disabling video support for this model"
                );
                None
            } else {
                Some(ModalitySupport {
                    placeholder,
                    processor,
                    config: video_preprocessor_config
                        .unwrap_or_else(|| preprocessor_config.clone()),
                })
            }
        } else {
            None
        };

        if image.is_none() && video.is_none() {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "no multimodal modality resolved; disabling multimodal support for this model"
            );
            return Ok(None);
        }

        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .map_err(|error| multimodal!("{error}"))?,
        );

        Ok(Some(Self {
            context,
            spec: ResolvedMultimodalSpec::new(raw_spec),
            image,
            video,
            media_connector,
        }))
    }

    /// Return the template-visible placeholder token for one modality, when
    /// this model supports it.
    ///
    /// The HF renderer uses these tokens while flattening media content in
    /// string content format.
    pub fn placeholder_token(&self, modality: Modality) -> Option<&str> {
        self.modality_support(modality)
            .map(|support| support.placeholder.token.as_str())
    }

    fn modality_support(&self, modality: Modality) -> Option<&ModalitySupport> {
        match modality {
            Modality::Image => self.image.as_ref(),
            Modality::Video => self.video.as_ref(),
            _ => None,
        }
    }
}

/// Load the video preprocessor config from its dedicated file, falling back
/// to the `video_processor` section of the combined processor config.
///
/// Returns `Ok(None)` when neither source provides one; callers then reuse
/// the image preprocessor config, mirroring HF processor behavior.
fn load_video_preprocessor_config(
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

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their media placeholders are expanded, and
/// preprocessed media features are attached for engine-core transport.
pub(crate) async fn finalize_rendered_prompt(
    request: &ChatRequest,
    rendered: RenderedPrompt,
    info: Option<&MultimodalModelInfo>,
    model_dtype: ModelDtype,
) -> Result<(Prompt, Option<MmFeatures>)> {
    if !request.has_multimodal() {
        return Ok((rendered.prompt, None));
    }
    let info = info.ok_or(Error::UnsupportedMultimodalRenderer)?;
    let mut prompt_token_ids = match rendered.prompt {
        Prompt::Text(prompt) => info
            .context
            .tokenizer()
            .encode(&prompt, request.add_special_tokens)
            .map_err(|error| multimodal!("{error}"))?,
        Prompt::TokenIds(token_ids) => token_ids,
    };
    let media_parts = extract_media_parts(request)?;
    let prepared = info.prepare_multimodal(media_parts, &mut prompt_token_ids, model_dtype).await?;

    Ok((Prompt::TokenIds(prompt_token_ids), Some(prepared)))
}

/// Extract media parts from chat messages in message/content order.
///
/// Assistant history is skipped because generated assistant blocks are already
/// represented as text for prompt rendering in this crate.
fn extract_media_parts(request: &ChatRequest) -> Result<Vec<MediaContentPart>> {
    let mut all_parts = Vec::new();
    for message in &request.messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        for part in parts {
            match part {
                ChatContentPart::Text { .. } => {}
                ChatContentPart::ImageUrl {
                    image_url,
                    detail,
                    uuid,
                } => all_parts.push(MediaContentPart::ImageUrl {
                    url: image_url.clone(),
                    detail: *detail,
                    uuid: uuid.clone(),
                }),
                ChatContentPart::VideoUrl { video_url, uuid } => {
                    all_parts.push(MediaContentPart::VideoUrl {
                        url: video_url.clone(),
                        uuid: uuid.clone(),
                    })
                }
            }
        }
    }
    Ok(all_parts)
}

/// Return the modality a media content part belongs to.
fn part_modality(part: &MediaContentPart) -> Modality {
    match part {
        MediaContentPart::Text { .. } => unreachable!("text parts are not media"),
        MediaContentPart::ImageUrl { .. } | MediaContentPart::ImageData { .. } => Modality::Image,
        MediaContentPart::ImageEmbeds { .. } => Modality::ImageEmbeds,
        MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => Modality::Video,
    }
}

impl MultimodalModelInfo {
    /// Run media fetch, per-modality preprocessing, prompt expansion, and
    /// feature build.
    ///
    /// `prompt_token_ids` is mutated in place because placeholder expansion
    /// changes both the final prompt and the offsets recorded in
    /// `PlaceholderRange`.
    async fn prepare_multimodal(
        &self,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
        model_dtype: ModelDtype,
    ) -> Result<MmFeatures> {
        if media_parts.is_empty() {
            return Ok(Vec::new());
        }
        let media_parts_len = media_parts.len();

        // Reject unsupported modalities before any fetch work happens.
        // TODO: also enforce per-modality item-count limits, aligned with the
        // engine's `--limit-mm-per-prompt` semantics.
        for part in &media_parts {
            let modality = part_modality(part);
            if self.modality_support(modality).is_none() {
                return Err(Error::UnsupportedModality {
                    modality: modality.to_string(),
                });
            }
        }

        let fetched = self.fetch_media(media_parts).await?;

        let mut prepared = Vec::new();
        if !fetched.images.is_empty() {
            prepared
                .push(self.prepare_images(fetched.images, fetched.image_uuids, model_dtype).await?);
        }
        if !fetched.videos.is_empty() {
            prepared
                .push(self.prepare_videos(fetched.videos, fetched.video_uuids, model_dtype).await?);
        }

        let lanes = prepared.iter_mut().map(PreparedMedia::expansion_lane).collect();
        let mut ranges = expand_prompt_token_ids(prompt_token_ids, lanes)?;

        let mut features = Vec::with_capacity(media_parts_len);
        for lane in prepared {
            let lane_ranges = ranges.remove(&lane.modality).unwrap_or_default();
            if lane_ranges.len() != lane.items.len() {
                bail_multimodal!(
                    "number of expanded `{}` placeholders {} does not match number of media items {}",
                    lane.modality,
                    lane_ranges.len(),
                    lane.items.len()
                );
            }
            for (item, range) in izip!(lane.items, lane_ranges) {
                features.push(MmFeatureSpec {
                    data: Some(item.data),
                    modality: lane.modality.to_string(),
                    identifier: item.uuid.unwrap_or_else(|| item.hash.clone()),
                    mm_position: range,
                    mm_hash: Some(item.hash),
                });
            }
        }
        // Mirror the Python frontend (`argsort_mm_positions`): features are
        // ordered by their placeholder position in the prompt.
        features.sort_by_key(|feature| feature.mm_position.offset);

        if features.len() != media_parts_len {
            bail_multimodal!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts_len
            );
        }
        Ok(features)
    }

    /// Fetch all media parts and split them per modality, preserving their
    /// request-order UUID metadata.
    async fn fetch_media(&self, media_parts: Vec<MediaContentPart>) -> Result<FetchedMedia> {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        for part in media_parts {
            tracker.push_part(part).map_err(|error| multimodal!("{error}"))?;
        }

        let mut tracker_output =
            tracker.finalize().await.map_err(|error| multimodal!("{error}"))?;

        let images = tracker_output
            .data
            .remove(&Modality::Image)
            .unwrap_or_default()
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Image(frame) => Ok(frame),
                _ => Err(multimodal!(
                    "tracker returned non-image media for the image modality"
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let image_uuids = tracker_output.uuids.remove(&Modality::Image).unwrap_or_default();

        let videos = tracker_output
            .data
            .remove(&Modality::Video)
            .unwrap_or_default()
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Video(clip) => Ok(clip),
                _ => Err(multimodal!(
                    "tracker returned non-video media for the video modality"
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let video_uuids = tracker_output.uuids.remove(&Modality::Video).unwrap_or_default();

        Ok(FetchedMedia {
            images,
            image_uuids,
            videos,
            video_uuids,
        })
    }

    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().expect("image support is checked before fetch");
        let preprocessed = self.preprocess_images(support, &frames).await?;
        let replacements =
            self.spec
                .prompt_replacements_for(&self.context, &preprocessed, Modality::Image)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let items = self.build_image_items(preprocessed, &frames, uuids, model_dtype)?;

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched image frames with the model's resolved vision
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        support: &ModalitySupport,
        image_frames: &[Arc<ImageFrame>],
    ) -> Result<PreprocessedEncoderInputs> {
        let config = support.config.clone();
        let processor = support.processor;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            processor.preprocess(&images, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }

    /// Preprocess fetched video clips one at a time and build per-item
    /// features.
    ///
    /// Unlike images, each clip runs through the preprocessor independently
    /// (a batch of one), so its tensors are complete per item and need no
    /// cross-item slicing.
    async fn prepare_videos(
        &self,
        clips: Vec<Arc<VideoClip>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.video.as_ref().expect("video support is checked before fetch");
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
            processor.preprocess_video(&frames, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("video preprocessing task failed: {error}"))?
    }

    /// Convert one batch of preprocessed image tensors into per-item engine
    /// kwargs.
    ///
    /// Tensor fields are sliced per item according to the model spec's field
    /// layout declarations.
    fn build_image_items(
        &self,
        preprocessed: PreprocessedEncoderInputs,
        frames: &[Arc<ImageFrame>],
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<Vec<PreparedItem>> {
        let len = frames.len();
        let tensors = tensor::collect_tensors(preprocessed, "pixel_values", model_dtype)?;

        let mut items = Vec::with_capacity(len);
        for (index, (frame, uuid)) in izip!(frames, uuids).enumerate() {
            let mut data = MmKwargsItem::new();
            for (key, tensor) in &tensors {
                let keep_on_cpu = self.spec.keep_on_cpu_keys.contains(key);
                let (value, field) = match self.spec.field_layouts.get(key) {
                    Some(FieldLayout::Batched) => (
                        tensor.batched_value_at(index)?,
                        MmField::Batched(MmBatchedField { keep_on_cpu }),
                    ),
                    Some(FieldLayout::Flat { sizes_key }) => {
                        let sizes = tensors.get(sizes_key).ok_or_else(|| {
                            multimodal!("flat tensor sizes key `{sizes_key}` is missing")
                        })?;
                        let (start, end) = tensor::flat_range_for_index(sizes, sizes_key, index)?;
                        (
                            tensor.flat_value_range(start, end)?,
                            MmField::Flat(MmFlatField {
                                slices: vec![MmSlice::Slice(SliceSpec {
                                    start: Some(0),
                                    stop: Some((end - start) as isize),
                                    step: None,
                                })],
                                dim: 0,
                                keep_on_cpu,
                            }),
                        )
                    }
                    None => (
                        tensor.clone(),
                        MmField::Shared(MmSharedField {
                            batch_size: len,
                            keep_on_cpu,
                        }),
                    ),
                };

                data.insert(
                    key.clone(),
                    MmFieldElem {
                        data: Some(value.try_into()?),
                        field,
                    },
                );
            }

            items.push(PreparedItem {
                data,
                hash: frame.hash.clone(),
                uuid,
            });
        }

        Ok(items)
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

/// One modality's queue of pending placeholder replacements for prompt
/// expansion.
struct ExpansionLane {
    modality: Modality,
    marker_token_id: u32,
    embed_token_id: u32,
    placeholder_token: String,
    replacements: VecDeque<PromptReplacement>,
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
fn expand_prompt_token_ids(
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

/// Adapter from the frontend tokenizer trait to `llm-multimodal`.
#[derive(Clone)]
struct TokenizerResolver(DynTokenizer);

impl TokenResolver for TokenizerResolver {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        self.0.encode(text, false).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::TokenId;
    use vllm_engine_core_client::protocol::multimodal::MmKwargValue;
    use vllm_engine_core_client::protocol::tensor::WireArrayData;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    const LLAMA4_IMAGE_START_ID: u32 = 200088;
    const LLAMA4_IMAGE_END_ID: u32 = 200089;
    const LLAMA4_IMAGE_ID: u32 = 200090;
    const LLAMA4_PATCH_ID: u32 = 200092;
    const LLAMA4_TILE_X_SEPARATOR_ID: u32 = 200093;
    const LLAMA4_TILE_Y_SEPARATOR_ID: u32 = 200094;

    const QWEN3_IMAGE_PAD_ID: u32 = 151655;
    const QWEN3_VIDEO_PAD_ID: u32 = 151656;

    fn llama4_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_start|>", LLAMA4_IMAGE_START_ID)
            .with_regular_token("<|image_end|>", LLAMA4_IMAGE_END_ID)
            .with_regular_token("<|image|>", LLAMA4_IMAGE_ID)
            .with_regular_token("<|patch|>", LLAMA4_PATCH_ID)
            .with_regular_token("<|tile_x_separator|>", LLAMA4_TILE_X_SEPARATOR_ID)
            .with_regular_token("<|tile_y_separator|>", LLAMA4_TILE_Y_SEPARATOR_ID)
    }

    fn qwen3_vl_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_pad|>", QWEN3_IMAGE_PAD_ID)
            .with_regular_token("<|video_pad|>", QWEN3_VIDEO_PAD_ID)
    }

    fn test_info(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
    ) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        MultimodalModelInfo::from_loaded(context, PreProcessorConfig::default(), None)
            .unwrap()
            .unwrap_or_else(|| panic!("{model_type} multimodal support should resolve"))
    }

    fn llama4_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "llama4",
            "image_token_index": LLAMA4_PATCH_ID,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        test_info("llama4", config, llama4_tokenizer())
    }

    fn qwen3_vl_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "qwen3_vl",
            "image_token_id": QWEN3_IMAGE_PAD_ID,
            "video_token_id": QWEN3_VIDEO_PAD_ID,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        test_info("qwen3_vl", config, qwen3_vl_tokenizer())
    }

    /// Build an expansion lane for one modality of `info`.
    fn lane(
        info: &MultimodalModelInfo,
        modality: Modality,
        replacements: Vec<PromptReplacement>,
    ) -> ExpansionLane {
        let placeholder = &info
            .modality_support(modality)
            .unwrap_or_else(|| panic!("{modality} support should resolve"))
            .placeholder;
        ExpansionLane {
            modality,
            marker_token_id: placeholder.marker_token_id,
            embed_token_id: placeholder.embed_token_id,
            placeholder_token: placeholder.token.clone(),
            replacements: replacements.into(),
        }
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
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let lanes = vec![lane(
            &info,
            Modality::Image,
            vec![llama4_multi_tile_replacement()],
        )];

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
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, 2, 3];
        let lanes = vec![lane(
            &info,
            Modality::Image,
            vec![llama4_single_tile_replacement()],
        )];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_ignores_empty_replacements() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![lane(&info, Modality::Image, Vec::new())];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap();

        assert!(ranges.is_empty());
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_leaves_prompt_unchanged_when_later_placeholder_missing() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![lane(
            &info,
            Modality::Image,
            vec![
                llama4_single_tile_replacement(),
                llama4_single_tile_replacement(),
            ],
        )];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![lane(
            &info,
            Modality::Image,
            vec![PromptReplacement::sequence(
                Modality::Image,
                "<|image|>",
                Vec::new(),
            )],
        )];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let lanes = vec![lane(
            &info,
            Modality::Image,
            vec![
                llama4_single_tile_replacement(),
                llama4_single_tile_replacement(),
            ],
        )];

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
    fn qwen3_vl_resolves_image_and_video_support() {
        let info = qwen3_vl_info();

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<|image_pad|>")
        );
        assert_eq!(
            info.placeholder_token(Modality::Video),
            Some("<|video_pad|>")
        );
        assert_ne!(
            info.image.as_ref().unwrap().placeholder.marker_token_id,
            info.video.as_ref().unwrap().placeholder.marker_token_id,
        );
    }

    #[test]
    fn llama4_resolves_image_support_only() {
        let info = llama4_info();

        assert_eq!(info.placeholder_token(Modality::Image), Some("<|image|>"));
        assert_eq!(info.placeholder_token(Modality::Video), None);
    }

    #[test]
    fn expand_prompt_tokens_interleaves_image_and_video_lanes() {
        let info = qwen3_vl_info();
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
            lane(
                &info,
                Modality::Image,
                vec![
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
                ],
            ),
            lane(
                &info,
                Modality::Video,
                vec![PromptReplacement::repeated(
                    Modality::Video,
                    "<|video_pad|>",
                    QWEN3_VIDEO_PAD_ID as TokenId,
                    4,
                )],
            ),
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
        let info = qwen3_vl_info();
        let mut prompt_token_ids = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let lanes = vec![
            lane(
                &info,
                Modality::Image,
                vec![PromptReplacement::repeated(
                    Modality::Image,
                    "<|image_pad|>",
                    QWEN3_IMAGE_PAD_ID as TokenId,
                    2,
                )],
            ),
            lane(
                &info,
                Modality::Video,
                vec![PromptReplacement::repeated(
                    Modality::Video,
                    "<|video_pad|>",
                    QWEN3_VIDEO_PAD_ID as TokenId,
                    4,
                )],
            ),
        ];

        let error = expand_prompt_token_ids(&mut prompt_token_ids, lanes).unwrap_err();

        assert!(matches!(
            error,
            Error::Multimodal(message)
                if message.contains("<|video_pad|>") && message.contains("`video`")
        ));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

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
        use llm_multimodal::ModelSpecificValue;
        use ndarray::ArrayD;

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
