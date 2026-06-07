//! Chat-layer multimodal image preparation.
//!
//! This module owns the narrow image-only multimodal path for chat requests:
//! it extracts image parts from structured chat messages, fetches and
//! preprocesses them through `llm-multimodal`, expands rendered prompt
//! placeholders after tokenization, and builds the engine-facing
//! `MmFeatures` payload.
//!
//! Raw media stays above `vllm-text`; this module lowers it into token IDs and
//! opaque tensor payloads before the request is handed to text generation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use itertools::izip;
use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, ImagePreProcessor, ImageProcessorRegistry, MediaConnector,
    MediaConnectorConfig, MediaContentPart, Modality, ModelMetadata, ModelProcessorSpec,
    ModelRegistry, PreProcessorConfig, PreprocessedImages, PreprocessedVideos, PromptReplacement,
    TokenResolver, TrackedMedia, VideoPreProcessor, VideoProcessorRegistry,
};
use tracing::warn;
use vllm_engine_core_client::protocol::ModelDtype;
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

/// Resolved multimodal support for one loaded model.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    spec: ResolvedMultimodalSpec,
    image_processor: ResolvedImageProcessor,
    video_processor: Option<ResolvedVideoProcessor>,
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

    /// Resolve a static image preprocessor for one loaded model.
    fn resolve_image_processor(&self) -> Option<&'static dyn ImagePreProcessor> {
        static REGISTRY: LazyLock<ImageProcessorRegistry> =
            LazyLock::new(ImageProcessorRegistry::with_defaults);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }

    /// Resolve a static video preprocessor for one loaded model.
    fn resolve_video_processor(&self) -> Option<&'static dyn VideoPreProcessor> {
        static REGISTRY: LazyLock<VideoProcessorRegistry> =
            LazyLock::new(VideoProcessorRegistry::with_defaults);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }
}

/// Static model-specific prompt and tensor-layout behavior.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    image_placeholder_token: String,
    image_placeholder_marker_token_id: u32,
    image_placeholder_embed_token_id: u32,
    video_placeholder_token: String,
    video_placeholder_marker_token_id: u32,
    video_placeholder_embed_token_id: u32,
    field_layouts: HashMap<String, FieldLayout>,
    keep_on_cpu_keys: HashSet<String>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec, context: &MultimodalModelContext) -> Result<Self> {
        let metadata = context.metadata();
        let image_placeholder_token =
            raw.placeholder_token(&metadata).map_err(|error| multimodal!("{error}"))?;
        let video_placeholder_token =
            raw.video_placeholder_token(&metadata).map_err(|error| multimodal!("{error}"))?;
        let image_placeholder_marker_token_id = context
            .tokenizer()
            .token_to_id(&image_placeholder_token)
            .ok_or_else(|| {
                multimodal!(
                    "placeholder token `{image_placeholder_token}` is not in the tokenizer vocabulary"
                )
            })?;
        let image_placeholder_embed_token_id =
            raw.placeholder_token_id(&metadata).map_err(|error| multimodal!("{error}"))? as u32;
        let video_placeholder_marker_token_id = context
            .tokenizer()
            .token_to_id(&video_placeholder_token)
            .ok_or_else(|| {
                multimodal!(
                    "placeholder token `{video_placeholder_token}` is not in the tokenizer vocabulary"
                )
            })?;
        let video_placeholder_embed_token_id =
            raw.video_placeholder_token_id(&metadata)
                .map_err(|error| multimodal!("{error}"))? as u32;

        Ok(Self {
            raw,
            image_placeholder_token,
            image_placeholder_marker_token_id,
            image_placeholder_embed_token_id,
            video_placeholder_token,
            video_placeholder_marker_token_id,
            video_placeholder_embed_token_id,
            field_layouts: raw.field_layouts(),
            keep_on_cpu_keys: raw.keep_on_cpu_keys().into_iter().collect(),
        })
    }

    fn image_prompt_replacements(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedImages,
    ) -> Result<Vec<ResolvedPromptReplacement>> {
        let replacements = self
            .raw
            .prompt_replacements(&context.metadata(), preprocessed)
            .map_err(|error| multimodal!("{error}"))?;
        Ok(replacements
            .into_iter()
            .map(|replacement| ResolvedPromptReplacement {
                replacement,
                placeholder_marker_token_id: self.image_placeholder_marker_token_id,
                placeholder_embed_token_id: self.image_placeholder_embed_token_id,
            })
            .collect())
    }

    fn video_prompt_replacements(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedVideos,
    ) -> Result<Vec<ResolvedPromptReplacement>> {
        let replacements = self
            .raw
            .video_prompt_replacements(&context.metadata(), preprocessed)
            .map_err(|error| multimodal!("{error}"))?;
        Ok(replacements
            .into_iter()
            .map(|replacement| ResolvedPromptReplacement {
                replacement,
                placeholder_marker_token_id: self.video_placeholder_marker_token_id,
                placeholder_embed_token_id: self.video_placeholder_embed_token_id,
            })
            .collect())
    }
}

/// Static video preprocessor plus its loaded config.
#[derive(Clone)]
struct ResolvedVideoProcessor {
    raw: &'static dyn VideoPreProcessor,
    config: PreProcessorConfig,
}

#[derive(Debug, Clone)]
struct ResolvedPromptReplacement {
    replacement: PromptReplacement,
    placeholder_marker_token_id: u32,
    placeholder_embed_token_id: u32,
}

#[derive(Clone)]
enum FetchedMediaItem {
    Image {
        frame: Arc<llm_multimodal::ImageFrame>,
        uuid: Option<String>,
    },
    Video {
        frame: Arc<llm_multimodal::VideoFrame>,
        uuid: Option<String>,
    },
}

impl FetchedMediaItem {
    fn modality(&self) -> Modality {
        match self {
            Self::Image { .. } => Modality::Image,
            Self::Video { .. } => Modality::Video,
        }
    }

    fn identifier(&self) -> String {
        match self {
            Self::Image { frame, uuid } => uuid.clone().unwrap_or_else(|| frame.hash.clone()),
            Self::Video { frame, uuid } => {
                uuid.clone().unwrap_or_else(|| frame.container_hash.clone())
            }
        }
    }

    fn media_hash(&self) -> String {
        match self {
            Self::Image { frame, .. } => frame.hash.clone(),
            Self::Video { frame, .. } => frame.container_hash.clone(),
        }
    }
}

#[derive(Debug)]
struct PreparedFeatureItem {
    data: MmKwargsItem,
    modality: String,
}

/// Static image preprocessor plus its loaded config.
#[derive(Clone)]
struct ResolvedImageProcessor {
    raw: &'static dyn ImagePreProcessor,
    config: PreProcessorConfig,
}

fn media_part_modality(part: &MediaContentPart) -> Option<Modality> {
    match part {
        MediaContentPart::Text { .. } => None,
        MediaContentPart::ImageUrl { .. }
        | MediaContentPart::ImageData { .. }
        | MediaContentPart::ImageEmbeds { .. } => Some(Modality::Image),
        MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
            Some(Modality::Video)
        }
    }
}

fn pop_tracked_media(
    items: &mut VecDeque<TrackedMedia>,
    uuids: &mut VecDeque<Option<String>>,
    modality: Modality,
) -> Result<FetchedMediaItem> {
    let media = items
        .pop_front()
        .ok_or_else(|| multimodal!("missing fetched {modality} item"))?;
    let uuid = uuids
        .pop_front()
        .ok_or_else(|| multimodal!("missing fetched {modality} UUID"))?;

    match (modality, media) {
        (Modality::Image, TrackedMedia::Image(frame)) => {
            Ok(FetchedMediaItem::Image { frame, uuid })
        }
        (Modality::Video, TrackedMedia::Video(frame)) => {
            Ok(FetchedMediaItem::Video { frame, uuid })
        }
        (Modality::Image, _) => Err(Error::UnsupportedMultimodalContent("non-image")),
        (Modality::Video, _) => Err(Error::UnsupportedMultimodalContent("non-video")),
        _ => Err(multimodal!(
            "unsupported fetched media modality `{modality}`"
        )),
    }
}

fn ensure_queue_drained<T>(queue: &VecDeque<T>, modality: Modality, label: &str) -> Result<()> {
    if queue.is_empty() {
        return Ok(());
    }
    bail_multimodal!("unconsumed {modality} {label} remaining after request-order rebuild")
}

fn media_parts_by_modality(
    fetched: &[FetchedMediaItem],
) -> (
    Vec<Arc<llm_multimodal::ImageFrame>>,
    Vec<Arc<llm_multimodal::VideoFrame>>,
) {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    for item in fetched {
        match item {
            FetchedMediaItem::Image { frame, .. } => images.push(Arc::clone(frame)),
            FetchedMediaItem::Video { frame, .. } => videos.push(Arc::clone(frame)),
        }
    }
    (images, videos)
}

impl MultimodalModelInfo {
    /// Load and resolve multimodal support from model files.
    ///
    /// Returns `Ok(Some(_))` only when both the model spec and image processor
    /// are registered. File read/parse failures are real errors; unsupported
    /// model families are logged and returned as `Ok(None)`.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        config_path: Option<&Path>,
        preprocessor_config_path: Option<&Path>,
        tokenizer: DynTokenizer,
    ) -> Result<Option<Self>> {
        let config = match config_path {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .map_err(|error| multimodal!("failed to read config.json: {error}"))?;
                serde_json::from_str(&text)
                    .map_err(|error| multimodal!("failed to parse config.json: {error}"))?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let preprocessor_config = match preprocessor_config_path {
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

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        let Some(spec) = context.resolve_model_spec() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "multimodal model spec is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };
        let spec = ResolvedMultimodalSpec::new(spec, &context)?;

        let Some(image_processor) = context.resolve_image_processor() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "image processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };
        let video_processor = context.resolve_video_processor().map(|raw| ResolvedVideoProcessor {
            raw,
            config: preprocessor_config.clone(),
        });

        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .map_err(|error| multimodal!("{error}"))?,
        );

        Ok(Some(Self {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: image_processor,
                config: preprocessor_config,
            },
            video_processor,
            media_connector,
        }))
    }

    /// Return the template-visible placeholder token for this model.
    ///
    /// The HF renderer uses this token while flattening image content in string
    /// content format.
    pub fn placeholder_token(&self) -> &str {
        &self.spec.image_placeholder_token
    }

    /// Return the template-visible video placeholder token for this model.
    ///
    /// The HF renderer uses this token while flattening video content in string
    /// content format.
    pub fn video_placeholder_token(&self) -> &str {
        &self.spec.video_placeholder_token
    }

    /// Run media fetch, preprocessing, prompt expansion, and feature build.
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
        let fetched = self.fetch_media(media_parts).await?;
        let (image_frames, video_frames) = media_parts_by_modality(&fetched);

        let image_preprocessed = if image_frames.is_empty() {
            None
        } else {
            Some(self.preprocess_images(&image_frames).await?)
        };
        let video_preprocessed = if video_frames.is_empty() {
            None
        } else {
            Some(self.preprocess_videos(&video_frames).await?)
        };

        let mut image_replacements: VecDeque<_> =
            if let Some(preprocessed) = image_preprocessed.as_ref() {
                self.spec.image_prompt_replacements(&self.context, preprocessed)?
            } else {
                Vec::new()
            }
            .into();
        let mut video_replacements: VecDeque<_> =
            if let Some(preprocessed) = video_preprocessed.as_ref() {
                self.spec.video_prompt_replacements(&self.context, preprocessed)?
            } else {
                Vec::new()
            }
            .into();
        let mut ordered_replacements = Vec::with_capacity(fetched.len());
        for item in &fetched {
            let replacement = match item.modality() {
                Modality::Image => image_replacements
                    .pop_front()
                    .ok_or_else(|| multimodal!("missing image prompt replacement"))?,
                Modality::Video => video_replacements
                    .pop_front()
                    .ok_or_else(|| multimodal!("missing video prompt replacement"))?,
                other => bail_multimodal!("unsupported prompt replacement modality `{other}`"),
            };
            ordered_replacements.push(replacement);
        }
        ensure_queue_drained(&image_replacements, Modality::Image, "prompt replacements")?;
        ensure_queue_drained(&video_replacements, Modality::Video, "prompt replacements")?;

        let ranges = self.expand_prompt_tokens(prompt_token_ids, ordered_replacements)?;
        let image_features = if let Some(preprocessed) = image_preprocessed {
            self.build_image_feature_items(preprocessed, image_frames.len(), model_dtype)?
        } else {
            VecDeque::new()
        };
        let video_features = if let Some(preprocessed) = video_preprocessed {
            self.build_video_feature_items(preprocessed, video_frames.len(), model_dtype)?
        } else {
            VecDeque::new()
        };
        let features = self.build_features(fetched, image_features, video_features, ranges)?;
        if features.len() != media_parts_len {
            bail_multimodal!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts_len
            );
        }
        Ok(features)
    }

    /// Fetch all media parts and rebuild them in original request order.
    async fn fetch_media(
        &self,
        media_parts: Vec<MediaContentPart>,
    ) -> Result<Vec<FetchedMediaItem>> {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        for part in &media_parts {
            tracker.push_part(part.clone()).map_err(|error| multimodal!("{error}"))?;
        }

        let tracker_output = tracker.finalize().await.map_err(|error| multimodal!("{error}"))?;
        let mut images: VecDeque<_> =
            tracker_output.data.get(&Modality::Image).cloned().unwrap_or_default().into();
        let mut image_uuids: VecDeque<_> =
            tracker_output.uuids.get(&Modality::Image).cloned().unwrap_or_default().into();
        let mut videos: VecDeque<_> =
            tracker_output.data.get(&Modality::Video).cloned().unwrap_or_default().into();
        let mut video_uuids: VecDeque<_> =
            tracker_output.uuids.get(&Modality::Video).cloned().unwrap_or_default().into();

        let mut fetched = Vec::with_capacity(media_parts.len());
        for part in media_parts {
            let Some(modality) = media_part_modality(&part) else {
                continue;
            };
            let item = match modality {
                Modality::Image => {
                    pop_tracked_media(&mut images, &mut image_uuids, Modality::Image)?
                }
                Modality::Video => {
                    pop_tracked_media(&mut videos, &mut video_uuids, Modality::Video)?
                }
                other => bail_multimodal!("unsupported fetched media modality `{other}`"),
            };
            fetched.push(item);
        }

        ensure_queue_drained(&images, Modality::Image, "media items")?;
        ensure_queue_drained(&videos, Modality::Video, "media items")?;
        ensure_queue_drained(&image_uuids, Modality::Image, "UUIDs")?;
        ensure_queue_drained(&video_uuids, Modality::Video, "UUIDs")?;
        Ok(fetched)
    }

    /// Preprocess fetched image frames with the model's resolved image
    /// processor.
    async fn preprocess_images(
        &self,
        image_frames: &[Arc<llm_multimodal::ImageFrame>],
    ) -> Result<PreprocessedImages> {
        let config = self.image_processor.config.clone();
        let processor = self.image_processor.raw;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        tokio::task::spawn_blocking(move || {
            processor.preprocess(&images, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }

    /// Preprocess fetched video frames with the model's resolved video
    /// processor.
    async fn preprocess_videos(
        &self,
        video_frames: &[Arc<llm_multimodal::VideoFrame>],
    ) -> Result<PreprocessedVideos> {
        let video_processor = self
            .video_processor
            .as_ref()
            .ok_or(Error::UnsupportedMultimodalContent("video_url"))?;
        let config = video_processor.config.clone();
        let processor = video_processor.raw;
        let videos = video_frames.iter().map(|frame| frame.frames().to_vec()).collect::<Vec<_>>();

        tokio::task::spawn_blocking(move || {
            processor.preprocess(&videos, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("video preprocessing task failed: {error}"))?
    }

    /// Replace rendered placeholder markers with model-specific replacement
    /// tokens.
    fn expand_prompt_tokens(
        &self,
        prompt_token_ids: &mut Vec<u32>,
        replacements: Vec<ResolvedPromptReplacement>,
    ) -> Result<Vec<PlaceholderRange>> {
        expand_prompt_token_ids(prompt_token_ids, replacements)
    }

    fn build_image_feature_items(
        &self,
        preprocessed: PreprocessedImages,
        item_count: usize,
        model_dtype: ModelDtype,
    ) -> Result<VecDeque<PreparedFeatureItem>> {
        let tensors = tensor::collect_image_tensors(preprocessed, model_dtype)?;
        self.build_feature_items_from_tensors("image", tensors, item_count)
    }

    fn build_video_feature_items(
        &self,
        preprocessed: PreprocessedVideos,
        item_count: usize,
        model_dtype: ModelDtype,
    ) -> Result<VecDeque<PreparedFeatureItem>> {
        let tensors = tensor::collect_video_tensors(preprocessed, model_dtype)?;
        self.build_feature_items_from_tensors("video", tensors, item_count)
    }

    fn build_feature_items_from_tensors(
        &self,
        modality: &str,
        tensors: HashMap<String, tensor::KwargValue>,
        item_count: usize,
    ) -> Result<VecDeque<PreparedFeatureItem>> {
        let mut items = VecDeque::with_capacity(item_count);
        for index in 0..item_count {
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
                            batch_size: item_count,
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

            items.push_back(PreparedFeatureItem {
                data,
                modality: modality.to_string(),
            });
        }

        Ok(items)
    }

    fn build_features(
        &self,
        fetched: Vec<FetchedMediaItem>,
        image_features: VecDeque<PreparedFeatureItem>,
        video_features: VecDeque<PreparedFeatureItem>,
        ranges: Vec<PlaceholderRange>,
    ) -> Result<MmFeatures> {
        if ranges.len() != fetched.len() {
            bail_multimodal!(
                "number of placeholder ranges {} does not match number of fetched media items {}",
                ranges.len(),
                fetched.len()
            );
        }

        let mut image_features = image_features;
        let mut video_features = video_features;
        let mut features = Vec::with_capacity(fetched.len());
        for (item, range) in izip!(fetched, ranges) {
            let feature = match item.modality() {
                Modality::Image => image_features
                    .pop_front()
                    .ok_or_else(|| multimodal!("missing prepared image feature"))?,
                Modality::Video => video_features
                    .pop_front()
                    .ok_or_else(|| multimodal!("missing prepared video feature"))?,
                other => bail_multimodal!("unsupported feature modality `{other}`"),
            };
            features.push(MmFeatureSpec {
                data: Some(feature.data),
                modality: feature.modality,
                identifier: item.identifier(),
                mm_position: range,
                mm_hash: Some(item.media_hash()),
            });
        }

        ensure_queue_drained(&image_features, Modality::Image, "feature items")?;
        ensure_queue_drained(&video_features, Modality::Video, "feature items")?;
        Ok(features)
    }
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
    let Prompt::Text(prompt) = rendered.prompt else {
        bail_multimodal!("multimodal chat renderer must return a text prompt before expansion");
    };
    let media_parts = extract_media_parts(request)?;

    let mut prompt_token_ids = info
        .context
        .tokenizer()
        .encode(&prompt, request.add_special_tokens)
        .map_err(|error| multimodal!("{error}"))?;
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

fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    replacements: Vec<ResolvedPromptReplacement>,
) -> Result<Vec<PlaceholderRange>> {
    if replacements.is_empty() {
        return Ok(Vec::new());
    }

    let replacement_growth = replacements.iter().fold(0usize, |total, replacement| {
        total.saturating_add(replacement.replacement.tokens.len().saturating_sub(1))
    });
    let mut expanded =
        Vec::with_capacity(prompt_token_ids.len().saturating_add(replacement_growth));
    let mut ranges = Vec::with_capacity(replacements.len());
    let mut cursor = 0usize;

    for replacement in replacements {
        let offset = find_next_token(
            prompt_token_ids,
            replacement.placeholder_marker_token_id,
            cursor,
        )
        .ok_or_else(|| {
            multimodal!(
                "placeholder token `{}` was not found in tokenized prompt",
                replacement.replacement.placeholder_token
            )
        })?;

        if replacement.replacement.tokens.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` expanded to no tokens",
                replacement.replacement.placeholder_token
            );
        }

        let replacement_len = replacement.replacement.tokens.len();
        let is_embed = {
            let mask = replacement
                .replacement
                .tokens
                .iter()
                .map(|&token| token as u32 == replacement.placeholder_embed_token_id)
                .collect::<Vec<_>>();
            WireTensor::from_bool(vec![replacement_len], mask).map_err(Error::Multimodal)?
        };

        expanded.extend_from_slice(&prompt_token_ids[cursor..offset]);
        let expanded_offset = expanded.len();
        expanded.extend(replacement.replacement.tokens.into_iter().map(|token| token as u32));
        ranges.push(PlaceholderRange {
            offset: expanded_offset,
            length: replacement_len,
            is_embed: Some(is_embed),
        });
        cursor = offset + 1;
    }

    expanded.extend_from_slice(&prompt_token_ids[cursor..]);
    *prompt_token_ids = expanded;
    Ok(ranges)
}

/// Find `needle` in `haystack`, starting at `start`.
///
/// This is intentionally order-preserving rather than a global replace: each
/// media item consumes the next placeholder occurrence.
fn find_next_token(haystack: &[u32], needle: u32, start: usize) -> Option<usize> {
    haystack
        .get(start..)?
        .iter()
        .position(|token| *token == needle)
        .map(|offset| start + offset)
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::TokenId;
    use vllm_engine_core_client::protocol::tensor::WireArrayData;
    use vllm_text::tokenizer::{IncrementalDecoder, Tokenizer, TokenizerError};

    use super::*;

    const LLAMA4_IMAGE_START_ID: u32 = 200088;
    const LLAMA4_IMAGE_END_ID: u32 = 200089;
    const LLAMA4_IMAGE_ID: u32 = 200090;
    const LLAMA4_PATCH_ID: u32 = 200092;
    const LLAMA4_TILE_X_SEPARATOR_ID: u32 = 200093;
    const LLAMA4_TILE_Y_SEPARATOR_ID: u32 = 200094;
    const TEST_VIDEO_MARKER_ID: u32 = 200095;
    const TEST_VIDEO_EMBED_ID: u32 = 200096;

    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> std::result::Result<Vec<u32>, TokenizerError> {
            Ok(match text {
                "<|image|>" => vec![LLAMA4_IMAGE_ID],
                text => text.bytes().map(u32::from).collect(),
            })
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> std::result::Result<String, TokenizerError> {
            Ok(String::new())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|image_start|>" => Some(LLAMA4_IMAGE_START_ID),
                "<|image_end|>" => Some(LLAMA4_IMAGE_END_ID),
                "<|image|>" => Some(LLAMA4_IMAGE_ID),
                "<|patch|>" => Some(LLAMA4_PATCH_ID),
                "<|tile_x_separator|>" => Some(LLAMA4_TILE_X_SEPARATOR_ID),
                "<|tile_y_separator|>" => Some(LLAMA4_TILE_Y_SEPARATOR_ID),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                LLAMA4_IMAGE_START_ID => Some("<|image_start|>".to_string()),
                LLAMA4_IMAGE_END_ID => Some("<|image_end|>".to_string()),
                LLAMA4_IMAGE_ID => Some("<|image|>".to_string()),
                LLAMA4_PATCH_ID => Some("<|patch|>".to_string()),
                LLAMA4_TILE_X_SEPARATOR_ID => Some("<|tile_x_separator|>".to_string()),
                LLAMA4_TILE_Y_SEPARATOR_ID => Some("<|tile_y_separator|>".to_string()),
                _ => None,
            }
        }

        fn create_decode_stream(
            &self,
            _prompt_token_ids: &[u32],
            _skip_special_tokens: bool,
            _min_bytes_to_buffer: usize,
        ) -> Box<dyn IncrementalDecoder + '_> {
            unreachable!("not used")
        }
    }

    fn test_info(model_type: &str, config: serde_json::Value) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(TestTokenizer)),
        };
        let spec = context
            .resolve_model_spec()
            .unwrap_or_else(|| panic!("{model_type} spec should match"));
        let spec = ResolvedMultimodalSpec::new(spec, &context).unwrap();
        let raw_image_processor = context
            .resolve_image_processor()
            .unwrap_or_else(|| panic!("{model_type} image processor should match"));
        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default()).unwrap(),
        );

        MultimodalModelInfo {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: raw_image_processor,
                config: PreProcessorConfig::default(),
            },
            video_processor: None,
            media_connector,
        }
    }

    fn llama4_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "llama4",
            "image_token_index": LLAMA4_PATCH_ID,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        test_info("llama4", config)
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

    fn resolve_image_replacements(
        replacements: Vec<PromptReplacement>,
    ) -> Vec<ResolvedPromptReplacement> {
        replacements
            .into_iter()
            .map(|replacement| ResolvedPromptReplacement {
                replacement,
                placeholder_marker_token_id: LLAMA4_IMAGE_ID,
                placeholder_embed_token_id: LLAMA4_PATCH_ID,
            })
            .collect()
    }

    #[test]
    fn expand_prompt_tokens_marks_only_llama4_patch_tokens_as_embed() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let replacements = resolve_image_replacements(vec![llama4_multi_tile_replacement()]);

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap();

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
        let replacements = resolve_image_replacements(vec![llama4_single_tile_replacement()]);

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_ignores_empty_replacements() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, Vec::new()).unwrap();

        assert!(ranges.is_empty());
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_leaves_prompt_unchanged_when_later_placeholder_missing() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let replacements = vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ];
        let replacements = resolve_image_replacements(replacements);

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let replacements = resolve_image_replacements(vec![PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            Vec::new(),
        )]);

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let replacements = vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ];
        let replacements = resolve_image_replacements(replacements);

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap();

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
    fn expand_prompt_token_ids_supports_mixed_image_and_video_placeholders() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, TEST_VIDEO_MARKER_ID, 3];
        let replacements = vec![
            ResolvedPromptReplacement {
                replacement: PromptReplacement::sequence(
                    Modality::Image,
                    "<|image|>",
                    vec![LLAMA4_IMAGE_START_ID as TokenId, LLAMA4_PATCH_ID as TokenId],
                ),
                placeholder_marker_token_id: LLAMA4_IMAGE_ID,
                placeholder_embed_token_id: LLAMA4_PATCH_ID,
            },
            ResolvedPromptReplacement {
                replacement: PromptReplacement::sequence(
                    Modality::Video,
                    "<|video_pad|>",
                    vec![
                        TEST_VIDEO_MARKER_ID as TokenId,
                        TEST_VIDEO_EMBED_ID as TokenId,
                        TEST_VIDEO_MARKER_ID as TokenId,
                    ],
                ),
                placeholder_marker_token_id: TEST_VIDEO_MARKER_ID,
                placeholder_embed_token_id: TEST_VIDEO_EMBED_ID,
            },
        ];

        let ranges = expand_prompt_token_ids(&mut prompt_token_ids, replacements).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_PATCH_ID,
                2,
                TEST_VIDEO_MARKER_ID,
                TEST_VIDEO_EMBED_ID,
                TEST_VIDEO_MARKER_ID,
                3,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 2);
        assert_bool_mask(&ranges[0], &[false, true]);
        assert_eq!(ranges[1].offset, 4);
        assert_eq!(ranges[1].length, 3);
        assert_bool_mask(&ranges[1], &[false, true, false]);
    }
}
