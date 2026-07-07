//! Chat-layer multimodal media preparation.
//!
//! This module owns the narrow image/video multimodal path for chat requests:
//! it extracts media parts from structured chat messages, fetches and
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

use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, MediaConnector, MediaConnectorConfig, MediaContentPart,
    Modality, ModelMetadata, ModelProcessorSpec, ModelRegistry, PreProcessorConfig,
    PreprocessedEncoderInputs as PreprocessedMedia, PromptReplacement, Tokenizer as TokenResolver,
    TrackedMedia, VisionPreProcessor, VisionProcessorRegistry,
};
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

/// Resolved multimodal support for one loaded model.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    spec: ResolvedMultimodalSpec,
    vision_processor: ResolvedVisionProcessor,
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
    fn resolve_vision_processor(&self) -> Option<&'static dyn VisionPreProcessor> {
        static REGISTRY: LazyLock<VisionProcessorRegistry> =
            LazyLock::new(VisionProcessorRegistry::with_defaults);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }
}

/// Static model-specific prompt and tensor-layout behavior.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    placeholders: HashMap<Modality, ResolvedPlaceholder>,
    field_layouts: HashMap<String, FieldLayout>,
    keep_on_cpu_keys: HashSet<String>,
}

#[derive(Clone)]
struct ResolvedPlaceholder {
    token: String,
    marker_token_id: u32,
    embed_token_id: u32,
    wrapper_start_token_id: Option<u32>,
    wrapper_end_token_id: Option<u32>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec, context: &MultimodalModelContext) -> Result<Self> {
        let mut placeholders = HashMap::new();
        placeholders.insert(
            Modality::Image,
            Self::resolve_placeholder(raw, context, Modality::Image)?,
        );
        if let Ok(video_placeholder) = Self::resolve_placeholder(raw, context, Modality::Video) {
            placeholders.insert(Modality::Video, video_placeholder);
        }

        Ok(Self {
            raw,
            placeholders,
            field_layouts: raw.field_layouts(),
            keep_on_cpu_keys: raw.keep_on_cpu_keys().into_iter().collect(),
        })
    }

    fn resolve_placeholder(
        raw: &'static dyn ModelProcessorSpec,
        context: &MultimodalModelContext,
        modality: Modality,
    ) -> Result<ResolvedPlaceholder> {
        let metadata = context.metadata();
        let token = raw
            .placeholder_token_for(&metadata, modality)
            .map_err(|error| multimodal!("{error}"))?;
        // This is the rendered prompt marker, so resolve it from the token
        // string itself. Do not use `placeholder_token_id_for()` for this:
        // for some specs that ID is the replacement vision/patch token.
        let marker_token_id = context.tokenizer().token_to_id(&token).ok_or_else(|| {
            multimodal!("placeholder token `{token}` is not in the tokenizer vocabulary")
        })?;
        let embed_token_id = raw
            .placeholder_token_id_for(&metadata, modality)
            .map_err(|error| multimodal!("{error}"))? as u32;

        let wrapper_start_token_id = (modality == Modality::Video)
            .then(|| {
                metadata
                    .config_u32(&["vision_start_token_id"])
                    .or_else(|| context.tokenizer().token_to_id("<|vision_start|>"))
            })
            .flatten();
        let wrapper_end_token_id = (modality == Modality::Video)
            .then(|| {
                metadata
                    .config_u32(&["vision_end_token_id"])
                    .or_else(|| context.tokenizer().token_to_id("<|vision_end|>"))
            })
            .flatten();

        Ok(ResolvedPlaceholder {
            token,
            marker_token_id,
            embed_token_id,
            wrapper_start_token_id,
            wrapper_end_token_id,
        })
    }

    fn placeholder_token(&self, modality: Modality) -> Option<&str> {
        self.placeholders.get(&modality).map(|placeholder| placeholder.token.as_str())
    }

    fn prompt_replacements(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedMedia,
        modality: Modality,
    ) -> Result<Vec<PromptReplacement>> {
        self.raw
            .prompt_replacements_for(&context.metadata(), preprocessed, modality)
            .map_err(|error| multimodal!("{error}"))
    }
}

/// Static vision preprocessor plus its loaded config.
#[derive(Clone)]
struct ResolvedVisionProcessor {
    raw: &'static dyn VisionPreProcessor,
    config: PreProcessorConfig,
}

/// Request-scoped fetched media, kept together with tracker UUID metadata.
struct FetchedMedia {
    items: Vec<FetchedMediaItem>,
}

#[derive(Clone)]
enum FetchedMediaItem {
    Image {
        frame: Arc<llm_multimodal::ImageFrame>,
        uuid: Option<String>,
    },
    Video {
        clip: Arc<llm_multimodal::VideoClip>,
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

    fn hash(&self) -> &str {
        match self {
            Self::Image { frame, .. } => &frame.hash,
            Self::Video { clip, .. } => &clip.hash,
        }
    }

    fn uuid(&self) -> Option<&String> {
        match self {
            Self::Image { uuid, .. } | Self::Video { uuid, .. } => uuid.as_ref(),
        }
    }
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

        let Some(vision_processor) = context.resolve_vision_processor() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "image processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .map_err(|error| multimodal!("{error}"))?,
        );

        Ok(Some(Self {
            context,
            spec,
            vision_processor: ResolvedVisionProcessor {
                raw: vision_processor,
                config: preprocessor_config,
            },
            media_connector,
        }))
    }

    /// Return the template-visible image placeholder token for this model.
    ///
    /// The HF renderer uses this token while flattening image content in string
    /// content format.
    pub fn placeholder_token(&self) -> &str {
        self.placeholder_token_for(Modality::Image)
            .expect("image placeholder is resolved during multimodal setup")
    }

    /// Return the template-visible placeholder token for one modality.
    pub fn placeholder_token_for(&self, modality: Modality) -> Option<&str> {
        self.spec.placeholder_token(modality)
    }
}

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their image placeholders are expanded, and
/// preprocessed image features are attached for engine-core transport.
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
                    });
                }
            }
        }
    }
    Ok(all_parts)
}

impl MultimodalModelInfo {
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
        let image_frames = fetched
            .items
            .iter()
            .filter_map(|item| match item {
                FetchedMediaItem::Image { frame, .. } => Some(Arc::clone(frame)),
                FetchedMediaItem::Video { .. } => None,
            })
            .collect::<Vec<_>>();

        let image_batch = if image_frames.is_empty() {
            None
        } else {
            let preprocessed = self.preprocess_images(&image_frames).await?;
            let replacements =
                self.spec.prompt_replacements(&self.context, &preprocessed, Modality::Image)?;
            if replacements.len() != image_frames.len() {
                bail_multimodal!(
                    "number of image prompt replacements {} does not match number of images {}",
                    replacements.len(),
                    image_frames.len()
                );
            }
            Some(PreparedMediaBatch {
                len: image_frames.len(),
                replacements,
                tensors: tensor::collect_tensors(preprocessed, model_dtype, Modality::Image)?,
            })
        };

        let mut video_batches = Vec::new();
        for item in &fetched.items {
            let FetchedMediaItem::Video { clip, .. } = item else {
                continue;
            };
            let preprocessed = self.preprocess_video(Arc::clone(clip)).await?;
            let replacements =
                self.spec.prompt_replacements(&self.context, &preprocessed, Modality::Video)?;
            if replacements.len() != 1 {
                bail_multimodal!(
                    "number of video prompt replacements {} does not match one video",
                    replacements.len()
                );
            }
            video_batches.push(PreparedMediaBatch {
                len: 1,
                replacements,
                tensors: tensor::collect_tensors(preprocessed, model_dtype, Modality::Video)?,
            });
        }

        let mut replacements = Vec::with_capacity(fetched.items.len());
        let mut image_replacement_index = 0usize;
        let mut video_batch_index = 0usize;
        for item in &fetched.items {
            match item {
                FetchedMediaItem::Image { .. } => {
                    let image_batch = image_batch.as_ref().ok_or_else(|| {
                        multimodal!("image media item has no preprocessed image batch")
                    })?;
                    let replacement =
                        image_batch.replacements.get(image_replacement_index).ok_or_else(|| {
                            multimodal!(
                                "image prompt replacement {image_replacement_index} is missing"
                            )
                        })?;
                    replacements.push(replacement.clone());
                    image_replacement_index += 1;
                }
                FetchedMediaItem::Video { .. } => {
                    let video_batch = video_batches.get(video_batch_index).ok_or_else(|| {
                        multimodal!("video media item has no preprocessed video batch")
                    })?;
                    replacements.push(video_batch.replacements[0].clone());
                    video_batch_index += 1;
                }
            }
        }

        let ranges = self.expand_prompt_tokens(prompt_token_ids, replacements)?;
        let features = self.build_features(fetched, image_batch, video_batches, ranges)?;
        if features.len() != media_parts_len {
            bail_multimodal!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts_len
            );
        }
        Ok(features)
    }

    /// Fetch all media parts and preserve request-order UUID metadata.
    async fn fetch_media(&self, media_parts: Vec<MediaContentPart>) -> Result<FetchedMedia> {
        let mut modalities = Vec::with_capacity(media_parts.len());
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        for part in media_parts {
            modalities.push(media_part_modality(&part)?);
            tracker.push_part(part).map_err(|error| multimodal!("{error}"))?;
        }

        let tracker_output = tracker.finalize().await.map_err(|error| multimodal!("{error}"))?;
        let mut images = tracker_output
            .data
            .get(&Modality::Image)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<VecDeque<_>>();
        let mut image_uuids = tracker_output
            .uuids
            .get(&Modality::Image)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<VecDeque<_>>();
        let mut videos = tracker_output
            .data
            .get(&Modality::Video)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<VecDeque<_>>();
        let mut video_uuids = tracker_output
            .uuids
            .get(&Modality::Video)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<VecDeque<_>>();

        let mut items = Vec::with_capacity(modalities.len());
        for modality in modalities {
            match modality {
                Modality::Image => {
                    let media = images.pop_front().ok_or_else(|| {
                        multimodal!("tracker returned fewer images than requested")
                    })?;
                    let uuid = image_uuids.pop_front().flatten();
                    match media {
                        TrackedMedia::Image(frame) => {
                            items.push(FetchedMediaItem::Image { frame, uuid })
                        }
                        _ => bail_multimodal!("tracker returned non-image media for image part"),
                    }
                }
                Modality::Video => {
                    let media = videos.pop_front().ok_or_else(|| {
                        multimodal!("tracker returned fewer videos than requested")
                    })?;
                    let uuid = video_uuids.pop_front().flatten();
                    match media {
                        TrackedMedia::Video(clip) => {
                            items.push(FetchedMediaItem::Video { clip, uuid })
                        }
                        _ => bail_multimodal!("tracker returned non-video media for video part"),
                    }
                }
                _ => bail_multimodal!("unsupported fetched media modality `{modality}`"),
            }
        }

        Ok(FetchedMedia { items })
    }

    /// Preprocess fetched image frames with the model's resolved vision processor.
    async fn preprocess_images(
        &self,
        image_frames: &[Arc<llm_multimodal::ImageFrame>],
    ) -> Result<PreprocessedMedia> {
        let config = self.vision_processor.config.clone();
        let processor = self.vision_processor.raw;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        tokio::task::spawn_blocking(move || {
            processor.preprocess(&images, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }

    /// Preprocess one fetched video clip with the model's resolved vision processor.
    async fn preprocess_video(
        &self,
        clip: Arc<llm_multimodal::VideoClip>,
    ) -> Result<PreprocessedMedia> {
        let config = self.vision_processor.config.clone();
        let processor = self.vision_processor.raw;

        tokio::task::spawn_blocking(move || {
            if let Some(rgb_video) = clip.rgb_video() {
                let rgb_frames = rgb_video.frame_refs().map_err(|error| multimodal!("{error}"))?;
                if let Ok(preprocessed) = processor.preprocess_video_rgb(&rgb_frames, &config) {
                    return Ok(preprocessed);
                }
            }

            let frames = clip.materialized_frames().map_err(|error| multimodal!("{error}"))?;
            processor
                .preprocess_video(&frames, &config)
                .map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("video preprocessing task failed: {error}"))?
    }

    /// Replace rendered placeholder markers with model-specific replacement tokens.
    fn expand_prompt_tokens(
        &self,
        prompt_token_ids: &mut Vec<u32>,
        replacements: Vec<PromptReplacement>,
    ) -> Result<Vec<PlaceholderRange>> {
        expand_prompt_token_ids(prompt_token_ids, replacements, &self.spec.placeholders)
    }

    /// Convert preprocessed tensors into engine-core multimodal features.
    fn build_features(
        &self,
        fetched: FetchedMedia,
        image_batch: Option<PreparedMediaBatch>,
        video_batches: Vec<PreparedMediaBatch>,
        ranges: Vec<PlaceholderRange>,
    ) -> Result<MmFeatures> {
        if ranges.len() != fetched.items.len() {
            bail_multimodal!(
                "number of placeholder ranges {} does not match number of media items {}",
                ranges.len(),
                fetched.items.len()
            );
        }

        let mut features = Vec::with_capacity(fetched.items.len());
        let mut image_index = 0usize;
        let mut video_index = 0usize;
        for (item, range) in fetched.items.into_iter().zip(ranges) {
            match item.modality() {
                Modality::Image => {
                    let image_batch = image_batch.as_ref().ok_or_else(|| {
                        multimodal!("image media item has no preprocessed image batch")
                    })?;
                    features.push(self.build_feature(
                        &image_batch.tensors,
                        image_index,
                        image_batch.len,
                        item,
                        range,
                    )?);
                    image_index += 1;
                }
                Modality::Video => {
                    let video_batch = video_batches.get(video_index).ok_or_else(|| {
                        multimodal!("video media item has no preprocessed video batch")
                    })?;
                    features.push(self.build_feature(
                        &video_batch.tensors,
                        0,
                        video_batch.len,
                        item,
                        range,
                    )?);
                    video_index += 1;
                }
                _ => bail_multimodal!("unsupported media modality `{}`", item.modality()),
            }
        }

        Ok(features)
    }

    fn build_feature(
        &self,
        tensors: &HashMap<String, tensor::KwargValue>,
        index: usize,
        len: usize,
        item: FetchedMediaItem,
        range: PlaceholderRange,
    ) -> Result<MmFeatureSpec> {
        let mut data = MmKwargsItem::new();
        for (key, tensor) in tensors {
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

        let hash = item.hash().to_string();
        let identifier = item.uuid().cloned().unwrap_or_else(|| hash.clone());
        Ok(MmFeatureSpec {
            data: Some(data),
            modality: item.modality().to_string(),
            identifier,
            mm_position: range,
            mm_hash: Some(hash),
        })
    }
}

struct PreparedMediaBatch {
    len: usize,
    replacements: Vec<PromptReplacement>,
    tensors: HashMap<String, tensor::KwargValue>,
}

fn media_part_modality(part: &MediaContentPart) -> Result<Modality> {
    match part {
        MediaContentPart::ImageUrl { .. } | MediaContentPart::ImageData { .. } => {
            Ok(Modality::Image)
        }
        MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
            Ok(Modality::Video)
        }
        MediaContentPart::Text { .. } => Err(Error::UnsupportedMultimodalContent("text")),
        MediaContentPart::ImageEmbeds { .. } => {
            Err(Error::UnsupportedMultimodalContent("image_embeds"))
        }
    }
}

fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    replacements: Vec<PromptReplacement>,
    placeholders: &HashMap<Modality, ResolvedPlaceholder>,
) -> Result<Vec<PlaceholderRange>> {
    if replacements.is_empty() {
        return Ok(Vec::new());
    }

    let replacement_growth = replacements.iter().fold(0usize, |total, replacement| {
        total.saturating_add(replacement.tokens.len().saturating_sub(1))
    });
    let mut expanded =
        Vec::with_capacity(prompt_token_ids.len().saturating_add(replacement_growth));
    let mut ranges = Vec::with_capacity(replacements.len());
    let mut cursor = 0usize;

    for replacement in replacements {
        let placeholder = placeholders.get(&replacement.modality).ok_or_else(|| {
            multimodal!(
                "unsupported prompt replacement modality `{}`",
                replacement.modality
            )
        })?;
        let offset = find_next_token(prompt_token_ids, placeholder.marker_token_id, cursor)
            .ok_or_else(|| {
                multimodal!(
                    "placeholder token `{}` was not found in tokenized prompt",
                    replacement.placeholder_token
                )
            })?;

        if replacement.tokens.is_empty() {
            bail_multimodal!(
                "placeholder token `{}` expanded to no tokens",
                replacement.placeholder_token
            );
        }

        let replacement_len = replacement.tokens.len();
        let mut mask = replacement
            .tokens
            .iter()
            .map(|&token| token as u32 == placeholder.embed_token_id)
            .collect::<Vec<_>>();

        expanded.extend_from_slice(&prompt_token_ids[cursor..offset]);
        let mut expanded_offset = expanded.len();
        let mut range_len = replacement_len;
        if should_include_video_wrapper(prompt_token_ids, offset, replacement.modality, placeholder)
        {
            expanded_offset = expanded_offset.saturating_sub(1);
            range_len += 2;
            mask.insert(0, false);
            mask.push(false);
        }
        let is_embed = WireTensor::from_bool(vec![range_len], mask).map_err(Error::Multimodal)?;

        expanded.extend(replacement.tokens.into_iter().map(|token| token as u32));
        ranges.push(PlaceholderRange {
            offset: expanded_offset,
            length: range_len,
            is_embed: Some(is_embed),
        });
        cursor = offset + 1;
    }

    expanded.extend_from_slice(&prompt_token_ids[cursor..]);
    *prompt_token_ids = expanded;

    Ok(ranges)
}

fn should_include_video_wrapper(
    prompt_token_ids: &[u32],
    marker_offset: usize,
    modality: Modality,
    placeholder: &ResolvedPlaceholder,
) -> bool {
    if modality != Modality::Video {
        return false;
    }
    let Some(start_token_id) = placeholder.wrapper_start_token_id else {
        return false;
    };
    let Some(end_token_id) = placeholder.wrapper_end_token_id else {
        return false;
    };
    marker_offset > 0
        && prompt_token_ids.get(marker_offset - 1) == Some(&start_token_id)
        && prompt_token_ids.get(marker_offset + 1) == Some(&end_token_id)
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

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        self.0.encode(text, false).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use llm_multimodal::{TokenId, VideoClip, VideoSource};
    use vllm_engine_core_client::protocol::tensor::WireArrayData;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    const LLAMA4_IMAGE_START_ID: u32 = 200088;
    const LLAMA4_IMAGE_END_ID: u32 = 200089;
    const LLAMA4_IMAGE_ID: u32 = 200090;
    const LLAMA4_PATCH_ID: u32 = 200092;
    const LLAMA4_TILE_X_SEPARATOR_ID: u32 = 200093;
    const LLAMA4_TILE_Y_SEPARATOR_ID: u32 = 200094;

    fn llama4_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_start|>", LLAMA4_IMAGE_START_ID)
            .with_regular_token("<|image_end|>", LLAMA4_IMAGE_END_ID)
            .with_regular_token("<|image|>", LLAMA4_IMAGE_ID)
            .with_regular_token("<|patch|>", LLAMA4_PATCH_ID)
            .with_regular_token("<|tile_x_separator|>", LLAMA4_TILE_X_SEPARATOR_ID)
            .with_regular_token("<|tile_y_separator|>", LLAMA4_TILE_Y_SEPARATOR_ID)
    }

    fn test_info(model_type: &str, config: serde_json::Value) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(llama4_tokenizer())),
        };
        let spec = context
            .resolve_model_spec()
            .unwrap_or_else(|| panic!("{model_type} spec should match"));
        let spec = ResolvedMultimodalSpec::new(spec, &context).unwrap();
        let raw_vision_processor = context
            .resolve_vision_processor()
            .unwrap_or_else(|| panic!("{model_type} image processor should match"));
        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default()).unwrap(),
        );

        MultimodalModelInfo {
            context,
            spec,
            vision_processor: ResolvedVisionProcessor {
                raw: raw_vision_processor,
                config: PreProcessorConfig::default(),
            },
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

    #[test]
    fn expand_prompt_tokens_marks_only_llama4_patch_tokens_as_embed() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let replacements = vec![llama4_multi_tile_replacement()];

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
        let replacements = vec![llama4_single_tile_replacement()];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_token_ids_handles_video_only_replacements() {
        let mut prompt_token_ids = vec![1, 300, 2];
        let mut placeholders = HashMap::new();
        placeholders.insert(
            Modality::Video,
            ResolvedPlaceholder {
                token: "<|video|>".to_string(),
                marker_token_id: 300,
                embed_token_id: 301,
                wrapper_start_token_id: None,
                wrapper_end_token_id: None,
            },
        );
        let replacements = vec![PromptReplacement::sequence(
            Modality::Video,
            "<|video|>",
            vec![301, 301, 301],
        )];

        let ranges =
            expand_prompt_token_ids(&mut prompt_token_ids, replacements, &placeholders).unwrap();

        assert_eq!(prompt_token_ids, vec![1, 301, 301, 301, 2]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
        assert_bool_mask(&ranges[0], &[true, true, true]);
    }

    #[test]
    fn expand_prompt_token_ids_includes_video_vision_wrapper_when_present() {
        let mut prompt_token_ids = vec![1, 302, 300, 303, 2];
        let mut placeholders = HashMap::new();
        placeholders.insert(
            Modality::Video,
            ResolvedPlaceholder {
                token: "<|video|>".to_string(),
                marker_token_id: 300,
                embed_token_id: 301,
                wrapper_start_token_id: Some(302),
                wrapper_end_token_id: Some(303),
            },
        );
        let replacements = vec![PromptReplacement::sequence(
            Modality::Video,
            "<|video|>",
            vec![301, 301],
        )];

        let ranges =
            expand_prompt_token_ids(&mut prompt_token_ids, replacements, &placeholders).unwrap();

        assert_eq!(prompt_token_ids, vec![1, 302, 301, 301, 303, 2]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 4);
        assert_bool_mask(&ranges[0], &[false, true, true, false]);
    }

    #[test]
    fn expand_prompt_token_ids_handles_mixed_image_and_video_replacements() {
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, 300, 3];
        let mut placeholders = HashMap::new();
        placeholders.insert(
            Modality::Image,
            ResolvedPlaceholder {
                token: "<|image|>".to_string(),
                marker_token_id: LLAMA4_IMAGE_ID,
                embed_token_id: LLAMA4_PATCH_ID,
                wrapper_start_token_id: None,
                wrapper_end_token_id: None,
            },
        );
        placeholders.insert(
            Modality::Video,
            ResolvedPlaceholder {
                token: "<|video|>".to_string(),
                marker_token_id: 300,
                embed_token_id: 301,
                wrapper_start_token_id: None,
                wrapper_end_token_id: None,
            },
        );
        let replacements = vec![
            llama4_single_tile_replacement(),
            PromptReplacement::sequence(Modality::Video, "<|video|>", vec![301, 301]),
        ];

        let ranges =
            expand_prompt_token_ids(&mut prompt_token_ids, replacements, &placeholders).unwrap();

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
                301,
                301,
                3,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 5);
        assert_eq!(ranges[1].offset, 7);
        assert_eq!(ranges[1].length, 2);
        assert_bool_mask(&ranges[1], &[true, true]);
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

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            Vec::new(),
        )];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn build_features_handles_video_only_without_image_batch() {
        let mut info = llama4_info();
        info.spec.field_layouts.clear();

        let clip = Arc::new(VideoClip::new(
            Vec::new(),
            Bytes::from_static(b"video"),
            VideoSource::InlineBytes,
            "video-hash".to_string(),
        ));
        let fetched = FetchedMedia {
            items: vec![FetchedMediaItem::Video {
                clip,
                uuid: Some("video-uuid".to_string()),
            }],
        };
        let mut tensors = HashMap::new();
        tensors.insert(
            "pixel_values".to_string(),
            tensor::KwargValue::F32Tensor {
                data: vec![1.0, 2.0],
                shape: vec![2],
            },
        );
        let video_batches = vec![PreparedMediaBatch {
            len: 1,
            replacements: Vec::new(),
            tensors,
        }];
        let ranges = vec![PlaceholderRange {
            offset: 1,
            length: 2,
            is_embed: None,
        }];

        let features = info.build_features(fetched, None, video_batches, ranges).unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!(features[0].modality, "video");
        assert_eq!(features[0].identifier, "video-uuid");
        assert_eq!(features[0].mm_hash.as_deref(), Some("video-hash"));
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let replacements = vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ];

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
}
