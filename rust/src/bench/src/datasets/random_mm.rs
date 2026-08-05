// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io::Cursor;
use std::sync::Arc;

use base64::Engine as _;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// A bucket key: (height, width, num_frames). num_frames=1 means image, >1 means video.
#[derive(Debug, Clone)]
pub struct MmBucketKey {
    pub height: u32,
    pub width: u32,
    pub num_frames: u32,
}

/// Per-modality hard caps.
#[derive(Debug, Clone)]
pub struct MmLimitPerPrompt {
    pub image: usize,
    pub video: usize,
}

impl Default for MmLimitPerPrompt {
    fn default() -> Self {
        Self {
            image: 255,
            video: 1,
        }
    }
}

/// Parse the limit-mm-per-prompt JSON string, e.g. `{"image": 3, "video": 0}`.
pub fn parse_limit_mm_per_prompt(s: &str) -> Result<MmLimitPerPrompt> {
    let v: serde_json::Value = serde_json::from_str(s)
        .map_err(|e| BenchError::Config(format!("Invalid --random-mm-limit-mm-per-prompt: {e}")))?;
    let obj = v.as_object().ok_or_else(|| {
        BenchError::Config("--random-mm-limit-mm-per-prompt must be a JSON object".into())
    })?;
    let image = obj.get("image").and_then(|v| v.as_u64()).unwrap_or(255) as usize;
    let video = obj.get("video").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    Ok(MmLimitPerPrompt { image, video })
}

/// Parse the bucket config string in Python-style syntax.
///
/// Accepts: `{(256,256,1): 0.5, (720,1280,1): 0.5}`
/// Each key is `(height, width, num_frames)` and value is the probability weight.
pub fn parse_bucket_config(s: &str) -> Result<Vec<(MmBucketKey, f64)>> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| BenchError::Config("Bucket config must be wrapped in {}".into()))?;

    let mut buckets = Vec::new();
    let chars: Vec<char> = inner.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace and commas
        while i < len && (chars[i].is_whitespace() || chars[i] == ',') {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Expect '('
        if chars[i] != '(' {
            return Err(BenchError::Config(format!(
                "Expected '(' in bucket config at position {i}"
            )));
        }
        i += 1;

        // Read until ')'
        let tuple_start = i;
        while i < len && chars[i] != ')' {
            i += 1;
        }
        if i >= len {
            return Err(BenchError::Config("Unclosed '(' in bucket config".into()));
        }
        let tuple_str: String = chars[tuple_start..i].iter().collect();
        i += 1; // skip ')'

        // Skip whitespace, then expect ':'
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len || chars[i] != ':' {
            return Err(BenchError::Config(
                "Expected ':' after tuple in bucket config".into(),
            ));
        }
        i += 1;

        // Skip whitespace
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }

        // Read the probability value until ',' or end
        let val_start = i;
        while i < len && chars[i] != ',' {
            i += 1;
        }
        let val_str: String = chars[val_start..i].iter().collect();

        // Parse tuple
        let parts: Vec<&str> = tuple_str.split(',').collect();
        if parts.len() != 3 {
            return Err(BenchError::Config(format!(
                "Bucket key must have 3 values (height,width,num_frames), got: ({tuple_str})"
            )));
        }

        let height: u32 = parts[0].trim().parse().map_err(|_| {
            BenchError::Config(format!(
                "Invalid height in bucket config: '{}'",
                parts[0].trim()
            ))
        })?;
        let width: u32 = parts[1].trim().parse().map_err(|_| {
            BenchError::Config(format!(
                "Invalid width in bucket config: '{}'",
                parts[1].trim()
            ))
        })?;
        let num_frames: u32 = parts[2].trim().parse().map_err(|_| {
            BenchError::Config(format!(
                "Invalid num_frames in bucket config: '{}'",
                parts[2].trim()
            ))
        })?;
        let prob: f64 = val_str.trim().parse().map_err(|_| {
            BenchError::Config(format!(
                "Invalid probability in bucket config: '{}'",
                val_str.trim()
            ))
        })?;

        if prob < 0.0 {
            return Err(BenchError::Config(format!(
                "Bucket probability must be non-negative, got: {prob}"
            )));
        }

        buckets.push((
            MmBucketKey {
                height,
                width,
                num_frames,
            },
            prob,
        ));
    }

    if buckets.is_empty() {
        return Err(BenchError::Config(
            "Bucket config must have at least one entry".into(),
        ));
    }

    Ok(buckets)
}

/// JSON fragment prefix/suffix for image content blocks.
const IMG_JSON_PREFIX: &str = r#"{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"#;
const IMG_JSON_SUFFIX: &str = r#""}}"#;

/// Generate a synthetic random JPEG image and return it as a pre-serialized JSON fragment.
///
/// Builds the complete JSON string in a single allocation:
/// `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<b64>"}}`
///
/// The base64 data is written directly into the final string — no intermediate
/// String or format!() copy.
fn generate_random_image(width: u32, height: u32, rng: &mut StdRng) -> Result<Arc<str>> {
    let pixel_count = (width as usize) * (height as usize) * 3;
    let mut pixels = vec![0u8; pixel_count];
    rng.fill(pixels.as_mut_slice());

    let img = image::RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| BenchError::Config("Failed to create image from random pixels".into()))?;

    // Pre-allocate JPEG buffer (random pixels compress poorly, estimate ~60% of raw)
    let estimated_jpeg = pixel_count * 3 / 5;
    let mut buf = Cursor::new(Vec::with_capacity(estimated_jpeg));
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| BenchError::Config(format!("Failed to encode JPEG: {e}")))?;

    let jpeg_bytes = buf.into_inner();

    // Pre-compute exact output size: prefix + base64_len + suffix
    let b64_len = jpeg_bytes.len().div_ceil(3) * 4;
    let total_len = IMG_JSON_PREFIX.len() + b64_len + IMG_JSON_SUFFIX.len();

    // Single allocation: write base64 directly into the JSON fragment string
    let mut json_fragment = String::with_capacity(total_len);
    json_fragment.push_str(IMG_JSON_PREFIX);
    base64::engine::general_purpose::STANDARD.encode_string(&jpeg_bytes, &mut json_fragment);
    json_fragment.push_str(IMG_JSON_SUFFIX);

    Ok(Arc::from(json_fragment))
}

/// Sample multimodal items for a single request.
///
/// Returns a list of (height, width, num_frames) tuples.
fn sample_mm_items(
    rng: &mut StdRng,
    min_items: usize,
    max_items: usize,
    buckets: &[(MmBucketKey, f64)],
    limit: &MmLimitPerPrompt,
) -> Vec<MmBucketKey> {
    let num_items = if min_items == max_items {
        min_items
    } else {
        rng.random_range(min_items..=max_items)
    };

    // Filter to non-zero probability buckets
    let active_buckets: Vec<&(MmBucketKey, f64)> =
        buckets.iter().filter(|(_, p)| *p > 0.0).collect();
    if active_buckets.is_empty() || num_items == 0 {
        return Vec::new();
    }

    let total_weight: f64 = active_buckets.iter().map(|(_, p)| p).sum();
    if total_weight <= 0.0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(num_items);
    let mut image_count = 0usize;
    let mut video_count = 0usize;

    for _ in 0..num_items {
        // Build normalized weights considering remaining capacity
        let mut weights: Vec<f64> = Vec::with_capacity(active_buckets.len());
        for (key, prob) in &active_buckets {
            let is_video = key.num_frames > 1;
            let at_limit = if is_video {
                video_count >= limit.video
            } else {
                image_count >= limit.image
            };
            weights.push(if at_limit { 0.0 } else { *prob });
        }

        let w_total: f64 = weights.iter().sum();
        if w_total <= 0.0 {
            break; // All modalities at limit
        }

        // Weighted random selection (strict `<` to avoid selecting zero-weight buckets)
        let r = rng.random::<f64>() * w_total;
        let mut cumulative = 0.0;
        // Default to last non-zero-weight bucket (floating-point accumulation fallback)
        let mut selected_idx = weights.iter().rposition(|w| *w > 0.0).unwrap_or(0);
        for (i, w) in weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                selected_idx = i;
                break;
            }
        }

        let (key, _) = &active_buckets[selected_idx];
        if key.num_frames > 1 {
            video_count += 1;
        } else {
            image_count += 1;
        }
        result.push(key.clone());
    }

    result
}

/// Generate random multimodal dataset.
///
/// Mirrors Python's RandomMultiModalDataset.sample() from datasets.py.
/// Generates text prompts with exact token lengths and random images/videos.
pub fn generate_random_mm_dataset(
    tokenizer: &TokenizerKind,
    num_requests: usize,
    input_len: usize,
    output_len: usize,
    prefix_len: usize,
    range_ratio: crate::config::RangeRatio,
    seed: u64,
    request_id_prefix: &str,
    base_items_per_request: usize,
    num_mm_items_range_ratio: f64,
    limit: &MmLimitPerPrompt,
    buckets: &[(MmBucketKey, f64)],
    enable_multimodal_chat: bool,
) -> Result<Vec<SampleRequest>> {
    if !(0.0..=1.0).contains(&num_mm_items_range_ratio) {
        return Err(BenchError::Config(
            "num_mm_items_range_ratio must be in [0, 1]".into(),
        ));
    }

    // Check for video buckets with non-zero probability
    for (key, prob) in buckets {
        if key.num_frames > 1 && *prob > 0.0 {
            return Err(BenchError::Config(
                "Video generation (num_frames > 1) is not yet supported in Rust. \
                 Set video bucket probabilities to 0.0."
                    .into(),
            ));
        }
    }

    // Compute item count bounds
    let n = base_items_per_request as f64;
    let r = num_mm_items_range_ratio;
    let min_items = (n * (1.0 - r)).floor().max(0.0) as usize;
    let max_items = (n * (1.0 + r)).ceil() as usize;
    // Clamp to total modality limit
    let total_limit = limit.image + limit.video;
    let max_items = max_items.min(total_limit);
    let min_items = min_items.min(max_items);

    let vocab_size = tokenizer.vocab_size();
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }

    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = input_len.saturating_sub(num_special);

    // Python semantics: sample uniformly from [len*(1-r), len*(1+r)].
    let (input_low, input_high) = range_ratio.input_bounds(real_input_len);
    let (output_low, output_high) = range_ratio.output_bounds(output_len);

    // Pre-generate per-request params
    let mut rng = StdRng::seed_from_u64(seed);

    struct RequestParams {
        input_len: usize,
        output_len: usize,
        offset: usize,
    }

    let params: Vec<RequestParams> = (0..num_requests)
        .map(|_| {
            let il = if input_low == input_high {
                input_low
            } else {
                rng.random_range(input_low..=input_high)
            };
            let ol = if output_low == output_high {
                output_low
            } else {
                rng.random_range(output_low..=output_high)
            };
            let off = rng.random_range(0..vocab_size as usize);
            RequestParams {
                input_len: il,
                output_len: ol,
                offset: off,
            }
        })
        .collect();

    // Pre-generate multimodal item configs per request
    let mm_configs: Vec<Vec<MmBucketKey>> = (0..num_requests)
        .map(|_| sample_mm_items(&mut rng, min_items, max_items, buckets, limit))
        .collect();

    // Generate text prompts (need text for chat backend, not just token IDs)
    let prefix_token_ids = if prefix_len > 0 {
        generate_prefix(tokenizer, &allowed_tokens, prefix_len, seed)?
    } else {
        Vec::new()
    };

    // Generate token sequences
    let prefix_ref = &prefix_token_ids;
    let allowed_ref = &allowed_tokens;

    let token_sequences: Vec<Vec<u32>> = params
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let at_len = allowed_ref.len();
            let mut seq = Vec::with_capacity(prefix_ref.len() + p.input_len);
            seq.extend_from_slice(prefix_ref);
            for j in 0..p.input_len {
                seq.push(allowed_ref[(p.offset + i + j) % at_len]);
            }
            seq
        })
        .collect();

    let target_lens: Vec<usize> = params.iter().map(|p| prefix_len + p.input_len).collect();

    // Decode tokens to text (chat backend needs text prompts for multimodal)
    let prompts: Result<Vec<String>> = token_sequences
        .into_par_iter()
        .enumerate()
        .map(|(i, tokens)| {
            let (text, _adjusted) = super::random::gen_prompt_decode_to_target_len(
                tokenizer,
                &tokens,
                target_lens[i],
                false,
                allowed_ref,
            )?;
            Ok(text)
        })
        .collect();
    let prompts = prompts?;

    // Generate images for each request (parallel per request)
    // Each request gets its own RNG seeded deterministically.
    let rid_prefix = request_id_prefix.to_string();
    let result: Vec<SampleRequest> = prompts
        .into_par_iter()
        .enumerate()
        .map(|(i, prompt)| {
            let mut item_rng =
                StdRng::seed_from_u64(seed.wrapping_add(i as u64).wrapping_add(0xBEEF));
            let mm_items: Vec<Arc<str>> = mm_configs[i]
                .iter()
                .map(|key| {
                    generate_random_image(key.width, key.height, &mut item_rng)
                        .expect("Image generation should not fail")
                })
                .collect();

            let mm_content: Option<Arc<[Arc<str>]>> = if mm_items.is_empty() {
                None
            } else {
                Some(Arc::from(mm_items))
            };

            // --enable-multimodal-chat: pre-build the full chat `messages` array
            // (text part + mm items) at dataset time, mirroring Python's
            // apply_multimodal_chat_transformation. mm content moves inside the
            // messages string; the backend splices it verbatim.
            let (mm_content, chat_messages_json) = if enable_multimodal_chat {
                let msgs = build_chat_messages_json(&prompt, mm_content.as_deref());
                (None, Some(Arc::from(msgs.as_str())))
            } else {
                (mm_content, None)
            };

            SampleRequest {
                prompt: Arc::from(prompt.as_str()),
                prompt_len: target_lens[i],
                expected_output_len: params[i].output_len,
                request_id: Some(format!("{rid_prefix}{i}")),
                multi_modal_content: mm_content,
                chat_messages_json,
                ..Default::default()
            }
        })
        .collect();

    Ok(result)
}

/// Pre-serialize the OpenAI chat `messages` array for --enable-multimodal-chat.
///
/// Produces `[{"role":"user","content":[{"type":"text","text":"..."},<frag>,...]}]`
/// by concatenating the JSON-escaped prompt with the pre-serialized mm fragments,
/// so the ~200KB+ base64 image data is never parsed or re-serialized.
pub(crate) fn build_chat_messages_json(prompt: &str, mm_items: Option<&[Arc<str>]>) -> String {
    let mm_total: usize =
        mm_items.map(|items| items.iter().map(|f| f.len() + 1).sum()).unwrap_or(0);
    let mut msgs = String::with_capacity(64 + prompt.len() * 2 + mm_total);
    msgs.push_str(r#"[{"role":"user","content":[{"type":"text","text":"#);
    // serde_json::to_string on &str produces a JSON-escaped quoted string
    msgs.push_str(&serde_json::to_string(prompt).unwrap());
    msgs.push('}');
    for fragment in mm_items.unwrap_or(&[]) {
        msgs.push(',');
        msgs.push_str(fragment);
    }
    msgs.push_str("]}]");
    msgs
}

fn generate_prefix(
    tokenizer: &TokenizerKind,
    allowed_tokens: &[u32],
    prefix_len: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD));
    let tokens: Vec<u32> = (0..prefix_len)
        .map(|_| allowed_tokens[rng.random_range(0..allowed_tokens.len())])
        .collect();

    let (_, adjusted) = super::random::gen_prompt_decode_to_target_len(
        tokenizer,
        &tokens,
        prefix_len,
        false,
        allowed_tokens,
    )?;
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bucket_config_basic() {
        let input = "{(256,256,1): 0.5, (720,1280,1): 0.5}";
        let buckets = parse_bucket_config(input).unwrap();
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].0.height, 256);
        assert_eq!(buckets[0].0.width, 256);
        assert_eq!(buckets[0].0.num_frames, 1);
        assert!((buckets[0].1 - 0.5).abs() < 1e-10);
        assert_eq!(buckets[1].0.height, 720);
        assert_eq!(buckets[1].0.width, 1280);
    }

    #[test]
    fn test_parse_bucket_config_single() {
        let input = "{(1024, 800, 1): 1.0}";
        let buckets = parse_bucket_config(input).unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].0.height, 1024);
        assert_eq!(buckets[0].0.width, 800);
        assert_eq!(buckets[0].0.num_frames, 1);
        assert!((buckets[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_bucket_config_with_video() {
        let input = "{(256,256,1): 0.4, (720,1280,1): 0.4, (720,1280,16): 0.2}";
        let buckets = parse_bucket_config(input).unwrap();
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[2].0.num_frames, 16);
    }

    #[test]
    fn test_build_chat_messages_json_valid_and_ordered() {
        let frag: Arc<str> =
            Arc::from(r#"{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AAAA"}}"#);
        let msgs = build_chat_messages_json("hi \"there\"\nline2", Some(&[frag]));
        let v: serde_json::Value = serde_json::from_str(&msgs).expect("must be valid JSON");
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["role"], "user");
        let content = v[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hi \"there\"\nline2");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn test_build_chat_messages_json_text_only() {
        let msgs = build_chat_messages_json("plain", None);
        let v: serde_json::Value = serde_json::from_str(&msgs).unwrap();
        let content = v[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "plain");
    }

    #[test]
    fn test_parse_limit_mm_per_prompt() {
        let input = r#"{"image": 3, "video": 0}"#;
        let limit = parse_limit_mm_per_prompt(input).unwrap();
        assert_eq!(limit.image, 3);
        assert_eq!(limit.video, 0);
    }

    #[test]
    fn test_parse_limit_mm_per_prompt_defaults() {
        let input = r#"{}"#;
        let limit = parse_limit_mm_per_prompt(input).unwrap();
        assert_eq!(limit.image, 255);
        assert_eq!(limit.video, 1);
    }

    #[test]
    fn test_sample_mm_items_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let buckets = vec![
            (
                MmBucketKey {
                    height: 256,
                    width: 256,
                    num_frames: 1,
                },
                0.5,
            ),
            (
                MmBucketKey {
                    height: 720,
                    width: 1280,
                    num_frames: 1,
                },
                0.5,
            ),
        ];
        let limit = MmLimitPerPrompt { image: 5, video: 0 };
        let items = sample_mm_items(&mut rng, 2, 3, &buckets, &limit);
        assert!(items.len() >= 2 && items.len() <= 3);
        for item in &items {
            assert_eq!(item.num_frames, 1);
        }
    }

    #[test]
    fn test_sample_mm_items_respects_limit() {
        let mut rng = StdRng::seed_from_u64(42);
        let buckets = vec![(
            MmBucketKey {
                height: 256,
                width: 256,
                num_frames: 1,
            },
            1.0,
        )];
        let limit = MmLimitPerPrompt { image: 2, video: 0 };
        let items = sample_mm_items(&mut rng, 5, 5, &buckets, &limit);
        // Should be capped at 2 due to image limit
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_generate_random_image() {
        let mut rng = StdRng::seed_from_u64(42);
        let result = generate_random_image(64, 64, &mut rng).unwrap();
        // Result is a pre-serialized JSON fragment
        assert!(
            result
                .starts_with(r#"{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"#)
        );
        assert!(result.ends_with(r#""}}"#));
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["type"], "image_url");
    }
}
