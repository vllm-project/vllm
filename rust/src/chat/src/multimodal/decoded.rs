// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Images that the caller decoded before the request reached this crate.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use image::{DynamicImage, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};
use llm_multimodal::{ImageDetail, ImageFrame, ImageSource, hasher};

use crate::error::{Result, bail_invalid_decoded_image, invalid_decoded_image};

/// An image that the caller already decoded to contiguous 8-bit pixels in
/// `[height, width, channels]` order: 1 channel is L, 2 is LA, 3 is RGB, and
/// 4 is RGBA.
///
/// The frame keeps the channel layout, so the image goes through the same RGB
/// conversion as an encoded image with that layout.
///
/// The content hash covers the dimensions, the channel count, and the pixels.
/// A URL or an encoded image gets a hash of its encoded bytes instead, so the
/// same picture sent in both forms does not share processor or encoder cache
/// entries.
#[derive(Clone)]
pub struct DecodedImage {
    frame: Arc<ImageFrame>,
    uuid: Option<String>,
}

impl DecodedImage {
    /// Validate the pixel buffer and compute its content hash.
    ///
    /// The hash reads every pixel, so call this from a blocking context for
    /// large images.
    pub fn new(
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        channels: u32,
        uuid: Option<String>,
    ) -> Result<Self> {
        let expected = Self::byte_len(width, height, channels)?;
        if pixels.len() != expected {
            bail_invalid_decoded_image!(
                "{} bytes do not match {width}x{height}x{channels} ({expected} bytes)",
                pixels.len()
            );
        }
        let hash = content_hash(width, height, channels, &pixels);
        let image = match channels {
            1 => GrayImage::from_raw(width, height, pixels).map(DynamicImage::ImageLuma8),
            2 => GrayAlphaImage::from_raw(width, height, pixels).map(DynamicImage::ImageLumaA8),
            3 => RgbImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgb8),
            _ => RgbaImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgba8),
        }
        .ok_or_else(|| invalid_decoded_image!("pixels do not fit {width}x{height}x{channels}"))?;
        let frame = ImageFrame::new(
            image,
            // Nothing downstream reads the encoded bytes, and there are none.
            bytes::Bytes::new(),
            ImageDetail::default(),
            ImageSource::InlineBytes,
            hash,
        );
        Ok(Self {
            frame: Arc::new(frame),
            uuid,
        })
    }

    /// Byte length of a `width` x `height` image with 8-bit `channels`.
    pub fn byte_len(width: u32, height: u32, channels: u32) -> Result<usize> {
        if width == 0 || height == 0 {
            bail_invalid_decoded_image!("{width}x{height} has no pixels");
        }
        if !(1..=4).contains(&channels) {
            bail_invalid_decoded_image!("{channels} channels is not 1, 2, 3, or 4");
        }
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(channels as usize))
            .ok_or_else(|| invalid_decoded_image!("{width}x{height}x{channels} overflows usize"))
    }

    pub fn hash(&self) -> &str {
        &self.frame.hash
    }

    pub(super) fn into_parts(self) -> (Arc<ImageFrame>, Option<String>) {
        (self.frame, self.uuid)
    }
}

impl fmt::Debug for DecodedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = self.frame.size();
        f.debug_struct("DecodedImage")
            .field("width", &size.width)
            .field("height", &size.height)
            .field("color", &self.frame.image().ok().map(DynamicImage::color))
            .field("hash", &self.frame.hash)
            .field("uuid", &self.uuid)
            .finish()
    }
}

/// A load that gives a decoded image, for example a read from the memory of
/// another process.
///
/// Chat polls the load only after the request passes validation, so the
/// future must not start work before it is first polled. After that, chat
/// polls it until it ends, even when another item of the request fails.
pub struct DeferredImage(BoxFuture<'static, Result<DecodedImage>>);

impl DeferredImage {
    pub fn new(load: impl Future<Output = Result<DecodedImage>> + Send + 'static) -> Self {
        Self(Box::pin(load))
    }

    pub(super) fn load(self) -> BoxFuture<'static, Result<DecodedImage>> {
        self.0
    }
}

impl fmt::Debug for DeferredImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DeferredImage")
    }
}

/// Blake3 over the layout and the pixel digest, so the same bytes with
/// another shape get another key.
fn content_hash(width: u32, height: u32, channels: u32, pixels: &[u8]) -> String {
    let pixels = hasher::hash_image(pixels);
    hasher::hash_image(format!("u8x{channels}:{width}x{height}:{pixels}").as_bytes())
}

#[cfg(test)]
mod tests {
    use thiserror_ext::AsReport as _;

    use super::*;

    #[test]
    fn decoded_image_builds_a_frame_of_the_declared_size() {
        let image = DecodedImage::new(vec![7; 3 * 2 * 3], 3, 2, 3, Some("id".to_owned())).unwrap();
        let (frame, uuid) = image.into_parts();
        assert_eq!(frame.size(), llm_multimodal::ImageSize::new(3, 2));
        assert_eq!(uuid.as_deref(), Some("id"));
        assert!(frame.raw_bytes().is_empty());
    }

    #[test]
    fn decoded_image_rejects_a_buffer_that_does_not_match_its_dimensions() {
        for len in [17, 19] {
            let error = DecodedImage::new(vec![0; len], 3, 2, 3, None).unwrap_err();
            assert!(error.is_request_validation_error());
            assert!(
                error.to_report_string().contains("(18 bytes)"),
                "{}",
                error.to_report_string()
            );
        }
        for (width, height, channels) in [(0, 2, 3), (2, 0, 3), (2, 2, 0), (2, 2, 5)] {
            assert!(DecodedImage::new(Vec::new(), width, height, channels, None).is_err());
        }
    }

    #[test]
    fn frames_keep_the_channel_layout() {
        for (channels, color) in [
            (1, ::image::ColorType::L8),
            (2, ::image::ColorType::La8),
            (3, ::image::ColorType::Rgb8),
            (4, ::image::ColorType::Rgba8),
        ] {
            let image =
                DecodedImage::new(vec![9; 2 * channels as usize], 2, 1, channels, None).unwrap();
            assert_eq!(image.frame.image().unwrap().color(), color);
        }
    }

    #[test]
    fn content_hash_covers_pixels_and_shape() {
        let pixels = vec![1u8; 12];
        let wide = DecodedImage::new(pixels.clone(), 4, 1, 3, None).unwrap();
        let tall = DecodedImage::new(pixels.clone(), 1, 4, 3, None).unwrap();
        let rgba = DecodedImage::new(pixels.clone(), 3, 1, 4, None).unwrap();
        let again = DecodedImage::new(pixels, 4, 1, 3, None).unwrap();
        let other = DecodedImage::new(vec![2u8; 12], 4, 1, 3, None).unwrap();
        assert_eq!(wide.hash(), again.hash());
        assert_ne!(wide.hash(), tall.hash());
        assert_ne!(wide.hash(), rgba.hash());
        assert_ne!(wide.hash(), other.hash());
        // Same form as the URL path's blake3 keys.
        assert_eq!(wide.hash().len(), 64);
        assert!(wide.hash().chars().all(|c| c.is_ascii_hexdigit()));
    }
}
