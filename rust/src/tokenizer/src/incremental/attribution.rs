// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::mem::take;

use smallvec::SmallVec;

/// Position of one generated token in decoded text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenAnchor {
    /// The token contributed to visible decoded text.
    Visible { byte_offset: u32 },
    /// The token produced no visible decoded bytes.
    ZeroWidth { byte_offset: u32 },
}

impl TokenAnchor {
    fn byte_offset(self) -> u32 {
        match self {
            Self::Visible { byte_offset } | Self::ZeroWidth { byte_offset } => byte_offset,
        }
    }

    /// Return a new anchor offset by `delta` bytes.
    #[must_use]
    pub(super) fn offset_by(self, delta: i64) -> Self {
        let byte_offset = i64::from(self.byte_offset())
            .checked_add(delta)
            .and_then(|offset| u32::try_from(offset).ok())
            .expect("token anchor byte offset out of range");
        match self {
            Self::Visible { .. } => Self::Visible { byte_offset },
            Self::ZeroWidth { .. } => Self::ZeroWidth { byte_offset },
        }
    }
}

/// One generated token and its position in decoded text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenAttribution {
    /// Token ID passed to [`super::IncrementalDecoder::push_token`].
    pub token_id: u32,
    /// Position of the token in the decoded text.
    pub anchor: TokenAnchor,
}

/// Decoded text and the generated tokens attributed to it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodedText {
    /// Decoded UTF-8 text.
    pub text: String,
    /// One entry per generated token, in generation order.
    ///
    /// Offsets are local to `text`. Visible offsets may repeat when multiple
    /// tokens jointly decode into one character. A zero-width offset may equal
    /// `text.len()`.
    /// Four inline records cover one complete UTF-8 byte-fallback sequence.
    pub attributions: SmallVec<[TokenAttribution; 4]>,
}

impl DecodedText {
    /// Create decoded text without token attribution records. Mostly for testing purposes.
    pub fn unattributed(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            attributions: SmallVec::new(),
        }
    }

    /// Return true when both the text and the attributions are empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.attributions.is_empty()
    }

    /// Take the decoded text and attributions, leaving `self` empty.
    pub fn take(&mut self) -> Self {
        take(self)
    }

    /// Clear the text and attributions, keeping the allocations.
    pub fn clear(&mut self) {
        self.text.clear();
        self.attributions.clear();
    }

    /// Split off and return the prefix `[0, len)`; the suffix stays in `self`,
    /// with its token anchors rebased by `-len`.
    ///
    /// Anchors follow the first-byte ownership rule: a `Visible` token moves to
    /// the prefix when its first byte lies inside the prefix, so a token
    /// straddling the split belongs to the span holding its first byte. A
    /// `ZeroWidth` token moves to the prefix when its offset is `<= len`,
    /// matching the `take_ready` cutoff rule and letting
    /// `drain_prefix(text.len())` take trailing zero-width tokens. Relative
    /// order across anchor kinds is preserved.
    ///
    /// Panics if `len` is not a char boundary, via `String::drain`.
    #[must_use]
    pub fn drain_prefix(&mut self, len: usize) -> DecodedText {
        let text: String = self.text.drain(..len).collect();
        let len = offset_as_u32(len);

        let mut prefix = DecodedText {
            text,
            attributions: SmallVec::new(),
        };
        for attribution in take(&mut self.attributions) {
            let in_prefix = match attribution.anchor {
                TokenAnchor::Visible { byte_offset } => byte_offset < len,
                TokenAnchor::ZeroWidth { byte_offset } => byte_offset <= len,
            };
            if in_prefix {
                prefix.attributions.push(attribution);
            } else {
                self.attributions.push(TokenAttribution {
                    token_id: attribution.token_id,
                    anchor: attribution.anchor.offset_by(-i64::from(len)),
                });
            }
        }
        prefix
    }

    /// Append another decoded fragment, rebasing its token anchors.
    pub fn append(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }

        let byte_offset = offset_as_u32(self.text.len());
        let other_len = offset_as_u32(other.text.len());
        let _combined_len = byte_offset.checked_add(other_len).expect("decoded text exceeds 4 GiB");

        self.text.push_str(&other.text);
        self.attributions.extend(other.attributions.into_iter().map(|attribution| {
            TokenAttribution {
                token_id: attribution.token_id,
                anchor: attribution.anchor.offset_by(i64::from(byte_offset)),
            }
        }));
    }
}

#[derive(Clone, Copy, Debug)]
enum PendingAnchor {
    Unresolved,
    Resolved(TokenAnchor),
}

#[derive(Clone, Copy, Debug)]
struct PendingAttribution {
    token_id: u32,
    anchor: PendingAnchor,
}

/// Cumulative decoded text plus the attribution state needed to emit local
/// chunks.
#[derive(Default)]
pub(super) struct AttributedTextBuffer {
    text: String,
    emitted_byte_offset: usize,
    attributions: Vec<PendingAttribution>,
    emitted_attribution_count: usize,
    pending_start: Option<usize>,
}

impl AttributedTextBuffer {
    /// Return all decoded text accumulated so far.
    pub(super) fn text(&self) -> &str {
        &self.text
    }

    /// Return the decoded text length in bytes.
    pub(super) fn len(&self) -> usize {
        self.text.len()
    }

    /// Record a token awaiting visible decoded output.
    pub(super) fn record_pending_token(&mut self, token_id: u32) {
        self.pending_start.get_or_insert(self.attributions.len());
        self.attributions.push(PendingAttribution {
            token_id,
            anchor: PendingAnchor::Unresolved,
        });
    }

    /// Record a token that produced no visible decoded output.
    pub(super) fn record_zero_width_token(&mut self, token_id: u32) {
        self.attributions.push(PendingAttribution {
            token_id,
            anchor: PendingAnchor::Resolved(TokenAnchor::ZeroWidth {
                byte_offset: offset_as_u32(self.text.len()),
            }),
        });
    }

    /// Append visible text and anchor pending tokens at its start.
    pub(super) fn append_visible_text(&mut self, text: &str) {
        let byte_offset = offset_as_u32(self.text.len());
        if let Some(pending_start) = self.pending_start.take() {
            for attribution in &mut self.attributions[pending_start..] {
                if matches!(attribution.anchor, PendingAnchor::Unresolved) {
                    attribution.anchor =
                        PendingAnchor::Resolved(TokenAnchor::Visible { byte_offset });
                }
            }
        }
        self.text.push_str(text);
    }

    /// Resolve remaining pending tokens at the current text boundary.
    pub(super) fn resolve_pending_zero_width(&mut self) {
        let Some(pending_start) = self.pending_start.take() else {
            return;
        };
        let byte_offset = offset_as_u32(self.text.len());
        for attribution in &mut self.attributions[pending_start..] {
            if matches!(attribution.anchor, PendingAnchor::Unresolved) {
                attribution.anchor =
                    PendingAnchor::Resolved(TokenAnchor::ZeroWidth { byte_offset });
            }
        }
    }

    /// Truncate text and collapse removed anchors to the new boundary.
    pub(super) fn truncate(&mut self, byte_offset: usize) {
        self.text.truncate(byte_offset);
        let byte_offset = offset_as_u32(byte_offset);
        for attribution in &mut self.attributions {
            let PendingAnchor::Resolved(anchor) = &mut attribution.anchor else {
                continue;
            };
            if anchor.byte_offset() >= byte_offset {
                *anchor = TokenAnchor::ZeroWidth { byte_offset };
            }
        }
    }

    /// Emit decoded text and attributions ready through `cutoff`.
    pub(super) fn take_ready(&mut self, cutoff: usize) -> Option<DecodedText> {
        let chunk_start = self.emitted_byte_offset;
        let cutoff_u32 = offset_as_u32(cutoff);
        let mut attribution_end = self.emitted_attribution_count;

        for attribution in &self.attributions[self.emitted_attribution_count..] {
            let PendingAnchor::Resolved(anchor) = attribution.anchor else {
                break;
            };
            let ready = match anchor {
                TokenAnchor::Visible { byte_offset } => byte_offset < cutoff_u32,
                TokenAnchor::ZeroWidth { byte_offset } => byte_offset <= cutoff_u32,
            };
            if !ready {
                break;
            }
            attribution_end += 1;
        }

        if cutoff == chunk_start && attribution_end == self.emitted_attribution_count {
            return None;
        }

        let chunk_start_u32 = offset_as_u32(chunk_start);
        let attributions = self.attributions[self.emitted_attribution_count..attribution_end]
            .iter()
            .map(|attribution| {
                let PendingAnchor::Resolved(anchor) = attribution.anchor else {
                    unreachable!("ready attributions must be resolved")
                };
                TokenAttribution {
                    token_id: attribution.token_id,
                    anchor: anchor.offset_by(-i64::from(chunk_start_u32)),
                }
            })
            .collect();
        let text = self.text[chunk_start..cutoff].to_string();

        self.emitted_byte_offset = cutoff;
        self.emitted_attribution_count = attribution_end;
        Some(DecodedText { text, attributions })
    }

    /// Emit the final chunk and take the complete decoded output.
    pub(super) fn finish(&mut self) -> (Option<DecodedText>, DecodedText) {
        let last_chunk = self.take_ready(self.text.len());
        let attributions = take(&mut self.attributions)
            .into_iter()
            .map(|attribution| {
                let PendingAnchor::Resolved(anchor) = attribution.anchor else {
                    unreachable!("finish requires every attribution to be resolved")
                };
                TokenAttribution {
                    token_id: attribution.token_id,
                    anchor,
                }
            })
            .collect();
        let full = DecodedText {
            text: take(&mut self.text),
            attributions,
        };
        self.emitted_byte_offset = 0;
        self.emitted_attribution_count = 0;
        self.pending_start = None;
        (last_chunk, full)
    }
}

fn offset_as_u32(offset: usize) -> u32 {
    u32::try_from(offset).expect("decoded text exceeds 4 GiB")
}

#[cfg(test)]
mod tests;
