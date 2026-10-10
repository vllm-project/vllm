// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::TemplateMessage;
use crate::{ChatContent, ChatMessage, MediaPartSource};

/// Keep media features aligned with placeholders when system messages move.
pub(super) fn consolidated_media_order(
    source: &[ChatMessage],
    rendered: &[TemplateMessage<'_>],
) -> Vec<MediaPartSource> {
    let mut system = Vec::new();
    let mut other = Vec::new();
    for (message_index, (source, rendered)) in source.iter().zip(rendered).enumerate() {
        let content = match source {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. }
            | ChatMessage::Custom { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        let destination = if rendered.role == "system" {
            &mut system
        } else {
            &mut other
        };
        destination.extend(
            parts.iter().enumerate().filter_map(|(content_part_index, part)| {
                part.is_multimodal().then_some(MediaPartSource {
                    message_index,
                    content_part_index,
                })
            }),
        );
    }
    system.extend(other);
    system
}
