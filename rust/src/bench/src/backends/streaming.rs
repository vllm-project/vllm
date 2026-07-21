// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/// SSE streaming response handler.
///
/// Accumulates incoming byte chunks and extracts complete SSE messages.
/// Mirrors Python's `StreamedResponseHandler` from endpoint_request_func.py:22-60.
pub struct StreamedResponseHandler {
    buffer: String,
    /// Reusable message buffer — avoids allocating a new Vec per `add_chunk` call.
    messages: Vec<String>,
}

impl StreamedResponseHandler {
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(4096),
            messages: Vec::with_capacity(4),
        }
    }

    /// Add a chunk of bytes and return any complete SSE messages.
    ///
    /// The returned slice borrows from the handler and is valid until the next
    /// `add_chunk` call.
    pub fn add_chunk(&mut self, chunk_bytes: &[u8]) -> &[String] {
        self.messages.clear();

        let chunk_str = String::from_utf8_lossy(chunk_bytes);
        self.buffer.push_str(&chunk_str);

        // Split by double newlines (SSE message separator)
        while let Some(pos) = self.buffer.find("\n\n") {
            let message = self.buffer[..pos].trim().to_string();
            // Efficiently remove consumed bytes by shifting remaining data
            self.buffer.drain(..pos + 2);
            if !message.is_empty() {
                self.messages.push(message);
            }
        }

        // Handle buffered data without trailing `\n\n`.
        // Matches Python's speculative json.loads() in StreamedResponseHandler.
        // This matters for TTFT/ITL accuracy: when a data message and its `\n\n`
        // arrive in separate TCP segments, we want to emit the message at the
        // first segment's arrival time, not the second.
        //
        // Also handles multi-field SSE events where the buffer may start with
        // "event: ...\ndata: ..." (Dynamo frontend).
        let data_start = if self.buffer.starts_with("data: ") {
            Some(0)
        } else {
            // Look for a "data: " line in multi-field events
            self.buffer.find("\ndata: ").map(|p| p + 1)
        };
        if let Some(offset) = data_start {
            let content = self.buffer[offset + 6..].trim();
            if content == "[DONE]"
                || (!content.is_empty()
                    && serde_json::from_str::<&serde_json::value::RawValue>(content).is_ok())
            {
                self.messages.push(self.buffer.trim().to_string());
                self.buffer.clear();
            }
        }

        &self.messages
    }
}

/// Trim leading/trailing ASCII whitespace from a byte slice.
pub fn trim_bytes(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|p| p + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sse() {
        let mut handler = StreamedResponseHandler::new();
        let msgs =
            handler.add_chunk(b"data: {\"choices\":[{\"text\":\"hi\"}]}\n\ndata: [DONE]\n\n");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].contains("choices"));
        assert!(msgs[1].contains("[DONE]"));
    }

    #[test]
    fn test_split_chunks() {
        let mut handler = StreamedResponseHandler::new();
        let msgs1 = handler.add_chunk(b"data: {\"cho");
        assert!(msgs1.is_empty());
        let msgs2 = handler.add_chunk(b"ices\":[{\"text\":\"a\"}]}\n\n");
        assert_eq!(msgs2.len(), 1);
    }

    #[test]
    fn test_comment_lines() {
        let mut handler = StreamedResponseHandler::new();
        let msgs = handler.add_chunk(b": ping\n\ndata: {\"test\":1}\n\n");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].starts_with(":"));
    }

    #[test]
    fn test_done_without_newlines() {
        let mut handler = StreamedResponseHandler::new();
        let msgs = handler.add_chunk(b"data: [DONE]");
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("[DONE]"));
    }

    #[test]
    fn test_incomplete_json_in_buffer() {
        let mut handler = StreamedResponseHandler::new();
        let msgs = handler.add_chunk(b"data: {\"partial\":");
        assert!(msgs.is_empty());
        // Complete JSON without \n\n — speculative parse emits it
        let msgs2 = handler.add_chunk(b"true}");
        assert_eq!(msgs2.len(), 1);
        assert!(msgs2[0].contains("partial"));
    }

    #[test]
    fn test_multi_field_sse_event() {
        // Dynamo frontend sends "event: message\ndata: {...}\n\n"
        let mut handler = StreamedResponseHandler::new();
        let msgs =
            handler.add_chunk(b"event: message\ndata: {\"choices\":[{\"text\":\"hi\"}]}\n\n");
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("choices"));
        assert!(msgs[0].contains("event: message"));
    }

    #[test]
    fn test_multi_field_sse_speculative_parse() {
        // Multi-field event without trailing \n\n — speculative parse should emit it
        let mut handler = StreamedResponseHandler::new();
        let msgs = handler.add_chunk(b"event: message\ndata: {\"choices\":[{\"text\":\"hi\"}]}");
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("choices"));
    }
}
