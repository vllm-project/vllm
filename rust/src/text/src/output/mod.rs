//! Output processing helpers shared by text and chat layers.

pub use decoded::{DecodedTextEvent, TextDecodeOptions, decoded_text_event_stream};

mod decoded;

use futures::{StreamExt as _, pin_mut};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::{Error, Result, TextOutputStream};

/// Final decoded text plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedTextOutput {
    pub text: String,
    pub prompt_token_count: usize,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub stop_reason: Option<StopReason>,
}

#[allow(clippy::manual_async_fn, reason = "specify `Send` bound")]
#[easy_ext::ext(TextOutputStreamExt)]
impl<T: TextOutputStream> T {
    /// Collect the stream to completion and return the final decoded text plus terminal metadata.
    pub fn collect_output(self) -> impl Future<Output = Result<CollectedTextOutput>> + Send {
        async move {
            let stream = self;
            pin_mut!(stream);

            while let Some(event) = stream.next().await.transpose()? {
                match event {
                    DecodedTextEvent::Start | DecodedTextEvent::TextDelta { .. } => {}
                    DecodedTextEvent::Done {
                        text,
                        prompt_token_count,
                        token_ids,
                        finish_reason,
                        stop_reason,
                    } => {
                        return Ok(CollectedTextOutput {
                            text,
                            prompt_token_count,
                            token_ids,
                            finish_reason,
                            stop_reason,
                        });
                    }
                }
            }

            // Note: this is actually unreachable, as the underlying stream always emit an error on
            // unexpected close.
            Err(Error::StreamClosedBeforeTerminalOutput {
                request_id: "unknown".to_string(),
            })
        }
    }
}
