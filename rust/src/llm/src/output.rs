use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use futures::Stream;
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::{EngineCoreOutput, RequestOutputKind};

use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    pub request_id: String,
    pub prompt_token_ids: Arc<[u32]>,
    pub token_ids: Vec<u32>,
    pub raw: EngineCoreOutput,
}

pub struct GenerateOutputStream {
    output_kind: RequestOutputKind,
    prompt_token_ids: Arc<[u32]>,
    raw_stream: EngineCoreOutputStream,
    cumulative_token_ids: Vec<u32>,
}

impl GenerateOutputStream {
    fn new(
        output_kind: RequestOutputKind,
        prompt_token_ids: Arc<[u32]>,
        raw_stream: EngineCoreOutputStream,
    ) -> Self {
        Self {
            output_kind,
            prompt_token_ids,
            raw_stream,
            cumulative_token_ids: Vec::new(),
        }
    }
}

pub(crate) fn adapt_output_stream(
    output_kind: RequestOutputKind,
    prompt_token_ids: Arc<[u32]>,
    stream: EngineCoreOutputStream,
) -> GenerateOutputStream {
    GenerateOutputStream::new(output_kind, prompt_token_ids, stream)
}

impl Stream for GenerateOutputStream {
    type Item = Result<GenerateOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let raw = match ready!(Pin::new(&mut self.raw_stream).poll_next(cx)) {
                Some(Ok(raw)) => raw,
                Some(Err(error)) => return Poll::Ready(Some(Err(error.into()))),
                None => return Poll::Ready(None),
            };
            let finished = raw.finished();

            let output = match self.output_kind {
                RequestOutputKind::Delta => Some(GenerateOutput {
                    request_id: raw.request_id.clone(),
                    prompt_token_ids: self.prompt_token_ids.clone(),
                    token_ids: raw.new_token_ids.clone(),
                    raw,
                }),
                RequestOutputKind::Cumulative => {
                    self.cumulative_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    Some(GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_token_ids: self.prompt_token_ids.clone(),
                        token_ids: self.cumulative_token_ids.clone(),
                        raw,
                    })
                }
                RequestOutputKind::FinalOnly => {
                    self.cumulative_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    finished.then(|| GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_token_ids: self.prompt_token_ids.clone(),
                        token_ids: self.cumulative_token_ids.clone(),
                        raw,
                    })
                }
            };

            if let Some(output) = output {
                return Poll::Ready(Some(Ok(output)));
            }
        }
    }
}
