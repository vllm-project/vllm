// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::Histogram;

const REQUEST_RENDER_DURATION_BUCKETS: [f64; 16] = [
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0,
];

fn request_render_duration_histogram() -> Histogram {
    Histogram::new(REQUEST_RENDER_DURATION_BUCKETS)
}

/// The kind of request rendered, the `request_type` label.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RenderRequestType {
    Chat,
    Completion,
}

impl EncodeLabelValue for RenderRequestType {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        let value = match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
        };
        EncodeLabelValue::encode(&value, encoder)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RenderLabels {
    pub model_name: String,
    pub request_type: RenderRequestType,
}

pub(crate) type RenderHistogramFamily = Family<RenderLabels, Histogram, fn() -> Histogram>;

/// Online renderer Prometheus families.
///
/// Original Python definition: `vllm/renderers/metrics.py`.
pub struct RendererMetrics {
    pub request_render_duration_seconds: RenderHistogramFamily,
}

impl RendererMetrics {
    /// Register the renderer metric families into the shared registry.
    pub(crate) fn register(registry: &mut Registry) -> Self {
        let request_render_duration_seconds = RenderHistogramFamily::new_with_constructor(
            request_render_duration_histogram as fn() -> Histogram,
        );
        registry.register(
            "vllm:request_render_duration_seconds",
            "Time spent rendering a request, including chat templating, tokenization, and \
             multimodal preprocessing",
            request_render_duration_seconds.clone(),
        );

        Self {
            request_render_duration_seconds,
        }
    }

    /// Record the time one request took to render into prompt token IDs,
    /// labeled with the primary served model name.
    pub fn observe_request_render_duration(
        &self,
        model_name: &str,
        request_type: RenderRequestType,
        duration: Duration,
    ) {
        let labels = RenderLabels {
            model_name: model_name.to_string(),
            request_type,
        };
        self.request_render_duration_seconds
            .get_or_create(&labels)
            .observe(duration.as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use expect_test::expect;
    use prometheus_client::encoding::text::encode;
    use prometheus_client::registry::Registry;

    use super::{RenderRequestType, RendererMetrics};

    #[test]
    fn request_render_duration_matches_python_exposition() {
        let mut registry = Registry::default();
        let metrics = RendererMetrics::register(&mut registry);
        metrics.observe_request_render_duration(
            "served-model",
            RenderRequestType::Chat,
            Duration::from_millis(3),
        );

        let mut rendered = String::new();
        encode(&mut rendered, &registry).unwrap();
        expect![[r##"
            # HELP vllm:request_render_duration_seconds Time spent rendering a request, including chat templating, tokenization, and multimodal preprocessing.
            # TYPE vllm:request_render_duration_seconds histogram
            vllm:request_render_duration_seconds_sum{model_name="served-model",request_type="chat"} 0.003
            vllm:request_render_duration_seconds_count{model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.0001",model_name="served-model",request_type="chat"} 0
            vllm:request_render_duration_seconds_bucket{le="0.00025",model_name="served-model",request_type="chat"} 0
            vllm:request_render_duration_seconds_bucket{le="0.0005",model_name="served-model",request_type="chat"} 0
            vllm:request_render_duration_seconds_bucket{le="0.001",model_name="served-model",request_type="chat"} 0
            vllm:request_render_duration_seconds_bucket{le="0.0025",model_name="served-model",request_type="chat"} 0
            vllm:request_render_duration_seconds_bucket{le="0.005",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.01",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.025",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.05",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.1",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.25",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="0.5",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="1.0",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="2.5",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="5.0",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="10.0",model_name="served-model",request_type="chat"} 1
            vllm:request_render_duration_seconds_bucket{le="+Inf",model_name="served-model",request_type="chat"} 1
            # EOF
        "##]].assert_eq(&rendered);
    }
}
