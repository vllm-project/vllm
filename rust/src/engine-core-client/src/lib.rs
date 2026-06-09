use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use zeromq::{PubSubChannel, PubSubReceiver, PubSubSender, SocketType, ZmqError};

use crate::{
    api::{
        CompletionRequest, CompletionResponse, EngineError, EngineRequest, EngineResponse,
        HealthResponse, ModelInfoResponse,
    },
    config::{EngineConfig, ParallelConfig},
    executor::{Executor, ExecutorClass},
    stats::StatsCollector,
};

// Re-export types from the API module.
pub use crate::api::*;
pub use crate::config::*;
pub use crate::executor::*;
pub use crate::stats::*;

/// A client for the vLLM engine's multi-process backend.
///
/// This client communicates with the engine's backend processes using ZeroMQ
/// sockets. It handles the setup and teardown of these sockets, as well as
/// sending requests and receiving responses.
#[derive(Debug)]
pub struct EngineCoreClient {
    /// Configuration for the vLLM engine.
    pub vllm_config: EngineConfig,
    /// The class of executor to use for the engine.
    pub executor_class: ExecutorClass,
    /// Whether to log statistics.
    pub log_stats: bool,
    /// The ZeroMQ socket for sending requests to the engine.
    pub request_socket: PubSubSender,
    /// The ZeroMQ socket for receiving responses from the engine.
    pub response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending health checks to the engine.
    pub health_socket: PubSubSender,
    /// The ZeroMQ socket for receiving health check responses from the engine.
    pub health_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for publishing frontend statistics.
    pub frontend_stats_publish_socket: PubSubSender,
    /// The ZeroMQ socket for subscribing to frontend statistics.
    pub frontend_stats_subscribe_socket: PubSubReceiver,
    /// The ZeroMQ socket for publishing engine statistics.
    pub engine_stats_publish_socket: PubSubSender,
    /// The ZeroMQ socket for subscribing to engine statistics.
    pub engine_stats_subscribe_socket: PubSubReceiver,
    /// The ZeroMQ socket for publishing coordinator messages.
    pub coordinator_publish_socket: PubSubSender,
    /// The ZeroMQ socket for subscribing to coordinator messages.
    pub coordinator_subscribe_socket: PubSubReceiver,
    /// The ZeroMQ socket for publishing coordinator responses.
    pub coordinator_response_publish_socket: PubSubSender,
    /// The ZeroMQ socket for subscribing to coordinator responses.
    pub coordinator_response_subscribe_socket: PubSubReceiver,
    /// The ZeroMQ socket for receiving output from the engine.
    pub output_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending input to the engine.
    pub input_socket: PubSubSender,
    /// The ZeroMQ socket for receiving input from the engine.
    pub input_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending output to the engine.
    pub output_response_socket: PubSubSender,
    /// The ZeroMQ socket for receiving output responses from the engine.
    pub output_response_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending completion requests to the engine.
    pub completion_socket: PubSubSender,
    /// The ZeroMQ socket for receiving completion responses from the engine.
    pub completion_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending model info requests to the engine.
    pub model_info_socket: PubSubSender,
    /// The ZeroMQ socket for receiving model info responses from the engine.
    pub model_info_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending shutdown requests to the engine.
    pub shutdown_socket: PubSubSender,
    /// The ZeroMQ socket for receiving shutdown responses from the engine.
    pub shutdown_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending model load requests to the engine.
    pub model_load_socket: PubSubSender,
    /// The ZeroMQ socket for receiving model load responses from the engine.
    pub model_load_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending model unload requests to the engine.
    pub model_unload_socket: PubSubSender,
    /// The ZeroMQ socket for receiving model unload responses from the engine.
    pub model_unload_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending tokenization requests to the engine.
    pub tokenize_socket: PubSubSender,
    /// The ZeroMQ socket for receiving tokenization responses from the engine.
    pub tokenize_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending detokenization requests to the engine.
    pub detokenize_socket: PubSubSender,
    /// The ZeroMQ socket for receiving detokenization responses from the engine.
    pub detokenize_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output to the engine.
    pub stream_output_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output from the engine.
    pub stream_output_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input to the engine.
    pub stream_input_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input from the engine.
    pub stream_input_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v2: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v2: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v3: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v3: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v4: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v4: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v5: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v5: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v5: PubSubSender,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v5: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v5: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v5: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v5: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v6: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v6: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_request_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming completion responses from the engine.
    pub stream_completion_response_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming output requests to the engine.
    pub stream_output_request_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming output responses from the engine.
    pub stream_output_response_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming input requests to the engine.
    pub stream_input_request_socket_v7: PubSubSender,
    /// The ZeroMQ socket for receiving streaming input responses from the engine.
    pub stream_input_response_response_socket_v7: PubSubReceiver,
    /// The ZeroMQ socket for sending streaming completion requests to the engine.
    pub stream_completion_socket_v8
