use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Frontend-owned ZMQ addresses that are sent to the engine during startup
/// handshake initialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeAddresses {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub coordinator_input: Option<String>,
    pub coordinator_output: Option<String>,
    pub frontend_stats_publish_address: Option<String>,
}

/// Startup handshake payload sent from the frontend to initialize an engine
/// after receiving `HELLO`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeInitMessage {
    pub addresses: HandshakeAddresses,
    pub parallel_config: BTreeMap<String, u32>,
}
