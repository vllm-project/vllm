// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Persistence for development-only EngineCore reattach sessions.
//!
//! A session captures the frontend-owned ZMQ endpoints and the engine metadata
//! produced by the one-shot startup handshake. A replacement frontend can bind
//! the same endpoints and reconstruct its client without asking EngineCore to
//! repeat that handshake.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::transport::{ConnectedEngine, ConnectedTransport, EngineId};

/// Serializable transport state needed to reconnect a frontend to EngineCore.
#[derive(Debug, Serialize, Deserialize)]
struct EngineSession {
    input_address: String,
    output_address: String,
    engines: Vec<SessionEngine>,
}

/// Serializable subset of one [`ConnectedEngine`].
#[derive(Debug, Serialize, Deserialize)]
struct SessionEngine {
    engine_id: Vec<u8>,
    ready_response: EngineCoreReadyResponse,
}

impl EngineSession {
    /// Build a validated session snapshot from a live connected transport.
    fn from_connected(path: &Path, connected: &ConnectedTransport) -> Result<Self> {
        if connected.coordinator.is_some() {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: "reattach sessions do not support an in-process coordinator".to_string(),
            });
        }
        if connected.engines.len() != 1 {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: format!(
                    "reattach sessions currently require exactly one engine, found {}",
                    connected.engines.len()
                ),
            });
        }

        Ok(Self {
            input_address: connected.input_address.clone(),
            output_address: connected.output_address.clone(),
            engines: connected
                .engines
                .iter()
                .map(|engine| SessionEngine {
                    engine_id: engine.engine_id.to_vec(),
                    ready_response: engine.ready_response.clone(),
                })
                .collect(),
        })
    }

    /// Validate a decoded session and reconstruct its transport metadata.
    fn into_parts(self, path: &Path) -> Result<(String, String, Vec<ConnectedEngine>)> {
        if self.engines.len() != 1 {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: format!(
                    "reattach sessions currently require exactly one engine, found {}",
                    self.engines.len()
                ),
            });
        }

        Ok((
            self.input_address,
            self.output_address,
            self.engines
                .into_iter()
                .map(|engine| ConnectedEngine {
                    engine_id: EngineId::from(engine.engine_id),
                    ready_response: engine.ready_response,
                })
                .collect(),
        ))
    }
}

/// Atomically write a connected transport as a reattach session.
///
/// The temporary file is created in the destination directory so
/// [`NamedTempFile::persist`] can replace the session with a same-filesystem
/// rename. Dropping the temporary file cleans it up on serialization or persist
/// failure.
pub(crate) fn write(path: &Path, connected: &ConnectedTransport) -> Result<()> {
    let session = EngineSession::from_connected(path, connected)?;
    if path.file_name().is_none() {
        return Err(Error::EngineSession {
            path: path.to_path_buf(),
            message: "session path must name a file".to_string(),
        });
    }
    let directory = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary_file =
        NamedTempFile::new_in(directory).map_err(|error| Error::EngineSession {
            path: path.to_path_buf(),
            message: error.to_report_string(),
        })?;
    serde_json::to_writer_pretty(temporary_file.as_file_mut(), &session).map_err(|error| {
        Error::EngineSession {
            path: path.to_path_buf(),
            message: error.to_report_string(),
        }
    })?;
    temporary_file.persist(path).map_err(|error| Error::EngineSession {
        path: path.to_path_buf(),
        message: error.error.to_report_string(),
    })?;
    Ok(())
}

/// Read and validate a reattach session from disk.
///
/// Returns the saved input/output endpoints and reconstructed engine metadata
/// used by the reattach transport.
pub(crate) fn read(path: &Path) -> Result<(String, String, Vec<ConnectedEngine>)> {
    let bytes = fs::read(path).map_err(|error| Error::EngineSession {
        path: path.to_path_buf(),
        message: error.to_report_string(),
    })?;
    let session: EngineSession =
        serde_json::from_slice(&bytes).map_err(|error| Error::EngineSession {
            path: path.to_path_buf(),
            message: error.to_report_string(),
        })?;
    session.into_parts(path)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::mock_engine::default_ready_response;
    use crate::test_utils::setup_bootstrapped_mock_engine;
    use crate::transport::connect_reattach;

    #[tokio::test]
    async fn session_round_trip_restores_transport_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine-session.json");
        let input_address = format!("ipc://{}", dir.path().join("input.sock").display());
        let output_address = format!("ipc://{}", dir.path().join("output.sock").display());
        let connect_path = path.clone();
        let connect_input = input_address.clone();
        let connect_output = output_address.clone();
        let connected_task = tokio::spawn(async move {
            connect_reattach(
                &connect_path,
                &connect_input,
                &connect_output,
                vec![ConnectedEngine {
                    engine_id: EngineId::from_engine_index(0),
                    ready_response: default_ready_response(),
                }],
                Duration::from_secs(2),
            )
            .await
            .unwrap()
        });
        let (_dealer, _push) = setup_bootstrapped_mock_engine(
            input_address.clone(),
            output_address.clone(),
            EngineId::from_engine_index(0),
        )
        .await;
        let connected = connected_task.await.unwrap();

        write(&path, &connected).unwrap();
        let (actual_input, actual_output, engines) = read(&path).unwrap();

        assert_eq!(actual_input, input_address);
        assert_eq!(actual_output, output_address);
        assert_eq!(engines.len(), 1);
        assert_eq!(engines[0].engine_id.engine_index(), Some(0));
        assert_eq!(
            engines[0].ready_response.max_model_len,
            default_ready_response().max_model_len
        );
    }
}
