// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Selection of the gRPC services mounted on the frontend's gRPC port.

use std::fmt;
use std::str::FromStr;

use anyhow::{Result, bail};
use bitflags::bitflags;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use tonic::server::NamedService;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use crate::grpc::pb;
use crate::grpc::{
    ControlGrpcService, InferenceGrpcService, KvTransferGrpcService, RlControlGrpcService,
    kv_event_source,
};

bitflags! {
    /// gRPC services the frontend serves on `--grpc-port`. Services outside the
    /// set are not mounted and answer `Unimplemented`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct GrpcServices: u8 {
        const INFERENCE = 1 << 0;
        const CONTROL = 1 << 1;
        const KV_TRANSFER = 1 << 2;
        const RL_CONTROL = 1 << 3;
    }
}

/// One gRPC service in every representation the frontend needs: the flag, the
/// `--grpc-services` token, the gRPC service name the health reporter uses, and
/// the value `ServerInfo.services` carries.
pub(crate) struct GrpcServiceEntry {
    pub(crate) flag: GrpcServices,
    pub(crate) token: &'static str,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) service_name: &'static str,
    pub(crate) proto: pb::GrpcService,
}

/// Every service, in the order [`GrpcServices`] displays and reports them.
pub(crate) const GRPC_SERVICE_TABLE: [GrpcServiceEntry; 4] = [
    GrpcServiceEntry {
        flag: GrpcServices::INFERENCE,
        token: "inference",
        aliases: &[],
        service_name: <InferenceGrpcService as NamedService>::NAME,
        proto: pb::GrpcService::Inference,
    },
    GrpcServiceEntry {
        flag: GrpcServices::CONTROL,
        token: "control",
        aliases: &[],
        service_name: <ControlGrpcService as NamedService>::NAME,
        proto: pb::GrpcService::Control,
    },
    GrpcServiceEntry {
        flag: GrpcServices::KV_TRANSFER,
        token: "kv-transfer",
        aliases: &[],
        service_name: <KvTransferGrpcService as NamedService>::NAME,
        proto: pb::GrpcService::KvTransfer,
    },
    GrpcServiceEntry {
        flag: GrpcServices::RL_CONTROL,
        token: "rl-control",
        aliases: &["rl"],
        service_name: <RlControlGrpcService as NamedService>::NAME,
        proto: pb::GrpcService::RlControl,
    },
];

/// The entry for a single service flag.
pub(crate) fn service_entry(flag: GrpcServices) -> Option<&'static GrpcServiceEntry> {
    GRPC_SERVICE_TABLE.iter().find(|entry| entry.flag == flag)
}

/// The entries a set mounts, in table order.
pub(crate) fn mounted_entries(
    services: GrpcServices,
) -> impl Iterator<Item = &'static GrpcServiceEntry> {
    GRPC_SERVICE_TABLE.iter().filter(move |entry| services.contains(entry.flag))
}

const ALL_TOKEN: &str = "all";
const CONFIGURED_TOKEN: &str = "configured";

/// Rejection of a `--grpc-services` value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GrpcServicesParseError {
    #[error(
        "unknown gRPC service `{0}`; expected `all`, `configured`, or a comma-separated list of \
         inference, control, kv-transfer, rl-control"
    )]
    UnknownService(String),
    #[error(
        "expected `all`, `configured`, or a comma-separated list of gRPC services, got an empty \
         value"
    )]
    Empty,
    #[error("`{0}` selects the gRPC services on its own and cannot be listed with others")]
    SelectionWithOthers(String),
}

impl GrpcServices {
    /// Services derived from what the engines report they are configured for.
    /// Inference and control are always served.
    pub fn configured(ready: &[&EngineCoreReadyResponse]) -> Self {
        let mut services = Self::INFERENCE | Self::CONTROL;
        services.set(Self::KV_TRANSFER, kv_transfer_configured(ready));
        services.set(Self::RL_CONTROL, rl_configured(ready));
        services
    }
}

impl fmt::Display for GrpcServices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, entry) in mounted_entries(*self).enumerate() {
            if position > 0 {
                f.write_str(",")?;
            }
            f.write_str(entry.token)?;
        }
        Ok(())
    }
}

impl FromStr for GrpcServices {
    type Err = GrpcServicesParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut services = Self::empty();
        for token in value.split(',') {
            let token = token.trim();
            if token.eq_ignore_ascii_case(ALL_TOKEN) || token.eq_ignore_ascii_case(CONFIGURED_TOKEN)
            {
                return Err(GrpcServicesParseError::SelectionWithOthers(
                    token.to_ascii_lowercase(),
                ));
            }
            let entry = GRPC_SERVICE_TABLE
                .iter()
                .find(|entry| {
                    token.eq_ignore_ascii_case(entry.token)
                        || entry.aliases.iter().any(|alias| token.eq_ignore_ascii_case(alias))
                })
                .ok_or_else(|| GrpcServicesParseError::UnknownService(token.to_string()))?;
            services |= entry.flag;
        }
        Ok(services)
    }
}

/// What `--grpc-services` asked for, before the engines report what they
/// support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpcServiceSelection {
    /// Mount every service, whatever the engines are configured for.
    #[default]
    All,
    /// Derive the set from the engine ready responses.
    Configured,
    /// Serve exactly these services.
    Explicit(GrpcServices),
}

impl GrpcServiceSelection {
    /// Resolve the selection against the engine ready responses, rejecting an
    /// explicit service the engines are not configured for.
    pub fn resolve(self, ready: &[&EngineCoreReadyResponse]) -> Result<GrpcServices> {
        let services = match self {
            Self::All => return Ok(GrpcServices::all()),
            Self::Configured => return Ok(GrpcServices::configured(ready)),
            Self::Explicit(services) => services,
        };
        if services.is_empty() {
            bail!("--grpc-services must name at least one service");
        }
        if services.contains(GrpcServices::KV_TRANSFER) && !kv_transfer_configured(ready) {
            bail!(
                "--grpc-services requested kv-transfer, but no engine has a KV connector \
                 (--kv-transfer-config) or ZMQ KV cache events (--kv-events-config) configured"
            );
        }
        if services.contains(GrpcServices::RL_CONTROL) && !rl_configured(ready) {
            bail!(
                "--grpc-services requested rl-control, but the engines do not all have weight \
                 transfer (--weight-transfer-config) or sleep mode (--enable-sleep-mode) configured"
            );
        }
        Ok(services)
    }
}

impl fmt::Display for GrpcServiceSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str(ALL_TOKEN),
            Self::Configured => f.write_str(CONFIGURED_TOKEN),
            Self::Explicit(services) => write!(f, "{services}"),
        }
    }
}

impl FromStr for GrpcServiceSelection {
    type Err = GrpcServicesParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(GrpcServicesParseError::Empty);
        }
        if trimmed.eq_ignore_ascii_case(ALL_TOKEN) {
            return Ok(Self::All);
        }
        if trimmed.eq_ignore_ascii_case(CONFIGURED_TOKEN) {
            return Ok(Self::Configured);
        }
        Ok(Self::Explicit(trimmed.parse()?))
    }
}

impl Serialize for GrpcServiceSelection {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for GrpcServiceSelection {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(D::Error::custom)
    }
}

/// True when any engine gives the KV transfer RPCs something to report.
fn kv_transfer_configured(ready: &[&EngineCoreReadyResponse]) -> bool {
    ready
        .iter()
        .any(|ready| ready.kv_transfer_info.is_some() || kv_event_source(ready).is_some())
}

/// True when every engine can serve the RL RPCs, which drive them all at once.
fn rl_configured(ready: &[&EngineCoreReadyResponse]) -> bool {
    weight_transfer_backend(ready).is_some() || sleep_mode_enabled(ready)
}

/// The weight transfer backend every engine reports, or `None` when they
/// disagree or any of them has none.
pub(crate) fn weight_transfer_backend<'a>(
    ready: &[&'a EngineCoreReadyResponse],
) -> Option<&'a str> {
    let backend = ready.first()?.weight_transfer_backend.as_deref()?;
    ready
        .iter()
        .all(|ready| ready.weight_transfer_backend.as_deref() == Some(backend))
        .then_some(backend)
}

pub(crate) fn sleep_mode_enabled(ready: &[&EngineCoreReadyResponse]) -> bool {
    !ready.is_empty() && ready.iter().all(|ready| ready.enable_sleep_mode)
}

pub(crate) fn draft_weight_updates_enabled(ready: &[&EngineCoreReadyResponse]) -> bool {
    !ready.is_empty() && ready.iter().all(|ready| ready.supports_draft_weight_updates)
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::mock_engine::default_ready_response;
    use vllm_engine_core_client::protocol::handshake::{KvEventsConfig, KvTransferInfo};

    use super::*;

    fn kv_events(enabled: bool) -> KvEventsConfig {
        kv_events_with_publisher(enabled, "zmq")
    }

    fn kv_events_with_publisher(enabled: bool, publisher: &str) -> KvEventsConfig {
        KvEventsConfig {
            enable_kv_cache_events: enabled,
            publisher: publisher.to_string(),
            endpoint: "tcp://127.0.0.1:5557".to_string(),
            replay_endpoint: None,
            buffer_steps: 1,
            hwm: 1,
            max_queue_size: 1,
            topic: String::new(),
        }
    }

    fn kv_transfer_info() -> KvTransferInfo {
        KvTransferInfo {
            engine_id: "prefill-0".to_string(),
            kv_connector: "NixlConnector".to_string(),
            kv_role: "kv_producer".to_string(),
        }
    }

    #[test]
    fn selection_round_trips_through_display_and_from_str() {
        let cases = [
            GrpcServiceSelection::All,
            GrpcServiceSelection::Configured,
            GrpcServiceSelection::Explicit(GrpcServices::all()),
            GrpcServiceSelection::Explicit(GrpcServices::INFERENCE),
            GrpcServiceSelection::Explicit(GrpcServices::INFERENCE | GrpcServices::CONTROL),
            GrpcServiceSelection::Explicit(GrpcServices::KV_TRANSFER | GrpcServices::RL_CONTROL),
        ];
        for selection in cases {
            let rendered = selection.to_string();
            assert_eq!(
                rendered.parse::<GrpcServiceSelection>(),
                Ok(selection),
                "round trip failed for {rendered}"
            );
        }
    }

    #[test]
    fn selection_renders_stable_tokens() {
        assert_eq!(GrpcServiceSelection::All.to_string(), "all");
        assert_eq!(GrpcServiceSelection::Configured.to_string(), "configured");
        assert_eq!(
            GrpcServiceSelection::Explicit(GrpcServices::all()).to_string(),
            "inference,control,kv-transfer,rl-control"
        );
        assert_eq!(
            GrpcServiceSelection::Explicit(GrpcServices::KV_TRANSFER).to_string(),
            "kv-transfer"
        );
    }

    #[test]
    fn selection_parses_aliases_and_whitespace() {
        assert_eq!(
            " ALL ".parse::<GrpcServiceSelection>(),
            Ok(GrpcServiceSelection::All)
        );
        assert_eq!(
            "configured".parse::<GrpcServiceSelection>(),
            Ok(GrpcServiceSelection::Configured)
        );
        assert_eq!(
            "rl-control, kv-transfer".parse::<GrpcServiceSelection>(),
            Ok(GrpcServiceSelection::Explicit(
                GrpcServices::RL_CONTROL | GrpcServices::KV_TRANSFER
            ))
        );
        assert_eq!(
            "rl".parse::<GrpcServiceSelection>(),
            Ok(GrpcServiceSelection::Explicit(GrpcServices::RL_CONTROL)),
            "`rl` stays an accepted alias for `rl-control`"
        );
    }

    #[test]
    fn service_table_matches_the_wire_names() {
        let names: Vec<&str> = GRPC_SERVICE_TABLE.iter().map(|entry| entry.service_name).collect();
        assert_eq!(
            names,
            vec![
                "vllm.Inference",
                "vllm.Control",
                "vllm.KvTransfer",
                "vllm.RlControl"
            ]
        );
        for entry in &GRPC_SERVICE_TABLE {
            let proto_name = format!("{:?}", entry.proto);
            let service_name = entry.service_name.trim_start_matches("vllm.");
            assert_eq!(
                proto_name, service_name,
                "`ServerInfo.services` value does not name its service"
            );
        }
    }

    #[test]
    fn selection_rejects_bad_values() {
        assert_eq!(
            "".parse::<GrpcServiceSelection>(),
            Err(GrpcServicesParseError::Empty)
        );
        assert_eq!(
            "control,all".parse::<GrpcServiceSelection>(),
            Err(GrpcServicesParseError::SelectionWithOthers(
                "all".to_string()
            ))
        );
        assert_eq!(
            "control,configured".parse::<GrpcServiceSelection>(),
            Err(GrpcServicesParseError::SelectionWithOthers(
                "configured".to_string()
            ))
        );
        assert_eq!(
            "auto".parse::<GrpcServiceSelection>(),
            Err(GrpcServicesParseError::UnknownService("auto".to_string()))
        );
        assert_eq!(
            "control,kv_transfer".parse::<GrpcServiceSelection>(),
            Err(GrpcServicesParseError::UnknownService(
                "kv_transfer".to_string()
            ))
        );
    }

    #[test]
    fn serde_round_trips_as_a_string() {
        let selection =
            GrpcServiceSelection::Explicit(GrpcServices::CONTROL | GrpcServices::RL_CONTROL);
        let json = serde_json::to_string(&selection).expect("serialize selection");
        assert_eq!(json, r#""control,rl-control""#);
        assert_eq!(
            serde_json::from_str::<GrpcServiceSelection>(&json).expect("deserialize selection"),
            selection
        );
        assert_eq!(
            serde_json::from_str::<GrpcServiceSelection>(r#""all""#).expect("deserialize all"),
            GrpcServiceSelection::All
        );
        assert_eq!(
            serde_json::from_str::<GrpcServiceSelection>(r#""configured""#)
                .expect("deserialize configured"),
            GrpcServiceSelection::Configured
        );
    }

    #[test]
    fn configured_derives_services_from_ready_responses() {
        let base = GrpcServices::INFERENCE | GrpcServices::CONTROL;
        let cases: [(&str, Vec<EngineCoreReadyResponse>, GrpcServices); 10] = [
            ("plain engine", vec![default_ready_response()], base),
            (
                "kv connector",
                vec![EngineCoreReadyResponse {
                    kv_transfer_info: Some(kv_transfer_info()),
                    ..default_ready_response()
                }],
                base | GrpcServices::KV_TRANSFER,
            ),
            (
                "kv events enabled",
                vec![EngineCoreReadyResponse {
                    kv_events_config: Some(kv_events(true)),
                    ..default_ready_response()
                }],
                base | GrpcServices::KV_TRANSFER,
            ),
            (
                "kv events present but disabled",
                vec![EngineCoreReadyResponse {
                    kv_events_config: Some(kv_events(false)),
                    ..default_ready_response()
                }],
                base,
            ),
            (
                "kv events on a publisher the service cannot report",
                vec![EngineCoreReadyResponse {
                    kv_events_config: Some(kv_events_with_publisher(true, "kafka")),
                    ..default_ready_response()
                }],
                base,
            ),
            (
                "weight transfer",
                vec![EngineCoreReadyResponse {
                    weight_transfer_backend: Some("nccl".to_string()),
                    ..default_ready_response()
                }],
                base | GrpcServices::RL_CONTROL,
            ),
            (
                "sleep mode",
                vec![EngineCoreReadyResponse {
                    enable_sleep_mode: true,
                    ..default_ready_response()
                }],
                base | GrpcServices::RL_CONTROL,
            ),
            (
                "one engine of two has a KV connector",
                vec![
                    EngineCoreReadyResponse {
                        kv_transfer_info: Some(kv_transfer_info()),
                        ..default_ready_response()
                    },
                    default_ready_response(),
                ],
                base | GrpcServices::KV_TRANSFER,
            ),
            (
                "one engine of two enables sleep mode",
                vec![
                    EngineCoreReadyResponse {
                        enable_sleep_mode: true,
                        ..default_ready_response()
                    },
                    default_ready_response(),
                ],
                base,
            ),
            (
                "engines disagree on the weight transfer backend",
                vec![
                    EngineCoreReadyResponse {
                        weight_transfer_backend: Some("nccl".to_string()),
                        ..default_ready_response()
                    },
                    EngineCoreReadyResponse {
                        weight_transfer_backend: Some("ipc".to_string()),
                        ..default_ready_response()
                    },
                ],
                base,
            ),
        ];

        for (label, responses, expected) in cases {
            let ready: Vec<&EngineCoreReadyResponse> = responses.iter().collect();
            assert_eq!(
                GrpcServices::configured(&ready),
                expected,
                "configured set for {label}"
            );
            assert_eq!(
                GrpcServiceSelection::Configured
                    .resolve(&ready)
                    .expect("configured never fails"),
                expected,
                "resolved configured set for {label}"
            );
            assert_eq!(
                GrpcServiceSelection::All.resolve(&ready).expect("all never fails"),
                GrpcServices::all(),
                "all mounts everything for {label}"
            );
        }
    }

    #[test]
    fn explicit_selection_requires_engine_support() {
        let plain = [default_ready_response()];
        let ready: Vec<&EngineCoreReadyResponse> = plain.iter().collect();

        let kv = GrpcServiceSelection::Explicit(GrpcServices::CONTROL | GrpcServices::KV_TRANSFER);
        let error = kv.resolve(&ready).expect_err("kv-transfer is unsupported");
        assert!(
            error.to_string().contains("kv-transfer"),
            "unexpected error: {error}"
        );

        let rl = GrpcServiceSelection::Explicit(GrpcServices::RL_CONTROL);
        let error = rl.resolve(&ready).expect_err("rl-control is unsupported");
        assert!(
            error.to_string().contains("rl-control"),
            "unexpected error: {error}"
        );

        let inference = GrpcServiceSelection::Explicit(GrpcServices::INFERENCE);
        assert_eq!(
            inference.resolve(&ready).expect("inference is always available"),
            GrpcServices::INFERENCE
        );

        assert_eq!(
            GrpcServiceSelection::All.resolve(&ready).expect("all skips validation"),
            GrpcServices::all()
        );
    }

    #[test]
    fn explicit_selection_requires_every_engine_to_support_rl() {
        let mixed = [
            EngineCoreReadyResponse {
                enable_sleep_mode: true,
                ..default_ready_response()
            },
            default_ready_response(),
        ];
        let ready: Vec<&EngineCoreReadyResponse> = mixed.iter().collect();

        let error = GrpcServiceSelection::Explicit(GrpcServices::RL_CONTROL)
            .resolve(&ready)
            .expect_err("a mixed fleet cannot serve the RL RPCs");
        assert!(
            error.to_string().contains("rl-control"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn explicit_selection_rejects_an_empty_set() {
        let plain = [default_ready_response()];
        let ready: Vec<&EngineCoreReadyResponse> = plain.iter().collect();

        let error = GrpcServiceSelection::Explicit(GrpcServices::empty())
            .resolve(&ready)
            .expect_err("an empty set mounts nothing");
        assert!(
            error.to_string().contains("at least one service"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn explicit_selection_keeps_exactly_what_was_asked_for() {
        let configured = [EngineCoreReadyResponse {
            kv_transfer_info: Some(kv_transfer_info()),
            enable_sleep_mode: true,
            ..default_ready_response()
        }];
        let ready: Vec<&EngineCoreReadyResponse> = configured.iter().collect();

        assert_eq!(
            GrpcServiceSelection::Explicit(GrpcServices::KV_TRANSFER)
                .resolve(&ready)
                .expect("kv-transfer is configured"),
            GrpcServices::KV_TRANSFER
        );
        assert_eq!(
            GrpcServiceSelection::Configured
                .resolve(&ready)
                .expect("configured never fails"),
            GrpcServices::all()
        );
    }
}
