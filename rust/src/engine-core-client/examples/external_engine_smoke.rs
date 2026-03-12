use std::collections::BTreeMap;
use std::io::{Error as IoError, ErrorKind};
use std::net::TcpListener;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use clap::Parser;
use serde::Serialize;
use tokio::time::timeout;
use vllm_engine_core_client::{
    EngineCoreClient, EngineCoreOutput, EngineCoreRequest, ReadyMessage, RequestOutputKind,
    SamplingParams, ZmqEngineCoreClient, ZmqEngineCoreClientConfig,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSocket, ZmqMessage};

const MODEL_NAME: &str = "Qwen/Qwen3-0.6B";
const PROMPT_TOKEN_IDS: &[u32] = &[20841, 448, 6896, 25, 23811];

#[derive(Debug, Parser)]
#[command(about = "Smoke-test a Rust EngineCoreClient against an external vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long)]
    engine_identity_hex: Option<String>,
    #[arg(long, default_value_t = 0)]
    client_index: u32,
    #[arg(long, default_value_t = 30)]
    ready_timeout_secs: u64,
    #[arg(long, default_value_t = 120)]
    output_timeout_secs: u64,
    #[arg(long, default_value_t = 1)]
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct HandshakeAddresses {
    inputs: Vec<String>,
    outputs: Vec<String>,
    coordinator_input: Option<String>,
    coordinator_output: Option<String>,
    frontend_stats_publish_address: Option<String>,
}

#[derive(Debug, Serialize)]
struct HandshakeInitMessage {
    addresses: HandshakeAddresses,
    parallel_config: BTreeMap<String, u32>,
}

fn invalid_input(message: String) -> IoError {
    IoError::new(ErrorKind::InvalidInput, message)
}

fn parse_hex_bytes(hex: &str) -> Result<Vec<u8>, IoError> {
    if !hex.len().is_multiple_of(2) {
        return Err(invalid_input(format!(
            "hex string must have even length: {hex}"
        )));
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let chars: Vec<char> = hex.chars().collect();
    for pair in chars.chunks_exact(2) {
        let hi = pair[0].to_digit(16).ok_or_else(|| {
            invalid_input(format!(
                "invalid hex digit in engine identity: {}",
                pair[0]
            ))
        })?;
        let lo = pair[1].to_digit(16).ok_or_else(|| {
            invalid_input(format!(
                "invalid hex digit in engine identity: {}",
                pair[1]
            ))
        })?;
        bytes.push(((hi << 4) | lo) as u8);
    }

    Ok(bytes)
}

fn unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

fn unique_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_nanos();
    format!("rust-engine-core-smoke-{nanos}")
}

fn allocate_tcp_address(host: &str) -> Result<String, IoError> {
    let listener = TcpListener::bind((host, 0))
        .map_err(|error| invalid_input(format!("failed to allocate local port on {host}: {error}")))?;
    let port = listener
        .local_addr()
        .map_err(|error| invalid_input(format!("failed to read local addr: {error}")))?
        .port();
    drop(listener);
    Ok(format!("tcp://{host}:{port}"))
}

fn build_request(request_id: String, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id,
        prompt_token_ids: Some(PROMPT_TOKEN_IDS.to_vec()),
        mm_features: None,
        sampling_params: Some(SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: Some(max_tokens),
            output_kind: RequestOutputKind::FinalOnly,
            ..Default::default()
        }),
        pooling_params: None,
        arrival_time: unix_timestamp_secs(),
        lora_request: None,
        cache_salt: None,
        data_parallel_rank: None,
        prompt_embeds: None,
        client_index: 0,
        current_wave: 0,
        priority: 0,
        trace_headers: None,
        resumable: false,
        external_req_id: None,
        reasoning_ended: None,
    }
}

fn decode_router_message(
    message: ZmqMessage,
    expected_identity: Option<&[u8]>,
) -> Result<(Vec<u8>, ReadyMessage), Box<dyn std::error::Error>> {
    if message.len() != 2 {
        return Err(Box::new(invalid_input(format!(
            "expected 2 handshake frames, got {}",
            message.len()
        ))));
    }

    let frames = message.into_vec();
    let identity = frames[0].to_vec();
    if let Some(expected_identity) = expected_identity {
        if identity != expected_identity {
            return Err(Box::new(invalid_input(format!(
                "unexpected engine identity in handshake: expected {}, got {}",
                hex_string(expected_identity),
                hex_string(&identity)
            ))));
        }
    }

    let payload: ReadyMessage = rmp_serde::from_slice(&frames[1])?;
    Ok((identity, payload))
}

async fn perform_startup_handshake(
    handshake_address: &str,
    input_address: &str,
    output_address: &str,
    ready_timeout: Duration,
    expected_identity: Option<&[u8]>,
) -> Result<(Vec<u8>, ReadyMessage), Box<dyn std::error::Error>> {
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    let hello = timeout(ready_timeout, handshake_socket.recv()).await??;
    let (identity, hello_message) = decode_router_message(hello, expected_identity)?;
    if hello_message.status.as_deref() != Some("HELLO") {
        return Err(Box::new(invalid_input(format!(
            "expected HELLO during startup handshake, got {:?}",
            hello_message.status
        ))));
    }

    let init = HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![input_address.to_string()],
            outputs: vec![output_address.to_string()],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        },
        parallel_config: BTreeMap::new(),
    };
    let init_bytes = rmp_serde::to_vec_named(&init)?;
    let init_message = ZmqMessage::try_from(vec![
        Bytes::from(identity.clone()),
        Bytes::from(init_bytes),
    ])
    .expect("handshake router message must contain identity and payload");
    handshake_socket.send(init_message).await?;

    let ready = timeout(ready_timeout, handshake_socket.recv()).await??;
    let (_, ready_message) = decode_router_message(ready, Some(&identity))?;
    if ready_message.status.as_deref() != Some("READY") {
        return Err(Box::new(invalid_input(format!(
            "expected READY during startup handshake, got {:?}",
            ready_message.status
        ))));
    }

    Ok((identity, ready_message))
}

async fn wait_for_request_output(
    client: &mut ZmqEngineCoreClient,
    request_id: &str,
) -> vllm_engine_core_client::Result<EngineCoreOutput> {
    loop {
        let batch = client.next_output().await?;
        if let Some(output) = batch
            .outputs
            .into_iter()
            .find(|output| output.request_id == request_id)
        {
            return Ok(output);
        }
    }
}

async fn wait_for_timeout(
    client: &mut ZmqEngineCoreClient,
    request_id: &str,
    output_timeout: Duration,
) -> Result<EngineCoreOutput, Box<dyn std::error::Error>> {
    Ok(timeout(output_timeout, wait_for_request_output(client, request_id))
        .await
        .map_err(|_| {
            invalid_input(format!(
                "timed out waiting {:?} for output of request {request_id}",
                output_timeout
            ))
        })??)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let expected_identity = args
        .engine_identity_hex
        .as_deref()
        .map(parse_hex_bytes)
        .transpose()?;
    let request_id = unique_request_id();
    let input_address = allocate_tcp_address(&args.host)?;
    let output_address = allocate_tcp_address(&args.host)?;

    let mut input_socket = RouterSocket::new();
    input_socket.bind(&input_address).await?;

    let mut output_socket = PullSocket::new();
    output_socket.bind(&output_address).await?;

    let (engine_identity, handshake_ready_message) = perform_startup_handshake(
        &args.handshake_address,
        &input_address,
        &output_address,
        ready_timeout,
        expected_identity.as_deref(),
    )
    .await?;

    let mut client = ZmqEngineCoreClient::connect_with_sockets(
        ZmqEngineCoreClientConfig {
            input_address: input_address.clone(),
            output_address: output_address.clone(),
            engine_identity: engine_identity.clone(),
            ready_timeout,
            client_index: args.client_index,
        },
        input_socket,
        output_socket,
    )
    .await?;

    let ready_message = Some(handshake_ready_message);
    let request = build_request(request_id.clone(), args.max_tokens);
    client.add_request(request).await?;

    let output = wait_for_timeout(&mut client, &request_id, output_timeout).await?;
    let cleanup_output = if output.finished() {
        None
    } else {
        client.abort_requests(std::slice::from_ref(&request_id)).await?;
        Some(wait_for_timeout(&mut client, &request_id, output_timeout).await?)
    };

    client.shutdown().await?;

    println!("model={MODEL_NAME}");
    println!("handshake_address={}", args.handshake_address);
    println!("input_address={input_address}");
    println!("output_address={output_address}");
    println!("engine_identity_hex={}", hex_string(&engine_identity));
    println!("request_id={request_id}");
    println!("prompt_token_ids={PROMPT_TOKEN_IDS:?}");
    println!("ready_message={ready_message:?}");
    println!("new_token_ids={:?}", output.new_token_ids);
    println!("finish_reason={:?}", output.finish_reason);
    println!("stop_reason={:?}", output.stop_reason);
    if let Some(cleanup_output) = cleanup_output {
        println!("cleanup_finish_reason={:?}", cleanup_output.finish_reason);
    }

    Ok(())
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
