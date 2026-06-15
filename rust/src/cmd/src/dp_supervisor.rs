use std::ffi::OsString;
use std::process::Command as StdCommand;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Router, serve};
use futures::future::join_all;
use reqwest::Client;
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::{Instant, interval, sleep, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use vllm_managed_engine::process_group;

use crate::cli::ServeArgs;

const MONITOR_INTERVAL: Duration = Duration::from_millis(200);
const CHILD_EXIT_GRACE: Duration = Duration::from_secs(5);

impl ServeArgs {
    pub fn local_size(&self) -> usize {
        self.managed_engine
            .data_parallel_size_local
            .unwrap_or(self.managed_engine.data_parallel_size)
    }

    pub fn child_port(&self, local_rank: usize) -> u16 {
        self.port + local_rank as u16
    }

    pub fn child_ports(&self) -> Vec<u16> {
        (0..self.local_size()).map(|local_rank| self.child_port(local_rank)).collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.headless {
            return Err(
                "Error: --data-parallel-multi-port-external-lb does not support --headless because it manages child API servers"
                    .to_string(),
            );
        }
        if self.uds.is_some() {
            return Err(
                "Error: --data-parallel-multi-port-external-lb does not support --uds".to_string(),
            );
        }
        if self.runtime.grpc_port.is_some() {
            return Err(
                "Error: --data-parallel-multi-port-external-lb does not support --grpc-port"
                    .to_string(),
            );
        }
        if self.managed_engine.data_parallel_size < 2 {
            return Err(
                "Error: --data-parallel-multi-port-external-lb requires --data-parallel-size > 1"
                    .to_string(),
            );
        }

        let local_size = self.local_size();
        if local_size < 2 {
            return Err(
                "Error: --data-parallel-multi-port-external-lb requires --data-parallel-size-local >= 2"
                    .to_string(),
            );
        }
        if local_size > self.managed_engine.data_parallel_size {
            return Err(
                "Error: --data-parallel-size-local cannot exceed --data-parallel-size".to_string(),
            );
        }
        if self.managed_engine.data_parallel_size % local_size != 0 {
            return Err(
                "Error: --data-parallel-size must be divisible by --data-parallel-size-local"
                    .to_string(),
            );
        }
        let child_port_min = self.port;
        let child_port_max = self.port + local_size as u16 - 1;
        if child_port_min <= self.data_parallel_supervisor_port
            && self.data_parallel_supervisor_port <= child_port_max
        {
            return Err(format!(
                "Error: --data-parallel-supervisor-port {} overlaps with child rank ports {}-{}",
                self.data_parallel_supervisor_port, child_port_min, child_port_max
            ));
        }
        if self.dp_supervisor_probe_timeout_s == 0 {
            return Err("Error: --dp-supervisor-probe-timeout-s must be > 0".to_string());
        }
        if self.dp_supervisor_probe_failure_threshold == 0 {
            return Err("Error: --dp-supervisor-probe-failure-threshold must be >= 1".to_string());
        }
        let devices_per_rank = python_arg_value(
            &self.managed_engine.python_args,
            &["--tensor-parallel-size", "-tp"],
        )
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
            * python_arg_value(
                &self.managed_engine.python_args,
                &["--pipeline-parallel-size", "-pp"],
            )
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let required_visible_devices = local_size.saturating_mul(devices_per_rank);
        for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
            let Some(current) = std::env::var_os(key) else {
                continue;
            };
            let ids = current
                .to_string_lossy()
                .split(',')
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if ids.len() < required_visible_devices {
                return Err(format!(
                    "Error: {key} exposes {} device(s), but --data-parallel-size-local={local_size} with tp*pp={devices_per_rank} requires {required_visible_devices}",
                    ids.len()
                ));
            }
            break;
        }
        Ok(())
    }

    pub fn child_device_env(&self, local_rank: usize) -> Option<(String, String)> {
        let devices_per_rank = python_arg_value(
            &self.managed_engine.python_args,
            &["--tensor-parallel-size", "-tp"],
        )
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
            * python_arg_value(
                &self.managed_engine.python_args,
                &["--pipeline-parallel-size", "-pp"],
            )
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let start = local_rank.saturating_mul(devices_per_rank);
        let stop = start + devices_per_rank;
        let candidates = [
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ];

        for key in candidates {
            if let Some(current) = std::env::var_os(key) {
                let ids = current
                    .to_string_lossy()
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>();
                if ids.len() >= stop {
                    return Some((key.to_string(), ids[start..stop].join(",")));
                }
            }
        }

        let devices = (start..stop).map(|device| device.to_string()).collect::<Vec<_>>();
        (!devices.is_empty()).then(|| ("CUDA_VISIBLE_DEVICES".to_string(), devices.join(",")))
    }

    pub fn child_cli_args(&self, local_rank: usize) -> Vec<OsString> {
        child_cli_args_from_parent_args(std::env::args_os().skip(1), self.child_port(local_rank))
    }
}

fn python_arg_value<'a>(args: &'a [String], flags: &[&str]) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        let flag = window.first()?;
        flags.contains(&flag.as_str()).then_some(window.get(1)?.as_str())
    })
}

pub(crate) fn child_cli_args_from_parent_args<I>(args: I, child_port: u16) -> Vec<OsString>
where
    I: IntoIterator<Item = OsString>,
{
    let strip_value_flags = [
        "--data-parallel-address",
        "--data-parallel-size",
        "--data-parallel-size-local",
        "--dp-supervisor-probe-interval-s",
        "--dp-supervisor-probe-timeout-s",
        "--dp-supervisor-probe-failure-threshold",
        "--data-parallel-supervisor-port",
        "--data-parallel-rpc-port",
        "--handshake-port",
    ];
    let mut filtered_args = Vec::new();
    let mut input = args.into_iter();
    while let Some(arg) = input.next() {
        let Some(arg_str) = arg.to_str() else {
            filtered_args.push(arg);
            continue;
        };

        if arg_str == "--data-parallel-multi-port-external-lb"
            || matches!(
                arg_str.split_once('='),
                Some(("--data-parallel-multi-port-external-lb", "true"))
            )
        {
            continue;
        }

        if strip_value_flags.contains(&arg_str) {
            let _ = input.next();
            continue;
        }

        if let Some((flag, _)) = arg_str.split_once('=')
            && strip_value_flags.contains(&flag)
        {
            continue;
        }

        filtered_args.push(arg);
    }

    let value: OsString = child_port.to_string().into();
    let separator = filtered_args.iter().position(|arg| arg == "--").unwrap_or(filtered_args.len());
    for idx in 0..separator {
        if filtered_args[idx] == "--port" {
            if idx + 1 < separator {
                filtered_args[idx + 1] = value;
            } else {
                filtered_args.insert(idx + 1, value);
            }
            return filtered_args;
        }

        if let Some(arg) = filtered_args[idx].to_str()
            && let Some((matched_flag, _)) = arg.split_once('=')
            && matched_flag == "--port"
        {
            filtered_args[idx] = format!("--port={}", value.to_string_lossy()).into();
            return filtered_args;
        }
    }

    filtered_args.insert(separator, "--port".into());
    filtered_args.insert(separator + 1, value);
    filtered_args
}

#[derive(Default)]
struct SupervisorHealthState {
    ready: AtomicBool,
    shutting_down: AtomicBool,
}

impl SupervisorHealthState {
    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst) && !self.shutting_down.load(Ordering::SeqCst)
    }

    fn mark_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }

    fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.mark_ready(false);
    }
}

#[derive(Clone)]
struct ChildServer {
    name: String,
    port: u16,
    child: Arc<Mutex<Child>>,
}

async fn child_pid(child: &ChildServer) -> Option<u32> {
    child.child.lock().await.id()
}

async fn try_wait_child(child: &ChildServer) -> Result<Option<ExitStatus>> {
    let mut process = child.child.lock().await;
    process.try_wait().context("failed to poll child process")
}

async fn wait_for_child_exit(child: &ChildServer) -> Result<ExitStatus> {
    let mut ticker = interval(MONITOR_INTERVAL);
    loop {
        ticker.tick().await;
        if let Some(status) = try_wait_child(child).await? {
            return Ok(status);
        }
    }
}

pub async fn run(args: ServeArgs, shutdown: CancellationToken) -> Result<()> {
    args.validate().map_err(anyhow::Error::msg)?;

    let health_state = Arc::new(SupervisorHealthState::default());
    let health_shutdown = shutdown.child_token();
    let listener = TcpListener::bind((args.host.as_str(), args.data_parallel_supervisor_port))
        .await
        .with_context(|| {
            format!(
                "failed to bind dp supervisor on {}:{}",
                args.host, args.data_parallel_supervisor_port
            )
        })?;
    let local_addr = listener.local_addr().context("failed to inspect supervisor listener")?;
    let app = Router::new()
        .route("/health", get(supervisor_health))
        .route("/ready", get(supervisor_health))
        .route("/readyz", get(supervisor_health))
        .with_state(health_state.clone());
    info!(%local_addr, "starting DP supervisor health server");
    let health_server_shutdown = health_shutdown.clone();
    let health_server = tokio::spawn(async move {
        serve(listener, app)
            .with_graceful_shutdown(health_server_shutdown.cancelled_owned())
            .await
            .context("dp supervisor health server failed")
    });

    let children = spawn_children(&args).await?;
    info!(
        supervisor_port = args.data_parallel_supervisor_port,
        child_ports = ?args.child_ports(),
        local_size = args.local_size(),
        "started Rust DP supervisor"
    );

    let outcome = monitor_children(
        args.clone(),
        children.clone(),
        health_state.clone(),
        shutdown.clone(),
    )
    .await;

    health_state.begin_shutdown();
    shutdown.cancel();
    health_shutdown.cancel();

    let shutdown_result = shutdown_children(&args, &children).await;
    let server_result = health_server.await.context("health server task join failed")?;

    shutdown_result?;
    server_result?;
    outcome
}

async fn supervisor_health(State(state): State<Arc<SupervisorHealthState>>) -> StatusCode {
    if state.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn spawn_children(args: &ServeArgs) -> Result<Vec<ChildServer>> {
    let mut children = Vec::with_capacity(args.local_size());
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    for local_rank in 0..args.local_size() {
        let mut command = Command::from({
            let mut command = StdCommand::new(&current_exe);
            command.args(args.child_cli_args(local_rank));
            command
        });
        command.stdin(Stdio::null()).stdout(Stdio::inherit()).stderr(Stdio::inherit());
        process_group::configure(&mut command);
        if let Some((key, value)) = args.child_device_env(local_rank) {
            command.env(key, value);
        }
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn child ApiServer_{local_rank}"))?;
        children.push(ChildServer {
            name: format!("ApiServer_{local_rank}"),
            port: args.child_port(local_rank),
            child: Arc::new(Mutex::new(child)),
        });
    }
    Ok(children)
}

async fn monitor_children(
    args: ServeArgs,
    children: Vec<ChildServer>,
    health_state: Arc<SupervisorHealthState>,
    shutdown: CancellationToken,
) -> Result<()> {
    let probe_shutdown = shutdown.child_token();
    let probe_children = children.clone();
    let probe_state = health_state.clone();
    let probe_args = args.clone();
    let mut probe_task = Some(tokio::spawn(async move {
        probe_children_loop(probe_args, probe_children, probe_state, probe_shutdown).await
    }));
    let mut ticker = interval(MONITOR_INTERVAL);

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            _ = ticker.tick() => {
                let mut failed_children = 0usize;
                for child in &children {
                    if try_wait_child(child).await?.is_some() {
                        failed_children += 1;
                    }
                }
                if failed_children > 0 {
                    info!(failed_children, "DPSupervisor found exited DP Servers");
                    health_state.begin_shutdown();
                    shutdown.cancel();
                    return Ok(());
                }

                if probe_task.as_ref().is_some_and(JoinHandle::is_finished) {
                    let result = probe_task
                        .take()
                        .expect("probe task should be present")
                        .await;
                    let probe_error = match result {
                        Ok(Ok(())) => None,
                        Ok(Err(error)) => Some(error.to_string()),
                        Err(error) => Some(format!("probe task join failed: {error}")),
                    };
                    info!(
                        exception = probe_error.as_deref().unwrap_or("None"),
                        "DPSupervisor probe task stopped"
                    );
                    health_state.begin_shutdown();
                    shutdown.cancel();
                    return Ok(());
                }
            }
        }
    }
}

async fn probe_children_loop(
    args: ServeArgs,
    children: Vec<ChildServer>,
    health_state: Arc<SupervisorHealthState>,
    shutdown: CancellationToken,
) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(
            args.dp_supervisor_probe_timeout_s.into(),
        ))
        .build()
        .context("failed to build DP supervisor HTTP client")?;
    let host = match args.host.as_str() {
        "" | "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    let retry_delay = Duration::from_secs(args.dp_supervisor_probe_interval_s.into());

    while !shutdown.is_cancelled() {
        let threshold = if health_state.is_ready() {
            args.dp_supervisor_probe_failure_threshold
        } else {
            1
        };
        let probe_results = join_all(children.iter().map(|child| {
            probe_child_health(
                &client,
                format!("http://{host}:{}/health", child.port),
                threshold,
                retry_delay,
            )
        }))
        .await;
        let num_unhealthy = probe_results.into_iter().filter(|healthy| !healthy).count();

        if num_unhealthy == 0 {
            if !shutdown.is_cancelled() {
                health_state.mark_ready(true);
            }
        } else if health_state.is_ready() {
            info!(
                num_unhealthy,
                "DPSupervisor probe found unhealthy DP Servers"
            );
            health_state.begin_shutdown();
            shutdown.cancel();
            return Ok(());
        }

        if args.dp_supervisor_probe_interval_s == 0 {
            tokio::task::yield_now().await;
        } else {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                _ = sleep(Duration::from_secs(args.dp_supervisor_probe_interval_s.into())) => {}
            }
        }
    }

    Ok(())
}

async fn probe_child_health(
    client: &Client,
    url: String,
    threshold: usize,
    retry_delay: Duration,
) -> bool {
    for attempt in 0..threshold {
        match client.get(&url).send().await {
            Ok(response) => return response.status() == StatusCode::OK,
            Err(error) if error.is_connect() || error.is_timeout() => {
                if attempt + 1 < threshold && !retry_delay.is_zero() {
                    sleep(retry_delay).await;
                }
            }
            Err(_) => return false,
        }
    }

    false
}

async fn shutdown_children(args: &ServeArgs, children: &[ChildServer]) -> Result<()> {
    let shutdown_timeout = args.runtime.shutdown_timeout() + CHILD_EXIT_GRACE;
    info!(
        ?shutdown_timeout,
        child_count = children.len(),
        "shutting down DP supervisor children"
    );

    for child in children {
        if let Some(pid) = child_pid(child).await {
            process_group::terminate(pid)
                .with_context(|| format!("failed to terminate child {}", child.name))?;
        }
    }

    let deadline = Instant::now() + shutdown_timeout;
    for child in children {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        if timeout(remaining, wait_for_child_exit(child)).await.is_err() {
            warn!(
                child = %child.name,
                "child did not exit before shutdown deadline; sending SIGKILL"
            );
            if let Some(pid) = child_pid(child).await {
                process_group::kill(pid)
                    .with_context(|| format!("failed to kill child {}", child.name))?;
            }
            let _ = wait_for_child_exit(child).await?;
        }
    }

    Ok(())
}
