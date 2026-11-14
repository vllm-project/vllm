// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io;
use std::net::TcpListener;
use std::process::{Command as StdCommand, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::interval;
use tracing::info;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(200);
const MIN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Allocate one ephemeral TCP port for the managed headless-engine handshake on
/// the given host.
pub fn allocate_handshake_port(host: &str) -> Result<u16> {
    let listener = TcpListener::bind((host, 0)).context("failed to allocate handshake port")?;
    let port = listener
        .local_addr()
        .context("failed to inspect allocated handshake listener address")?
        .port();
    Ok(port)
}

/// Spawn configuration for one managed headless Python vLLM engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedEngineConfig {
    /// Python executable used to launch `vllm.entrypoints.cli.main`.
    pub python: String,
    /// Model identifier passed to `vllm ... serve <model>`.
    pub model: String,
    /// Host portion of the headless-engine handshake endpoint.
    pub handshake_host: String,
    /// Port portion of the headless-engine handshake endpoint.
    pub handshake_port: u16,
    /// Number of data parallel replicas across the whole deployment.
    ///
    /// The per-node replica count is forwarded separately in `python_args` as
    /// `--data-parallel-size-local`.
    pub data_parallel_size: usize,
    /// Handshake endpoint of the colocated frontend when it serves only this
    /// node's engines (hybrid or external DP LB). The engines then also
    /// handshake there for their input/output addresses, after fetching the
    /// coordinator and DP rendezvous from the frontend owning rank 0.
    pub client_handshake_address: Option<String>,
    /// Extra CLI arguments forwarded verbatim to Python vLLM.
    pub python_args: Vec<String>,
}

impl ManagedEngineConfig {
    /// Render the handshake address that the Rust frontend should dial.
    pub fn handshake_address(&self) -> String {
        format!("tcp://{}:{}", self.handshake_host, self.handshake_port)
    }

    /// Env vars every managed engine needs when serving the Rust frontend:
    /// `VLLM_USE_RUST_FRONTEND=1` tells Python vLLM to keep frontend-side
    /// multimodal normalization (`mm_device_do_normalize = False`,
    /// vllm/config/model.py) instead of expecting raw uint8 pixels.
    pub fn rust_frontend_envs() -> Vec<(String, String)> {
        vec![("VLLM_USE_RUST_FRONTEND".to_string(), "1".to_string())]
    }

    /// Build the concrete Python command line for the managed headless engine.
    pub fn to_command(&self) -> StdCommand {
        let mut command = StdCommand::new(&self.python);
        command
            .arg("-m")
            .arg("vllm.entrypoints.cli.main")
            .arg("serve")
            .arg(&self.model)
            .arg("--headless")
            .arg("--data-parallel-address")
            .arg(&self.handshake_host)
            .arg("--data-parallel-rpc-port")
            .arg(self.handshake_port.to_string())
            .arg("--data-parallel-size")
            .arg(self.data_parallel_size.to_string());
        if let Some(client_handshake_address) = &self.client_handshake_address {
            command.arg("--headless-client-handshake-address").arg(client_handshake_address);
        }
        command.args(&self.python_args);
        command.envs(Self::rust_frontend_envs());
        command
    }
}

/// RAII-style handle for one managed Python headless engine subprocess.
#[derive(Clone)]
pub struct ManagedEngineHandle {
    child: Arc<Mutex<Child>>,
    shutdown_started: Arc<AtomicBool>,
}

impl ManagedEngineHandle {
    /// Spawn one managed Python headless engine and return a handle for
    /// monitoring it.
    pub async fn spawn(config: ManagedEngineConfig) -> Result<Self> {
        let command = config.to_command();
        info!(
            handshake_address = %config.handshake_address(),
            ?command,
            "starting managed Python headless engine"
        );

        Self::spawn_command(command).await
    }

    /// Spawn one managed engine child in its own process group.
    async fn spawn_command(command: StdCommand) -> Result<Self> {
        let mut command = Command::from(command);
        command.stdin(Stdio::null()).stdout(Stdio::inherit()).stderr(Stdio::inherit());

        process_group::configure(&mut command);

        let child = command.spawn().context("failed to spawn managed engine")?;

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            shutdown_started: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Poll whether the managed engine has exited yet.
    pub async fn try_wait(&self) -> Option<ExitStatus> {
        let mut child = self.child.lock().await;
        child.try_wait().expect("failed to poll the status of managed engine")
    }

    /// Wait until the managed engine exits.
    pub async fn wait_for_exit(&self) -> ExitStatus {
        let mut interval = interval(CHILD_POLL_INTERVAL);
        loop {
            interval.tick().await;
            if let Some(status) = self.try_wait().await {
                return status;
            }
        }
    }

    /// Shut down the managed engine and wait for it to stop.
    ///
    /// SIGTERM goes only to the Python launcher process, not to the whole
    /// process group: the engine workers exit immediately on SIGTERM, which
    /// would fail every in-flight request, while the launcher drains them for
    /// up to `--shutdown-timeout` and then stops the workers in order. The
    /// group is still SIGKILLed if the launcher does not exit in time, and
    /// once more after it exits to reap any stragglers.
    pub async fn shutdown(&self, timeout: Duration) -> Result<()> {
        if self.shutdown_started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let Some(pid) = self.child.lock().await.id() else {
            return Ok(());
        };

        // Enforce a minimum shutdown timeout to give the engine process enough time to
        // clean up.
        let shutdown_timeout = std::cmp::max(timeout, MIN_SHUTDOWN_TIMEOUT);

        // First, try to gracefully terminate the launcher alone.
        info!(
            pid,
            ?shutdown_timeout,
            "shutting down managed engine with SIGTERM"
        );
        terminate(pid)?;

        // Wait for the process to exit on its own.
        if tokio::time::timeout(shutdown_timeout, self.wait_for_exit()).await.is_err() {
            // If it doesn't exit within the timeout, force kill it.
            info!(
                pid,
                "managed engine did not exit within timeout, sending SIGKILL"
            );
            process_group::kill(pid)?;
            let _ = self.wait_for_exit().await;
        }

        // Tear down the rest of the process subtree; an empty group is fine.
        process_group::kill(pid)?;
        Ok(())
    }
}

/// Send SIGTERM to one process.
fn terminate(pid: u32) -> Result<()> {
    signal(pid as i32, libc::SIGTERM)
}

/// Deliver one signal to a process or, for a negative `target`, to the process
/// group of that id.
fn signal(target: i32, signal: i32) -> Result<()> {
    let rc = unsafe { libc::kill(target, signal) };
    if rc == 0 {
        return Ok(());
    }

    let error = io::Error::last_os_error();
    if matches!(error.raw_os_error(), Some(code) if code == libc::ESRCH) {
        return Ok(());
    }
    Err(error).context("failed to signal managed engine")
}

/// Process group helper functions for managing the Python subprocess and its
/// potential children in a platform-aware way.
mod process_group {
    use super::*;

    /// Place the Python child into its own process group so `serve` can tear
    /// down the whole subtree rather than just the immediate shell process.
    pub fn configure(command: &mut Command) {
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    /// Send SIGKILL to the managed Python process group.
    pub fn kill(pid: u32) -> Result<()> {
        signal(-(pid as i32), libc::SIGKILL)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, Instant};

    use expect_test::expect;

    use super::{
        MIN_SHUTDOWN_TIMEOUT, ManagedEngineConfig, ManagedEngineHandle, allocate_handshake_port,
    };

    #[test]
    fn command_snapshot() {
        let config = ManagedEngineConfig {
            python: "python3".to_string(),
            model: "Qwen/Qwen3-0.6B".to_string(),
            handshake_host: "127.0.0.1".to_string(),
            handshake_port: 62100,
            data_parallel_size: 4,
            client_handshake_address: None,
            python_args: vec![
                "--data-parallel-size-local".to_string(),
                "2".to_string(),
                "--data-parallel-start-rank".to_string(),
                "2".to_string(),
                "--dtype".to_string(),
                "float16".to_string(),
                "--max-model-len".to_string(),
                "512".to_string(),
            ],
        };
        let command = config.to_command();
        let args = command.get_args().collect::<Vec<_>>();

        expect![[r#"
            [
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                "Qwen/Qwen3-0.6B",
                "--headless",
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                "62100",
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "2",
                "--dtype",
                "float16",
                "--max-model-len",
                "512",
            ]
        "#]]
        .assert_debug_eq(&args);

        let envs = command.get_envs().collect::<Vec<_>>();
        expect![[r#"
            [
                (
                    "VLLM_USE_RUST_FRONTEND",
                    Some(
                        "1",
                    ),
                ),
            ]
        "#]]
        .assert_debug_eq(&envs);
    }

    #[test]
    fn rust_frontend_envs_include_rust_frontend_flag() {
        expect![[r#"
            [
                (
                    "VLLM_USE_RUST_FRONTEND",
                    "1",
                ),
            ]
        "#]]
        .assert_debug_eq(&ManagedEngineConfig::rust_frontend_envs());
    }

    #[test]
    fn allocate_handshake_port_returns_non_zero_port() {
        let port = allocate_handshake_port("127.0.0.1").unwrap();
        assert_ne!(port, 0);
    }

    async fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.exists() {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {}",
                path.display()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn spawn_script(dir: &Path, script: &str) -> ManagedEngineHandle {
        let path = dir.join("child.sh");
        fs::write(&path, script).unwrap();
        let mut command = std::process::Command::new("sh");
        command.arg(&path).arg(dir);
        ManagedEngineHandle::spawn_command(command).await.unwrap()
    }

    fn process_alive(pid: i32) -> bool {
        if unsafe { libc::kill(pid, 0) } != 0 {
            return false;
        }
        // A zombie has been killed but not reaped yet; treat it as gone.
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        !matches!(stat.rsplit(") ").next(), Some(rest) if rest.starts_with('Z'))
    }

    #[tokio::test]
    async fn shutdown_signals_only_launcher_and_reaps_process_group() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("grandchild.sh"),
            "#!/bin/sh\ntrap 'touch \"$1/grandchild-term\"' TERM\necho $$ > \"$1/grandchild.pid\"\nsleep 600\n",
        )
        .unwrap();
        let handle = spawn_script(
            dir.path(),
            "#!/bin/sh\ntrap 'touch \"$1/child-term\"; exit 0' TERM\nsh \"$1/grandchild.sh\" \"$1\" &\ntouch \"$1/child-ready\"\nwait\n",
        )
        .await;

        wait_for_file(&dir.path().join("child-ready")).await;
        wait_for_file(&dir.path().join("grandchild.pid")).await;
        let grandchild_pid: i32 = fs::read_to_string(dir.path().join("grandchild.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();

        handle.shutdown(Duration::from_secs(30)).await.unwrap();

        assert!(dir.path().join("child-term").exists());
        assert!(!dir.path().join("grandchild-term").exists());

        // The post-exit group SIGKILL reaps the grandchild; allow a brief
        // window for the zombie to be reaped.
        let deadline = Instant::now() + Duration::from_secs(10);
        while process_alive(grandchild_pid) {
            assert!(
                Instant::now() < deadline,
                "grandchild {grandchild_pid} still alive"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[tokio::test]
    async fn shutdown_kills_group_when_launcher_ignores_sigterm() {
        let dir = tempfile::tempdir().unwrap();
        let handle = spawn_script(
            dir.path(),
            "#!/bin/sh\ntrap '' TERM\ntouch \"$1/child-ready\"\nsleep 600\n",
        )
        .await;
        wait_for_file(&dir.path().join("child-ready")).await;

        let started = Instant::now();
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
        let elapsed = started.elapsed();

        assert!(elapsed >= MIN_SHUTDOWN_TIMEOUT);
        assert!(elapsed < Duration::from_secs(15));
        assert!(handle.try_wait().await.is_some());
    }
}
