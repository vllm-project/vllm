use std::io;
use std::net::TcpListener;
use std::process::{Command as StdCommand, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{interval, timeout};
use tracing::info;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(200);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
/// Loopback host used for managed-mode handshake traffic between the Rust frontend
/// and the Python headless engine.
pub const MANAGED_ENGINE_HANDSHAKE_HOST: &str = "127.0.0.1";

/// Allocate one ephemeral loopback TCP port for the managed headless-engine handshake.
pub fn allocate_handshake_port() -> Result<u16> {
    let listener = TcpListener::bind((MANAGED_ENGINE_HANDSHAKE_HOST, 0))
        .context("failed to allocate loopback handshake port")?;
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
    ///
    /// In managed mode this is always loopback and is not user-configurable.
    pub handshake_host: String,
    /// Port portion of the headless-engine handshake endpoint.
    pub handshake_port: u16,
    /// Extra CLI arguments forwarded verbatim to Python vLLM.
    pub python_args: Vec<String>,
}

impl ManagedEngineConfig {
    /// Render the handshake address that the Rust frontend should dial.
    pub fn handshake_address(&self) -> String {
        format!("tcp://{}:{}", self.handshake_host, self.handshake_port)
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
            .arg("--data-parallel-size-local")
            .arg("1")
            .args(&self.python_args);
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
    /// Spawn one managed Python headless engine and return a handle for monitoring it.
    pub async fn spawn(config: ManagedEngineConfig) -> Result<Self> {
        let command = config.to_command();
        info!(
            handshake_address = %config.handshake_address(),
            ?command,
            "starting managed Python headless engine"
        );

        let mut command = Command::from(command);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

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
        child
            .try_wait()
            .expect("failed to poll the status of managed engine")
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

    /// Terminate the managed engine process group and wait for it to stop.
    pub async fn shutdown(&self) -> Result<()> {
        if self.shutdown_started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let Some(pid) = self.child.lock().await.id() else {
            return Ok(());
        };

        // First, try to gracefully terminate.
        process_group::terminate(pid)?;

        // Wait for the process to exit on its own.
        if timeout(SHUTDOWN_TIMEOUT, self.wait_for_exit())
            .await
            .is_ok()
        {
            return Ok(());
        }

        // If it doesn't exit within the timeout, force kill it.
        process_group::kill(pid)?;

        let _ = self.wait_for_exit().await;
        Ok(())
    }
}

/// Process group helper functions for managing the Python subprocess and its potential children in
/// a platform-aware way.
mod process_group {
    use super::*;

    #[cfg(unix)]
    /// Place the Python child into its own process group so `serve` can tear down
    /// the whole subtree rather than just the immediate shell process.
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

    #[cfg(not(unix))]
    pub fn configure(_: &mut Command) {}

    #[cfg(unix)]
    /// Send SIGTERM to the managed Python process group.
    pub fn terminate(pid: u32) -> Result<()> {
        signal(pid, libc::SIGTERM)
    }

    #[cfg(not(unix))]
    pub fn terminate(_: u32) -> Result<()> {
        Ok(())
    }

    #[cfg(unix)]
    /// Send SIGKILL to the managed Python process group.
    pub fn kill(pid: u32) -> Result<()> {
        signal(pid, libc::SIGKILL)
    }

    #[cfg(not(unix))]
    pub fn kill(_: u32) -> Result<()> {
        Ok(())
    }

    #[cfg(unix)]
    /// Deliver one signal to the managed Python process group.
    fn signal(pid: u32, signal: i32) -> Result<()> {
        let rc = unsafe { libc::kill(-(pid as i32), signal) };
        if rc == 0 {
            return Ok(());
        }

        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(code) if code == libc::ESRCH) {
            return Ok(());
        }
        Err(error).context("failed to signal managed engine process group")
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{ManagedEngineConfig, allocate_handshake_port};

    #[test]
    #[cfg(unix)]
    fn command_snapshot() {
        let config = ManagedEngineConfig {
            python: "python3".to_string(),
            model: "Qwen/Qwen3-0.6B".to_string(),
            handshake_host: "127.0.0.1".to_string(),
            handshake_port: 62100,
            python_args: vec![
                "--dtype".to_string(),
                "float16".to_string(),
                "--max-model-len".to_string(),
                "512".to_string(),
            ],
        };

        expect![[r#"
            Command {
                program: "python3",
                args: [
                    "python3",
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "serve",
                    "Qwen/Qwen3-0.6B",
                    "--headless",
                    "--data-parallel-address",
                    "127.0.0.1",
                    "--data-parallel-rpc-port",
                    "62100",
                    "--data-parallel-size-local",
                    "1",
                    "--dtype",
                    "float16",
                    "--max-model-len",
                    "512",
                ],
            }
        "#]]
        .assert_debug_eq(&config.to_command());
    }

    #[test]
    fn allocate_handshake_port_returns_non_zero_port() {
        let port = allocate_handshake_port().unwrap();
        assert_ne!(port, 0);
    }
}
