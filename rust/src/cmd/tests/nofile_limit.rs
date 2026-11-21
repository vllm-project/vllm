// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::process::CommandExt as _;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn read_http_headers(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0_u8; 256];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before response headers completed",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }
    }
}

fn send_health_check(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")?;
    let response = read_http_headers(stream)?;
    assert!(
        response.starts_with("HTTP/1.1 200 "),
        "expected HTTP 200 response, got: {response:?}"
    );
    Ok(())
}

fn wait_for_ready(child: &mut ChildGuard, addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if let Some(status) = child.0.try_wait().expect("failed to poll child status") {
            panic!("vllm-rs exited prematurely with {status}");
        }
        if let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(100)) {
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
            if send_health_check(&mut stream).is_ok() {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for vllm-rs render server at {addr}");
}

/// Regression test for #68023: when `vllm-rs` starts with the typical Linux
/// default soft `RLIMIT_NOFILE=1024` (and a higher hard limit), it must raise
/// its soft limit so it can accept more than 1,024 concurrent connections
/// without failing `accept()` with `EMFILE`.
#[test]
fn render_raises_soft_nofile_above_1024_default() {
    const TARGET_CONNECTIONS: usize = 1100;
    const INITIAL_SOFT_LIMIT: u64 = 1024;

    let (_, hard_limit) = rlimit::Resource::NOFILE.get().expect("failed to get RLIMIT_NOFILE");
    if hard_limit <= TARGET_CONNECTIONS as u64 + 128 {
        eprintln!(
            "skipping test: RLIMIT_NOFILE hard limit ({hard_limit}) too low for {TARGET_CONNECTIONS} connections"
        );
        return;
    }

    // Ensure the test process itself can hold >1,024 client sockets even if the
    // test runner started with a 1024 soft limit.
    rlimit::increase_nofile_limit(u64::MAX).expect("failed to raise test runner NOFILE limit");

    let model_dir = tempfile::tempdir().expect("failed to create temp model dir");
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{"vocab_size": 128, "eos_token_id": [7, 9]}"#,
    )
    .expect("failed to write config.json");

    let port_probe = TcpListener::bind("127.0.0.1:0").expect("failed to bind ephemeral port");
    let addr = port_probe.local_addr().expect("failed to inspect port");
    drop(port_probe);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vllm-rs"));
    cmd.args([
        "render",
        model_dir.path().to_str().expect("temp path is valid utf-8"),
        "--skip-tokenizer-init",
        "--host",
        "127.0.0.1",
        "--port",
        &addr.port().to_string(),
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    // SAFETY: `rlimit::Resource::NOFILE.set` only invokes `setrlimit(2)` with no
    // heap allocations, so it is async-signal-safe in `pre_exec`.
    unsafe {
        cmd.pre_exec(move || rlimit::Resource::NOFILE.set(INITIAL_SOFT_LIMIT, hard_limit));
    }

    let mut child = ChildGuard(cmd.spawn().expect("failed to spawn vllm-rs render"));
    wait_for_ready(&mut child, addr);

    let mut connections = Vec::with_capacity(TARGET_CONNECTIONS);
    for i in 0..TARGET_CONNECTIONS {
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
            .unwrap_or_else(|err| panic!("connect failed at connection {i}: {err}"));
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
        send_health_check(&mut stream)
            .unwrap_or_else(|err| panic!("health check failed at connection {i}: {err}"));
        connections.push(stream);
    }

    assert_eq!(connections.len(), TARGET_CONNECTIONS);
}
