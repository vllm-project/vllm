use std::time::Duration;

use expect_test::expect;
use vllm_engine_core_client::TransportMode;
use vllm_server::Config;

use super::{Cli, Command};

#[test]
fn serve_args_forward_python_flags_with_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "../vllm/.venv/bin/python",
        "--max-model-len",
        "512",
        "--",
        "--dtype",
        "float16",
    ])
    .unwrap();

    expect![[r#"
        Cli {
            command: Serve(
                ServeArgs {
                    headless: false,
                    python: "../vllm/.venv/bin/python",
                    host: "127.0.0.1",
                    port: 8000,
                    handshake_host: "127.0.0.1",
                    handshake_port: None,
                    data_parallel_size: 1,
                    data_parallel_size_local: None,
                    runtime: SharedRuntimeArgs {
                        model: "Qwen/Qwen3-0.6B",
                        engine_ready_timeout_secs: 300,
                        tool_call_parser: None,
                        reasoning_parser: None,
                        max_model_len: Some(
                            512,
                        ),
                        enable_log_requests: false,
                        disable_log_stats: false,
                    },
                    python_args: [
                        "--dtype",
                        "float16",
                    ],
                },
            ),
        }
    "#]]
    .assert_debug_eq(&cli);
}

#[test]
fn serve_args_auto_forward_python_flags_without_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "python3",
        "--dtype",
        "float16",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["--dtype", "float16"]);
}

#[test]
fn serve_args_auto_forward_python_multi_char_alias_without_separator() {
    let cli = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "-tp", "2"]).unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["--tensor-parallel-size", "2"]);
}

#[test]
fn serve_args_keep_frontend_unsupported_args_before_separator() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--tokenizer-mode",
        "auto",
        "--dtype",
        "float16",
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value 'auto' for '--tokenizer-mode <TOKENIZER_MODE>': argument is not implemented in Rust frontend yet

        Remove this unsupported argument to continue.

        Alternatively, if you intend to pass it only to the Python engine, put it after `--` (e.g., `-- <arg>`).
        This may lead to unexpected behavior as the Rust frontend will completely ignore that argument.

        For more information, try '--help'.
    "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_value_arg() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--uds",
        "/tmp/vllm.sock",
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value '/tmp/vllm.sock' for '--uds <UDS>': argument is not implemented in Rust frontend yet

        Remove this unsupported argument to continue.

        Alternatively, if you intend to pass it only to the Python engine, put it after `--` (e.g., `-- <arg>`).
        This may lead to unexpected behavior as the Rust frontend will completely ignore that argument.

        For more information, try '--help'.
    "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_flag_arg() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--allow-credentials"])
        .unwrap_err();

    expect![[r#"
        error: invalid value 'true' for '--allow-credentials [<ALLOW_CREDENTIALS>]': argument is not implemented in Rust frontend yet

        Remove this unsupported argument to continue.

        Alternatively, if you intend to pass it only to the Python engine, put it after `--` (e.g., `-- <arg>`).
        This may lead to unexpected behavior as the Rust frontend will completely ignore that argument.

        For more information, try '--help'.
    "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_no_flag_alias() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--no-enable-log-deltas",
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value 'true' for '--enable-log-deltas [<ENABLE_LOG_DELTAS>]': argument is not implemented in Rust frontend yet

        Remove this unsupported argument to continue.

        Alternatively, if you intend to pass it only to the Python engine, put it after `--` (e.g., `-- <arg>`).
        This may lead to unexpected behavior as the Rust frontend will completely ignore that argument.

        For more information, try '--help'.
    "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn frontend_args_accept_json() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--coordinator-address",
        "tcp://127.0.0.1:7000",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","engine_count":2}"#,
    ])
    .unwrap();

    expect![[r#"
        Cli {
            command: Frontend(
                FrontendArgs {
                    listen_fd: 3,
                    input_address: "ipc:///tmp/input.sock",
                    output_address: "ipc:///tmp/output.sock",
                    coordinator_address: Some(
                        "tcp://127.0.0.1:7000",
                    ),
                    engine_count: 1,
                    runtime: SharedRuntimeArgs {
                        model: "Qwen/Qwen3-0.6B",
                        engine_ready_timeout_secs: 300,
                        tool_call_parser: None,
                        reasoning_parser: None,
                        max_model_len: None,
                        enable_log_requests: false,
                        disable_log_stats: false,
                    },
                },
            ),
        }
    "#]]
    .assert_debug_eq(&cli);
}

#[test]
fn frontend_args_json_applies_defaults() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B"}"#,
    ])
    .unwrap();

    let Command::Frontend(args) = cli.command else {
        panic!("expected frontend args");
    };
    assert_eq!(args.runtime.model, "Qwen/Qwen3-0.6B");
    assert_eq!(args.runtime.engine_ready_timeout_secs, 300);
    assert_eq!(args.runtime.tool_call_parser, None);
    assert_eq!(args.runtime.reasoning_parser, None);
    assert_eq!(args.runtime.max_model_len, None);
}

#[test]
fn frontend_args_json_accepts_supported_non_default_fields() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","engine_ready_timeout_secs":42,"tool_call_parser":"hermes","reasoning_parser":"qwen3_thinking","max_model_len":8192}"#,
    ])
    .unwrap();

    let Command::Frontend(args) = cli.command else {
        panic!("expected frontend args");
    };
    assert_eq!(args.runtime.engine_ready_timeout_secs, 42);
    assert_eq!(args.runtime.tool_call_parser.as_deref(), Some("hermes"));
    assert_eq!(
        args.runtime.reasoning_parser.as_deref(),
        Some("qwen3_thinking")
    );
    assert_eq!(args.runtime.max_model_len, Some(8192));
}

#[test]
fn frontend_args_json_ignores_unknown_fields() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","unknown_field":"ignored","nested_unknown":{"x":1}}"#,
    ])
    .unwrap();

    let Command::Frontend(args) = cli.command else {
        panic!("expected frontend args");
    };
    assert_eq!(args.runtime.model, "Qwen/Qwen3-0.6B");
}

#[test]
fn frontend_args_json_accepts_noop_fields() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","api_server_count":2}"#,
    ])
    .unwrap();

    let Command::Frontend(args) = cli.command else {
        panic!("expected frontend args");
    };
    assert_eq!(args.runtime.model, "Qwen/Qwen3-0.6B");
}

#[test]
fn frontend_args_json_rejects_unsupported_fields() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","allow_credentials":true}"#,
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value '{"model_tag":"Qwen/Qwen3-0.6B","allow_credentials":true}' for '--args-json <JSON>': 
        The following arguments are not implemented in Rust frontend yet:
        - allow_credentials

        Remove these arguments to continue.

        For more information, try '--help'.
    "#]].assert_eq(&error.to_string());
}

#[test]
fn frontend_args_json_aggregates_multiple_unsupported_fields() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B","allow_credentials":true,"api_key":"secret"}"#,
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value '{"model_tag":"Qwen/Qwen3-0.6B","allow_credentials":true,"api_key":"secret"}' for '--args-json <JSON>': 
        The following arguments are not implemented in Rust frontend yet:
        - allow_credentials
        - api_key

        Remove these arguments to continue.

        For more information, try '--help'.
    "#]].assert_eq(&error.to_string());
}

#[test]
fn frontend_args_json_rejects_malformed_json() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B""#,
    ])
    .unwrap_err();

    expect![[r#"
        error: invalid value '{"model_tag":"Qwen/Qwen3-0.6B"' for '--args-json <JSON>': invalid JSON arguments: EOF while parsing an object at line 1 column 30

        For more information, try '--help'.
    "#]].assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_flags_before_model() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "--python", "python3", "Qwen/Qwen3-0.6B"])
        .unwrap_err();

    expect![[r#"
            error: serve requires the model to appear immediately after the subcommand

            Usage: vllm-rs serve <MODEL> [OPTIONS] [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_accept_headless_mode() {
    let cli = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--headless"]).unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert!(args.headless);
}

#[test]
fn serve_args_keep_python_passthrough_flags_after_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "python3",
        "--",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "float16",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(
        args.python_args,
        vec!["--tensor-parallel-size", "2", "--dtype", "float16"]
    );
}

#[test]
fn serve_args_keep_python_multi_char_alias_after_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "python3",
        "--",
        "-tp",
        "2",
        "--dtype",
        "float16",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["-tp", "2", "--dtype", "float16"]);
}

#[test]
fn serve_args_keep_frontend_arg_after_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--",
        "--uds",
        "/tmp/vllm.sock",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["--uds", "/tmp/vllm.sock"]);
}

#[test]
fn serve_args_keep_python_multi_char_engine_aliases_after_separator() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--",
        "-dpr",
        "1",
        "-dpl",
        "2",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["-dpr", "1", "-dpl", "2"]);
}

#[test]
fn serve_args_auto_forward_unknown_flags_without_separator() {
    let cli = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--foo", "bar"]).unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["--foo", "bar"]);
}

#[test]
fn serve_args_auto_forward_negative_value_without_separator() {
    let cli =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--dtype", "-1"]).unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert_eq!(args.python_args, vec!["--dtype", "-1"]);
}

#[test]
fn serve_args_accept_handshake_aliases() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "python3",
        "--handshake-host",
        "10.99.48.128",
        "--handshake-port",
        "13345",
        "--data-parallel-size",
        "4",
    ])
    .unwrap();

    expect![[r#"
        Cli {
            command: Serve(
                ServeArgs {
                    headless: false,
                    python: "python3",
                    host: "127.0.0.1",
                    port: 8000,
                    handshake_host: "10.99.48.128",
                    handshake_port: Some(
                        13345,
                    ),
                    data_parallel_size: 4,
                    data_parallel_size_local: None,
                    runtime: SharedRuntimeArgs {
                        model: "Qwen/Qwen3-0.6B",
                        engine_ready_timeout_secs: 300,
                        tool_call_parser: None,
                        reasoning_parser: None,
                        max_model_len: None,
                        enable_log_requests: false,
                        disable_log_stats: false,
                    },
                    python_args: [],
                },
            ),
        }
    "#]]
    .assert_debug_eq(&cli);
}

#[test]
fn serve_args_accept_data_parallel_primary_flags() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--data-parallel-address",
        "10.99.48.128",
        "--data-parallel-rpc-port",
        "13345",
        "--data-parallel-size",
        "4",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    assert!(!args.headless);
    assert_eq!(args.handshake_host, "10.99.48.128");
    assert_eq!(args.handshake_port, Some(13345));
    assert_eq!(args.data_parallel_size, 4);
}

#[test]
fn serve_frontend_config_uses_dp_address_as_advertised_host() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--handshake-host",
        "10.99.48.128",
        "--data-parallel-size",
        "4",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    let config = args.to_frontend_config("tcp://10.99.48.128:29550".to_string());

    let TransportMode::HandshakeOwner {
        handshake_address,
        advertised_host,
        engine_count,
        ready_timeout,
        local_input_address,
        local_output_address,
    } = &config.transport_mode
    else {
        panic!("expected handshake-owned transport");
    };

    assert_eq!(handshake_address, "tcp://10.99.48.128:29550");
    assert_eq!(advertised_host, "10.99.48.128");
    assert_eq!(*engine_count, 4);
    assert_eq!(*ready_timeout, Duration::from_secs(300));
    assert!(
        local_input_address
            .as_deref()
            .is_some_and(|address| address.starts_with("ipc://"))
    );
    assert!(
        local_output_address
            .as_deref()
            .is_some_and(|address| address.starts_with("ipc://"))
    );
    assert_ne!(local_input_address, local_output_address);

    expect![[r#"
        Config {
            transport_mode: HandshakeOwner {
                handshake_address: "tcp://10.99.48.128:29550",
                advertised_host: "10.99.48.128",
                engine_count: 4,
                ready_timeout: 300s,
                local_input_address: Some(
                    "<ipc input>",
                ),
                local_output_address: Some(
                    "<ipc output>",
                ),
            },
            coordinator_mode: MaybeInProc,
            model: "Qwen/Qwen3-0.6B",
            listener_mode: Bind {
                host: "127.0.0.1",
                port: 8000,
            },
            tool_call_parser: None,
            reasoning_parser: None,
            enable_log_requests: false,
            disable_log_stats: false,
        }
    "#]]
    .assert_debug_eq(&Config {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: handshake_address.clone(),
            advertised_host: advertised_host.clone(),
            engine_count: *engine_count,
            ready_timeout: *ready_timeout,
            local_input_address: Some("<ipc input>".to_string()),
            local_output_address: Some("<ipc output>".to_string()),
        },
        ..config.clone()
    });
}

#[test]
fn serve_frontend_config_keeps_tcp_transport_for_non_local_only_topology() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--data-parallel-address",
        "10.99.48.128",
        "--data-parallel-size",
        "4",
        "--data-parallel-size-local",
        "2",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    let config = args.to_frontend_config("tcp://10.99.48.128:29550".to_string());

    expect![[r#"
        Config {
            transport_mode: HandshakeOwner {
                handshake_address: "tcp://10.99.48.128:29550",
                advertised_host: "10.99.48.128",
                engine_count: 4,
                ready_timeout: 300s,
                local_input_address: None,
                local_output_address: None,
            },
            coordinator_mode: MaybeInProc,
            model: "Qwen/Qwen3-0.6B",
            listener_mode: Bind {
                host: "127.0.0.1",
                port: 8000,
            },
            tool_call_parser: None,
            reasoning_parser: None,
            enable_log_requests: false,
            disable_log_stats: false,
        }
    "#]]
    .assert_debug_eq(&config);
}

#[test]
fn frontend_args_reject_legacy_handshake_flags() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B"}"#,
        "--handshake-address",
        "tcp://127.0.0.1:62100",
    ])
    .unwrap_err();

    assert!(error.to_string().contains("--handshake-address"));
}

#[test]
fn frontend_config_uses_external_coordinator_when_coordinator_address_is_present() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "--listen-fd",
        "3",
        "--input-address",
        "ipc:///tmp/input.sock",
        "--output-address",
        "ipc:///tmp/output.sock",
        "--coordinator-address",
        "tcp://127.0.0.1:7000",
        "--engine-count",
        "2",
        "--args-json",
        r#"{"model_tag":"Qwen/Qwen3-0.6B"}"#,
    ])
    .unwrap();

    let Command::Frontend(args) = cli.command else {
        panic!("expected frontend args");
    };
    let config = args.into_config();

    expect![[r#"
        Config {
            transport_mode: Bootstrapped {
                input_address: "ipc:///tmp/input.sock",
                output_address: "ipc:///tmp/output.sock",
                engine_count: 2,
                ready_timeout: 300s,
            },
            coordinator_mode: External {
                address: "tcp://127.0.0.1:7000",
            },
            model: "Qwen/Qwen3-0.6B",
            listener_mode: InheritedFd {
                fd: 3,
            },
            tool_call_parser: None,
            reasoning_parser: None,
            enable_log_requests: false,
            disable_log_stats: false,
        }
    "#]]
    .assert_debug_eq(&config);
}
