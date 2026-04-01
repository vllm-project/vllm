use expect_test::expect;

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
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        runtime: SharedRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_count: 1,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: Some(
                                512,
                            ),
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
fn serve_args_reject_python_flags_without_separator() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--python",
        "python3",
        "--dtype",
        "float16",
    ])
    .unwrap_err();

    expect![[r#"
            error: unrecognized serve argument "--dtype"

            This may be a flag the Rust frontend does not support yet, or a Python vLLM engine flag.
            If it is a Python engine flag, pass it after `--`, for example:
                vllm-rs serve <model> -- --dtype

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_python_multi_char_engine_alias_without_separator() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "-tp", "2"]).unwrap_err();

    expect![[r#"
            error: unrecognized serve argument "--tensor-parallel-size"

            This may be a flag the Rust frontend does not support yet, or a Python vLLM engine flag.
            If it is a Python engine flag, pass it after `--`, for example:
                vllm-rs serve <model> -- --tensor-parallel-size

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

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
        "--tokenizer-mode",
        "auto",
    ])
    .unwrap_err();

    expect![[r#"
            error: invalid value 'auto' for '--tokenizer-mode <TOKENIZER_MODE>': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_flag_arg() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--trust-remote-code"])
        .unwrap_err();

    expect![[r#"
            error: invalid value 'true' for '--trust-remote-code [<TRUST_REMOTE_CODE>]': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_no_flag_alias() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--no-enable-lora"])
        .unwrap_err();

    expect![[r#"
            error: invalid value 'true' for '--enable-lora [<ENABLE_LORA>]': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_bare_hf_token() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--hf-token"]).unwrap_err();

    expect![[r#"
            error: invalid value 'true' for '--hf-token [<HF_TOKEN>]': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_server_value_arg() {
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

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_server_flag_arg() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--allow-credentials"])
        .unwrap_err();

    expect![[r#"
            error: invalid value 'true' for '--allow-credentials [<ALLOW_CREDENTIALS>]': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_server_no_flag_alias() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--no-enable-log-deltas",
    ])
    .unwrap_err();

    expect![[r#"
            error: invalid value 'true' for '--enable-log-deltas [<ENABLE_LOG_DELTAS>]': argument is not implemented in Rust frontend yet

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn frontend_args_accept_engine_count() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "frontend",
        "Qwen/Qwen3-0.6B",
        "--handshake-address",
        "tcp://127.0.0.1:62100",
        "--engine-count",
        "2",
    ])
    .unwrap();

    expect![[r#"
        Cli {
            command: Frontend(
                FrontendArgs {
                    advertised_host: "127.0.0.1",
                    handshake_address: "tcp://127.0.0.1:62100",
                    runtime: SharedRuntimeArgs {
                        model: "Qwen/Qwen3-0.6B",
                        host: "127.0.0.1",
                        port: 8000,
                        engine_count: 2,
                        ready_timeout_secs: 300,
                        tool_call_parser: None,
                        reasoning_parser: None,
                        max_model_len: None,
                    },
                },
            ),
        }
    "#]]
    .assert_debug_eq(&cli);
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
        "--engine-count",
        "4",
    ])
    .unwrap();

    expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "10.99.48.128",
                        handshake_port: Some(
                            13345,
                        ),
                        runtime: SharedRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_count: 4,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
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
    assert_eq!(args.runtime.engine_count, 4);
}

#[test]
fn serve_args_accept_known_flags_before_model() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "--python",
        "python3",
        "--data-parallel-size",
        "2",
        "Qwen/Qwen3-0.6B",
    ])
    .unwrap();

    expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        runtime: SharedRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_count: 2,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                        },
                        python_args: [],
                    },
                ),
            }
        "#]]
    .assert_debug_eq(&cli);
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

    expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        runtime: SharedRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_count: 1,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                        },
                        python_args: [
                            "--tensor-parallel-size",
                            "2",
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

    expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        runtime: SharedRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_count: 1,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                        },
                        python_args: [
                            "-tp",
                            "2",
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
fn serve_args_reject_rust_long_flag_after_separator() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--",
        "--port",
        "9123",
    ])
    .unwrap_err();

    expect![[r#"
            error: misplaced serve argument "--port" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--port" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_rust_short_flag_after_separator() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-h"]).unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-h" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "-h" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_python_multi_char_alias_for_rust_flag_after_separator() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-dp", "4"]).unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-dp" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--data-parallel-size" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_python_multi_char_alias_for_unsupported_value_arg_after_separator() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-dpn", "1"])
        .unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-dpn" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--data-parallel-rank" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
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
fn serve_args_reject_python_multi_char_alias_for_unsupported_flag_after_separator() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-dph"]).unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-dph" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--data-parallel-hybrid-lb" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_python_multi_char_alias_for_unsupported_external_lb_after_separator() {
    let error =
        Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-dpe"]).unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-dpe" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--data-parallel-external-lb" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_unsupported_server_arg_after_separator() {
    let error = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--",
        "--uds",
        "/tmp/vllm.sock",
    ])
    .unwrap_err();

    expect![[r#"
            error: misplaced serve argument "--uds" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--uds" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_args_reject_python_multi_char_alias_for_serve_only_arg_after_separator() {
    let error = Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--", "-asc", "2"])
        .unwrap_err();

    expect![[r#"
            error: misplaced serve argument "-asc" after `--`

            Arguments after `--` are forwarded directly to the managed Python `vllm serve --headless` process.
            Use "--api-server-count" before `--` to configure `vllm-rs serve`.

            Usage: serve [OPTIONS] <MODEL> [-- <PYTHON_ARGS>...]

            For more information, try '--help'.
        "#]]
    .assert_eq(&error.to_string());
}

#[test]
fn serve_frontend_config_uses_dp_address_for_both_handshake_and_transport_host() {
    let cli = Cli::try_parse_from([
        "vllm-rs",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--handshake-host",
        "10.99.48.128",
        "--engine-count",
        "4",
    ])
    .unwrap();

    let Command::Serve(args) = cli.command else {
        panic!("expected serve args");
    };
    let config = args.to_frontend_config("tcp://10.99.48.128:29550".to_string());

    expect![[r#"
            Config {
                handshake_address: "tcp://10.99.48.128:29550",
                engine_count: 4,
                model: "Qwen/Qwen3-0.6B",
                host: "127.0.0.1",
                port: 8000,
                advertised_host: "10.99.48.128",
                ready_timeout: 300s,
                tool_call_parser: None,
                reasoning_parser: None,
                max_model_len: None,
            }
        "#]]
    .assert_debug_eq(&config);
}
