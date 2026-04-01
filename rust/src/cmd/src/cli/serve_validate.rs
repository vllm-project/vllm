use std::collections::HashSet;
use std::ffi::OsString;

use clap::CommandFactory as _;
use clap::error::{ContextKind, ContextValue, ErrorKind};

use crate::cli::{Cli, Command};

/// Python `argparse` accepts these multi-character single-dash aliases, but `clap` cannot model
/// them directly.
const PYTHON_MULTI_CHAR_ALIASES: &[(&str, &str)] = &[
    ("-asc", "--api-server-count"),
    ("-pp", "--pipeline-parallel-size"),
    ("-tp", "--tensor-parallel-size"),
    ("-dcp", "--decode-context-parallel-size"),
    ("-pcp", "--prefill-context-parallel-size"),
    ("-dp", "--data-parallel-size"),
    ("-dpn", "--data-parallel-rank"),
    ("-dpr", "--data-parallel-start-rank"),
    ("-dpl", "--data-parallel-size-local"),
    ("-dpa", "--data-parallel-address"),
    ("-dpp", "--data-parallel-rpc-port"),
    ("-dpb", "--data-parallel-backend"),
    ("-dph", "--data-parallel-hybrid-lb"),
    ("-dpe", "--data-parallel-external-lb"),
    ("-ep", "--enable-expert-parallel"),
    ("-cc", "--compilation-config"),
    ("-ac", "--attention-config"),
];

/// Normalize Python-only multi-character single-dash aliases before they reach `clap`.
///
/// This only rewrites arguments before `--`. Passthrough arguments after `--` must stay byte-for-
/// byte unchanged so they can be forwarded directly to Python.
pub(super) fn normalize_python_arg_aliases(args: &[OsString]) -> Vec<OsString> {
    let mut normalized = Vec::with_capacity(args.len());
    let mut seen_serve = false;
    let mut in_passthrough = false;

    for (index, arg) in args.iter().enumerate() {
        if index == 0 {
            normalized.push(arg.clone());
            continue;
        }

        let text = arg.to_string_lossy();
        if !seen_serve {
            if text == "serve" {
                seen_serve = true;
            }
            normalized.push(arg.clone());
            continue;
        }

        if in_passthrough {
            normalized.push(arg.clone());
            continue;
        }

        if text == "--" {
            in_passthrough = true;
            normalized.push(arg.clone());
            continue;
        }

        if let Some(canonical) = normalize_python_multi_char_alias(&text) {
            normalized.push(canonical.into());
        } else {
            normalized.push(arg.clone());
        }
    }

    normalized
}

/// Rewrite clap errors about unknown arguments in the `serve` subcommand to clarify the `--`
/// separator for Python engine flags.
pub(super) fn rewrite_unknown_arg_error(args: &[OsString], error: clap::Error) -> clap::Error {
    if error.kind() != ErrorKind::UnknownArgument {
        return error;
    }
    let subcommand = args
        .iter()
        .skip(1)
        .find(|s| !s.to_string_lossy().starts_with('-'));
    if subcommand.map(|s| s.as_os_str()) != Some("serve".as_ref()) {
        return error;
    }
    let Some(ContextValue::String(unrecognized_arg)) = error.get(ContextKind::InvalidArg) else {
        return error;
    };

    let mut command = Cli::command();
    let serve_command = command
        .find_subcommand_mut("serve")
        .expect("serve subcommand should exist");
    serve_command.error(
        ErrorKind::UnknownArgument,
        format!(
            "unrecognized serve argument {unrecognized_arg:?}\n\n\
             This may be a flag the Rust frontend does not support yet, or a Python vLLM engine \
             flag.\nIf it is a Python engine flag, pass it after `--`, for example:\n    \
             vllm-rs serve <model> -- {unrecognized_arg}"
        ),
    )
}

/// Reject Rust-side `serve` flags when users accidentally place them after `--`.
pub(super) fn validate_passthrough_args(cli: Cli) -> Result<Cli, clap::Error> {
    let Command::Serve(args) = &cli.command else {
        return Ok(cli);
    };

    let Some((arg, canonical)) = find_misplaced_passthrough_arg(&args.python_args) else {
        return Ok(cli);
    };

    Err(build_misplaced_passthrough_arg_error(arg, canonical))
}

/// Find the first passthrough token that is actually a Rust-side `serve` option.
fn find_misplaced_passthrough_arg(python_args: &[String]) -> Option<(String, String)> {
    let (long_flags, short_flags) = collect_option_names();
    for arg in python_args {
        if let Some(rest) = arg.strip_prefix("--") {
            let name = rest.split_once('=').map_or(rest, |(name, _)| name);
            if long_flags.contains(name) {
                let canonical = format!("--{name}");
                return Some((canonical.clone(), canonical));
            }
            continue;
        }

        if let Some(canonical) = find_python_multi_char_alias(arg) {
            let name = canonical
                .strip_prefix("--")
                .expect("canonical alias should be a long option");
            if long_flags.contains(name) {
                return Some((arg.to_string(), canonical.to_string()));
            }
            continue;
        }

        let Some(rest) = arg.strip_prefix('-') else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }

        let Some(short) = rest.chars().next() else {
            continue;
        };
        if short_flags.contains(&short) {
            let canonical = format!("-{short}");
            return Some((canonical.clone(), canonical));
        }
    }

    None
}

fn normalize_python_multi_char_alias(arg: &str) -> Option<String> {
    find_python_multi_char_alias(arg).map(|canonical| match arg.split_once('=') {
        Some((_, value)) => format!("{canonical}={value}"),
        None => canonical.to_string(),
    })
}

/// Match Python-only multi-character single-dash aliases.
fn find_python_multi_char_alias(arg: &str) -> Option<&'static str> {
    PYTHON_MULTI_CHAR_ALIASES
        .iter()
        .find_map(|&(alias, canonical)| {
            (arg == alias || arg.starts_with(&format!("{alias}="))).then_some(canonical)
        })
}

/// Collect all long/short option names recognized by the Rust `serve` subcommand.
fn collect_option_names() -> (HashSet<String>, HashSet<char>) {
    let mut command = Cli::command();
    let serve_command = command
        .find_subcommand_mut("serve")
        .expect("serve subcommand should exist");

    let mut long_flags = HashSet::new();
    let mut short_flags = HashSet::new();
    for arg in serve_command.get_arguments() {
        if let Some(names) = arg.get_long_and_visible_aliases() {
            long_flags.extend(names.into_iter().map(str::to_owned));
        }
        if let Some(short) = arg.get_short() {
            short_flags.insert(short);
        }
        if let Some(short_aliases) = arg.get_visible_short_aliases() {
            short_flags.extend(short_aliases);
        }
    }

    // Clap injects built-in help flags separately from user-defined arguments.
    long_flags.insert("help".to_string());
    short_flags.insert('h');

    (long_flags, short_flags)
}

/// Build one targeted error for a Rust-side `serve` argument that was placed after `--`.
fn build_misplaced_passthrough_arg_error(arg: String, canonical: String) -> clap::Error {
    let mut command = Cli::command();
    let serve_command = command
        .find_subcommand_mut("serve")
        .expect("serve subcommand should exist");
    serve_command.error(
        ErrorKind::UnknownArgument,
        format!(
            "misplaced serve argument {arg:?} after `--`\n\n\
             Arguments after `--` are forwarded directly to the managed Python `vllm serve \
             --headless` process.\nUse {canonical:?} before `--` to configure `vllm-rs serve`."
        ),
    )
}
