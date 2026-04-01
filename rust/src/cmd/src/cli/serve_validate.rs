use std::collections::HashSet;
use std::ffi::OsString;

use clap::CommandFactory as _;
use clap::error::ErrorKind;

use crate::cli::Cli;

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

/// Repartition `serve` argv so Rust frontend-owned flags stay before `--`, while everything else
/// is forwarded to Python.
pub(super) fn repartition_serve_args(args: &[OsString]) -> Result<Vec<OsString>, clap::Error> {
    if args.get(1).map(|arg| arg.as_os_str()) != Some("serve".as_ref()) {
        return Ok(args.to_vec());
    }

    if args.get(2).is_none() {
        return Ok(args.to_vec());
    }

    let model = args[2].to_string_lossy();
    if is_help_flag(&model) {
        return Ok(args.to_vec());
    }
    if model == "--" || is_option_like(&model) {
        return Err(build_missing_model_error());
    }

    let (front_args, explicit_passthrough, had_separator) = split_serve_args(&args[3..]);
    let normalized_front_args = normalize_python_arg_aliases(front_args);
    let (long_flags, short_flags) = collect_frontend_option_names();

    let mut frontend_chunks = Vec::new();
    let mut python_chunks = Vec::new();
    let mut current_chunk = Vec::new();

    for arg in normalized_front_args {
        let text = arg.to_string_lossy();
        if is_option_like(&text) && !current_chunk.is_empty() {
            push_chunk(
                &mut frontend_chunks,
                &mut python_chunks,
                std::mem::take(&mut current_chunk),
                &long_flags,
                &short_flags,
            );
        }
        current_chunk.push(arg);
    }
    if !current_chunk.is_empty() {
        push_chunk(
            &mut frontend_chunks,
            &mut python_chunks,
            current_chunk,
            &long_flags,
            &short_flags,
        );
    }

    let mut repartitioned = vec![args[0].clone(), args[1].clone(), args[2].clone()];
    repartitioned.extend(frontend_chunks);
    if had_separator || !python_chunks.is_empty() || !explicit_passthrough.is_empty() {
        repartitioned.push("--".into());
        repartitioned.extend(python_chunks);
        repartitioned.extend(explicit_passthrough.iter().cloned());
    }

    Ok(repartitioned)
}

fn split_serve_args(args: &[OsString]) -> (&[OsString], &[OsString], bool) {
    if let Some(index) = args.iter().position(|arg| arg == "--") {
        (&args[..index], &args[index + 1..], true)
    } else {
        (args, &[], false)
    }
}

fn normalize_python_arg_aliases(args: &[OsString]) -> Vec<OsString> {
    args.iter()
        .map(|arg| {
            let text = arg.to_string_lossy();
            normalize_python_multi_char_alias(&text)
                .map(Into::into)
                .unwrap_or_else(|| arg.clone())
        })
        .collect()
}

fn normalize_python_multi_char_alias(arg: &str) -> Option<String> {
    find_python_multi_char_alias(arg).map(|canonical| match arg.split_once('=') {
        Some((_, value)) => format!("{canonical}={value}"),
        None => canonical.to_string(),
    })
}

fn find_python_multi_char_alias(arg: &str) -> Option<&'static str> {
    PYTHON_MULTI_CHAR_ALIASES
        .iter()
        .find_map(|&(alias, canonical)| {
            (arg == alias || arg.starts_with(&format!("{alias}="))).then_some(canonical)
        })
}

fn push_chunk(
    frontend_chunks: &mut Vec<OsString>,
    python_chunks: &mut Vec<OsString>,
    chunk: Vec<OsString>,
    long_flags: &HashSet<String>,
    short_flags: &HashSet<char>,
) {
    if chunk_head_is_frontend_owned(&chunk, long_flags, short_flags) {
        frontend_chunks.extend(chunk);
    } else {
        python_chunks.extend(chunk);
    }
}

fn chunk_head_is_frontend_owned(
    chunk: &[OsString],
    long_flags: &HashSet<String>,
    short_flags: &HashSet<char>,
) -> bool {
    let Some(head) = chunk.first() else {
        return false;
    };
    let head = head.to_string_lossy();

    if let Some(rest) = head.strip_prefix("--") {
        let name = rest.split_once('=').map_or(rest, |(name, _)| name);
        return long_flags.contains(name);
    }

    let Some(rest) = head.strip_prefix('-') else {
        return false;
    };
    let Some(short) = rest.chars().next() else {
        return false;
    };
    short_flags.contains(&short)
}

fn collect_frontend_option_names() -> (HashSet<String>, HashSet<char>) {
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

    long_flags.insert("help".to_string());
    short_flags.insert('h');

    (long_flags, short_flags)
}

fn is_option_like(arg: &str) -> bool {
    if arg == "--" {
        return false;
    }

    if let Some(rest) = arg.strip_prefix("--") {
        return rest.chars().next().is_some_and(char::is_alphabetic);
    }

    if let Some(rest) = arg.strip_prefix('-') {
        return rest.chars().next().is_some_and(char::is_alphabetic);
    }

    false
}

fn is_help_flag(arg: &str) -> bool {
    arg == "-h" || arg == "--help"
}

fn build_missing_model_error() -> clap::Error {
    let mut command = Cli::command();
    let serve_command = command
        .find_subcommand_mut("serve")
        .expect("serve subcommand should exist");
    serve_command.error(
        ErrorKind::MissingRequiredArgument,
        "serve requires the model to appear immediately after the subcommand",
    )
}
