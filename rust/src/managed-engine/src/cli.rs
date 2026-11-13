// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::ffi::OsString;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory};
use serde_json::{Map, Value};

use crate::data_parallel::DataParallelTopology;
use crate::{ManagedEngineConfig, allocate_handshake_port};

/// Managed Python headless-engine CLI arguments.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct ManagedEngineArgs {
    /// Python executable used to launch the managed headless vLLM engine.
    #[arg(long, env = "VLLM_RS_PYTHON", default_value = "python3")]
    pub python: String,
    /// Host/IP used both for the managed-engine handshake endpoint and the
    /// frontend-advertised input/output ZMQ socket addresses.
    #[arg(
        long = "data-parallel-address",
        visible_alias = "handshake-host",
        default_value = "127.0.0.1"
    )]
    pub handshake_host: String,
    /// Optional TCP port for the managed-engine handshake / data-parallel RPC
    /// endpoint.
    ///
    /// When omitted, the CLI allocates an ephemeral port automatically.
    #[arg(
        long = "data-parallel-rpc-port",
        visible_alias = "handshake-port",
        value_parser = clap::value_parser!(u16).range(1..)
    )]
    pub handshake_port: Option<u16>,
    /// Number of data parallel replicas across the whole deployment.
    #[arg(long, default_value_t = 1)]
    pub data_parallel_size: usize,
    /// Number of data parallel replicas to run on this node.
    #[arg(long)]
    pub data_parallel_size_local: Option<usize>,
    /// Maximum model context length forwarded to the managed Python engine.
    ///
    /// Rust leaves validation to Python so values such as `auto` and
    /// human-readable integers retain their engine-owned semantics.
    #[arg(long)]
    pub max_model_len: Option<String>,

    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main
    /// serve ...`.
    ///
    /// Arguments after an explicit `--` are forwarded verbatim. Before `--`,
    /// `vllm-rs serve` automatically keeps recognized frontend options on
    /// the Rust side and forwards everything else to Python.
    ///
    /// The explicit `--` passthrough is a last-resort escape hatch. Rust does
    /// not interpret, validate, or de-duplicate those arguments against
    /// managed-engine arguments that it appends later; if the same Python flag
    /// appears more than once, Python argparse owns the final result.
    #[arg(
        last = true,
        allow_hyphen_values = true,
        help_heading = "Passthrough arguments"
    )]
    pub python_args: Vec<String>,
}

impl ManagedEngineArgs {
    /// Build the handshake address shared by the Rust frontend and managed
    /// Python engine.
    pub fn handshake_address(&self, handshake_port: u16) -> String {
        format!("tcp://{}:{}", self.handshake_host, handshake_port)
    }

    /// Resolve the handshake port, either from the CLI argument (if specified)
    /// or by allocating a fresh port.
    pub fn resolve_handshake_port(&self) -> anyhow::Result<u16> {
        self.handshake_port
            .map(Ok)
            .unwrap_or_else(|| allocate_handshake_port(&self.handshake_host))
    }

    /// Build the managed Python-engine spawn configuration.
    pub fn into_config(
        self,
        model: String,
        max_logprobs: Option<i32>,
        profiler_config: Option<String>,
        reasoning_parser: Option<&str>,
        language_model_only: bool,
        disable_log_stats: bool,
        shutdown_timeout: u64,
        handshake_port: u16,
        limit_mm_per_prompt: Option<String>,
        hf_overrides: Option<String>,
        structured_outputs_config: Option<String>,
        topology: &DataParallelTopology,
        client_handshake_address: Option<String>,
    ) -> ManagedEngineConfig {
        let mut python_args = self.python_args;
        // Manually forward some args to the Python engine.
        if let Some(max_model_len) = self.max_model_len {
            python_args.push("--max-model-len".to_string());
            python_args.push(max_model_len);
        }
        if let Some(max_logprobs) = max_logprobs {
            python_args.push("--max-logprobs".to_string());
            python_args.push(max_logprobs.to_string());
        }
        if let Some(profiler_config) = profiler_config {
            python_args.push("--profiler-config".to_string());
            python_args.push(profiler_config);
        }
        if let Some(reasoning_parser) = reasoning_parser {
            python_args.push("--reasoning-parser".to_string());
            python_args.push(reasoning_parser.to_string());
        }
        if language_model_only {
            python_args.push("--language-model-only".to_string());
        }
        if disable_log_stats {
            python_args.push("--disable-log-stats".to_string());
        }
        // we must pass through shutdown_timeout to the engine,
        // otherwise inflight requests get aborted on shutdown
        if shutdown_timeout > 0 {
            python_args.push("--shutdown-timeout".to_string());
            python_args.push(shutdown_timeout.to_string());
        }
        // This node's engines launch as a headless internal-LB node; hybrid
        // and external LB differ only in which frontend they handshake with,
        // which `client_handshake_address` carries.
        if self.data_parallel_size_local.is_some() || topology.local_engines_only {
            python_args.push("--data-parallel-size-local".to_string());
            python_args.push(topology.local_size.to_string());
        }
        if topology.start_rank != 0 {
            python_args.push("--data-parallel-start-rank".to_string());
            python_args.push(topology.start_rank.to_string());
        }
        if let Some(limit_mm_per_prompt) = limit_mm_per_prompt {
            python_args.push("--limit-mm-per-prompt".to_string());
            python_args.push(limit_mm_per_prompt);
        }
        if let Some(hf_overrides) = hf_overrides {
            python_args.push("--hf-overrides".to_string());
            python_args.push(hf_overrides);
        }
        if let Some(structured_outputs_config) = structured_outputs_config {
            python_args.push("--structured-outputs-config".to_string());
            python_args.push(structured_outputs_config);
        }

        ManagedEngineConfig {
            python: self.python,
            model,
            handshake_host: self.handshake_host,
            handshake_port,
            data_parallel_size: self.data_parallel_size,
            client_handshake_address,
            python_args,
        }
    }
}

/// Python `argparse` accepts these multi-character single-dash aliases, but
/// `clap` cannot model them directly.
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

/// Repartition managed-engine argv so Rust-owned flags stay before `--`, while
/// everything else is forwarded to Python.
pub fn repartition_managed_engine_args<C>(
    args: &[OsString],
    subcommand: Option<&str>,
) -> Result<Vec<OsString>, clap::Error>
where
    C: CommandFactory,
{
    let command = C::command();
    let (prefix, real_args, command) = match subcommand {
        Some(subcommand) => {
            if !matches_subcommand(args, subcommand) {
                return Ok(args.to_vec());
            };

            let subcommand = command
                .find_subcommand(subcommand)
                .expect("managed-engine subcommand should exist");

            (args[..2].to_vec(), &args[2..], subcommand)
        }
        None => {
            let Some(program) = args.first() else {
                return Ok(args.to_vec());
            };

            (vec![program.clone()], &args[1..], &command)
        }
    };

    let mut repartitioned = prefix;
    repartitioned.extend(repartition_real_managed_engine_args(real_args, command)?);
    Ok(repartitioned)
}

fn repartition_real_managed_engine_args(
    args: &[OsString],
    command: &clap::Command,
) -> Result<Vec<OsString>, clap::Error> {
    let Some(model) = args.first() else {
        return Ok(args.to_vec());
    };

    let model = model.to_string_lossy();
    if is_help_flag(&model) {
        return Ok(args.to_vec());
    }
    if model == "--" || is_option_like(&model) {
        return Err(build_missing_model_error(command));
    }

    let (long_flags, short_flags) = collect_option_names(command);
    let (front_args, explicit_passthrough, had_separator) = split_managed_engine_args(&args[1..]);
    let normalized_front_args = merge_dotted_frontend_args(
        normalize_python_arg_aliases(front_args),
        &long_flags,
        command,
    )?;

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

    let mut repartitioned = vec![args[0].clone()];
    repartitioned.extend(frontend_chunks);
    if had_separator || !python_chunks.is_empty() || !explicit_passthrough.is_empty() {
        repartitioned.push("--".into());
        repartitioned.extend(python_chunks);
        repartitioned.extend(explicit_passthrough.iter().cloned());
    }

    Ok(repartitioned)
}

fn matches_subcommand(args: &[OsString], subcommand: &str) -> bool {
    args.get(1)
        .and_then(|arg| arg.to_str())
        .is_some_and(|candidate| candidate == subcommand)
}

fn split_managed_engine_args(args: &[OsString]) -> (&[OsString], &[OsString], bool) {
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
                .or_else(|| normalize_python_long_flag_name(&text))
                .map(Into::into)
                .unwrap_or_else(|| arg.clone())
        })
        .collect()
}

/// Python's `FlexibleArgumentParser` accepts underscores in long flag names
/// (`--served_model_name`), replacing them with dashes up to the first `.` or
/// `=`, so keys of dotted config forms keep their underscores.
fn normalize_python_long_flag_name(arg: &str) -> Option<String> {
    let rest = arg.strip_prefix("--")?;
    let (name, tail) = rest.split_at(rest.find(['.', '=']).unwrap_or(rest.len()));
    name.contains('_').then(|| format!("--{}{tail}", name.replace('_', "-")))
}

fn normalize_python_multi_char_alias(arg: &str) -> Option<String> {
    find_python_multi_char_alias(arg).map(|canonical| match arg.split_once('=') {
        Some((_, value)) => format!("{canonical}={value}"),
        None => canonical.to_string(),
    })
}

fn find_python_multi_char_alias(arg: &str) -> Option<&'static str> {
    PYTHON_MULTI_CHAR_ALIASES.iter().find_map(|&(alias, canonical)| {
        (arg == alias || arg.starts_with(&format!("{alias}="))).then_some(canonical)
    })
}

/// Python's `FlexibleArgumentParser` turns `--<flag>.<key>[.<key>...] <value>`
/// into one `--<flag> <JSON object>` per flag, appended after the other
/// arguments. Do the same for flags `vllm-rs serve` declares, so their dotted
/// forms reach clap instead of only the managed engine; the dotted forms of
/// every other flag go to Python unchanged, which expands them itself.
fn merge_dotted_frontend_args(
    args: Vec<OsString>,
    long_flags: &HashSet<String>,
    command: &clap::Command,
) -> Result<Vec<OsString>, clap::Error> {
    let mut merged = Map::new();
    let mut kept = Vec::with_capacity(args.len());
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy().into_owned();
        let Some((flag, path, inline_value)) =
            split_dotted_flag(&text).filter(|(flag, ..)| long_flags.contains(*flag))
        else {
            kept.push(arg);
            continue;
        };
        let value = match inline_value {
            Some(value) => value.to_owned(),
            None => {
                args.next().map(|value| value.to_string_lossy().into_owned()).ok_or_else(|| {
                    command.clone().error(
                        ErrorKind::InvalidValue,
                        format!("a value is required for '{text}'"),
                    )
                })?
            }
        };
        // `--<flag>.<key>+ a,b` appends list elements.
        let (path, value) = match path.strip_suffix('+') {
            Some(path) => (path, Value::from(value.split(',').collect::<Vec<_>>())),
            None => (path, parse_dotted_value(&value)),
        };
        let nested = path.rsplit('.').fold(value, |value, key| {
            Value::Object(Map::from_iter([(key.to_owned(), value)]))
        });
        let existing = merged.entry(flag).or_insert_with(|| Value::Object(Map::new()));
        merge_json(existing, nested);
    }
    for (flag, value) in merged {
        kept.push(format!("--{flag}").into());
        kept.push(value.to_string().into());
    }
    Ok(kept)
}

/// Split `--<flag>.<path>[=<value>]` into the flag name, the dotted key path,
/// and the inline value.
fn split_dotted_flag(arg: &str) -> Option<(&str, &str, Option<&str>)> {
    let rest = arg.strip_prefix("--")?;
    let (key, value) = match rest.split_once('=') {
        Some((key, value)) => (key, Some(value)),
        None => (rest, None),
    };
    let (flag, path) = key.split_once('.')?;
    Some((flag, path, value))
}

/// Python reads a dotted value as JSON, then as a human-readable integer, and
/// otherwise keeps it as a string.
fn parse_dotted_value(value: &str) -> Value {
    serde_json::from_str(value)
        .ok()
        .or_else(|| parse_human_readable_int(value).map(Value::from))
        .unwrap_or_else(|| Value::from(value))
}

/// Python's `human_readable_int`: `1k` is 1,000, `1K` is 1,024 and `25.6k` is
/// 25,600; a binary suffix takes no decimals.
fn parse_human_readable_int(value: &str) -> Option<i64> {
    let value = value.trim();
    let suffixed = value.char_indices().last().and_then(|(index, suffix)| {
        let multiplier = match suffix {
            'k' => 1_000,
            'm' => 1_000_000,
            'g' => 1_000_000_000,
            't' => 1_000_000_000_000,
            'K' => 1 << 10,
            'M' => 1 << 20,
            'G' => 1 << 30,
            'T' => 1 << 40,
            _ => return None,
        };
        let number = &value[..index];
        let is_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        let well_formed = match number.split_once('.') {
            Some((int, frac)) => is_digits(int) && is_digits(frac),
            None => is_digits(number),
        };
        well_formed.then_some((number, multiplier, suffix.is_ascii_uppercase()))
    });
    match suffixed {
        Some((number, multiplier, true)) => number.parse::<i64>().ok()?.checked_mul(multiplier),
        Some((number, multiplier, false)) => {
            Some((number.parse::<f64>().ok()? * multiplier as f64) as i64)
        }
        None => value.parse().ok(),
    }
}

/// Python's `recursive_dict_update`: objects merge key by key, lists
/// concatenate, and any other value is replaced.
fn merge_json(original: &mut Value, update: Value) {
    match (original, update) {
        (Value::Object(original), Value::Object(update)) => {
            for (key, value) in update {
                match original.get_mut(&key) {
                    Some(existing) => merge_json(existing, value),
                    None => {
                        original.insert(key, value);
                    }
                }
            }
        }
        (Value::Array(original), Value::Array(update)) => original.extend(update),
        (original, update) => *original = update,
    }
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

fn collect_option_names(command: &clap::Command) -> (HashSet<String>, HashSet<char>) {
    let mut long_flags = HashSet::new();
    let mut short_flags = HashSet::new();
    for arg in command.get_arguments() {
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

fn build_missing_model_error(command: &clap::Command) -> clap::Error {
    command.clone().error(
        ErrorKind::MissingRequiredArgument,
        "the model must appear immediately after the command",
    )
}
