use std::collections::HashSet;
use std::ffi::OsString;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory};

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

    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main
    /// serve ...`.
    ///
    /// Arguments after an explicit `--` are forwarded verbatim. Before `--`,
    /// `vllm-rs serve` automatically keeps recognized frontend options on
    /// the Rust side and forwards everything else to Python.
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
        max_model_len: Option<u32>,
        handshake_port: u16,
    ) -> ManagedEngineConfig {
        let mut python_args = self.python_args;
        // Manually forward some args to the Python engine.
        if let Some(max_model_len) = max_model_len {
            python_args.push("--max-model-len".to_string());
            python_args.push(max_model_len.to_string());
        }
        if let Some(data_parallel_size_local) = self.data_parallel_size_local {
            python_args.push("--data-parallel-size-local".to_string());
            python_args.push(data_parallel_size_local.to_string());
        }

        ManagedEngineConfig {
            python: self.python,
            model,
            handshake_host: self.handshake_host,
            handshake_port,
            data_parallel_size: self.data_parallel_size,
            python_args,
        }
    }

    /// Return the number of engines that the Rust frontend should expect to
    /// coordinate with.
    fn local_engine_count(&self) -> usize {
        self.data_parallel_size_local.unwrap_or(self.data_parallel_size)
    }

    /// Return whether the managed Rust frontend only needs to communicate with
    /// colocated engines.
    pub fn frontend_local_only(&self) -> bool {
        self.data_parallel_size_local != Some(0)
            && self.local_engine_count() == self.data_parallel_size
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
    let normalized_front_args = normalize_python_arg_aliases(front_args);

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
    PYTHON_MULTI_CHAR_ALIASES.iter().find_map(|&(alias, canonical)| {
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
