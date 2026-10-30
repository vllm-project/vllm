// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use anyhow::bail;
use clap::Args;

use crate::cli::ManagedEngineArgs;

/// Data parallel ranks whose engines one `vllm-rs serve` process launches,
/// and which of them its frontend connects to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataParallelTopology {
    /// First data parallel rank of this node.
    pub start_rank: u32,
    /// Number of data parallel ranks whose engines this process launches.
    pub local_size: usize,
    /// Number of data parallel ranks across the whole deployment.
    pub data_parallel_size: usize,
    /// Whether the frontend connects only to this node's ranks (hybrid or
    /// external DP LB) while frontends on other nodes serve the rest. Otherwise
    /// one frontend connects to every rank (internal DP LB).
    pub local_engines_only: bool,
}

impl DataParallelTopology {
    /// Number of engines the frontend connects to.
    pub fn frontend_engine_count(&self) -> usize {
        if self.local_engines_only {
            self.local_size
        } else {
            self.data_parallel_size
        }
    }

    /// Whether every engine the frontend connects to runs on this node.
    pub fn frontend_local_only(&self) -> bool {
        self.local_size != 0
            && (self.local_engines_only || self.local_size == self.data_parallel_size)
    }

    /// First data parallel rank the frontend connects to. The frontend whose
    /// range starts at rank 0 owns the deployment's handshake.
    pub fn frontend_engine_start_index(&self) -> u32 {
        if self.local_engines_only {
            self.start_rank
        } else {
            0
        }
    }
}

/// Data parallel load-balancing arguments of `vllm-rs serve`: which ranks this
/// node serves and whether its frontend serves only them.
#[derive(Debug, Clone, Default, Args, PartialEq, Eq)]
pub struct DataParallelLbArgs {
    /// Rank of the data parallel group served by this process. Implies
    /// `--data-parallel-external-lb`.
    #[arg(long, env = "VLLM_DP_RANK")]
    pub data_parallel_rank: Option<u32>,
    /// First data parallel rank on this node. Implies
    /// `--data-parallel-hybrid-lb` unless `--headless` is set.
    #[arg(long)]
    pub data_parallel_start_rank: Option<u32>,
    /// Whether to use "hybrid" DP LB mode: this node's frontend load
    /// balances between its local data parallel ranks, while an external LB
    /// balances between nodes. Set explicitly in conjunction with
    /// `--data-parallel-size-local` and `--data-parallel-start-rank`.
    #[arg(long)]
    pub data_parallel_hybrid_lb: bool,
    /// Whether to use "external" DP LB mode: every data parallel rank runs its
    /// own frontend, as in a "one-pod-per-rank" wide-EP setup in Kubernetes.
    /// Set implicitly when `--data-parallel-rank` is provided.
    #[arg(long)]
    pub data_parallel_external_lb: bool,
}

impl DataParallelLbArgs {
    /// Resolve the data parallel ranks of this process from the DP LB flags,
    /// which keep their `vllm serve` meaning.
    pub fn resolve(
        &self,
        engine: &ManagedEngineArgs,
        headless: bool,
    ) -> anyhow::Result<DataParallelTopology> {
        let data_parallel_size = engine.data_parallel_size;
        if self.data_parallel_hybrid_lb && self.data_parallel_external_lb {
            bail!(
                "`--data-parallel-hybrid-lb` and `--data-parallel-external-lb` cannot be enabled together"
            );
        }
        if headless && self.data_parallel_hybrid_lb {
            bail!("`--data-parallel-hybrid-lb` is not applicable with `--headless`");
        }
        if self.data_parallel_rank.is_some() && self.data_parallel_start_rank.is_some() {
            bail!("`--data-parallel-rank` and `--data-parallel-start-rank` cannot be set together");
        }

        let topology = if self.data_parallel_external_lb || self.data_parallel_rank.is_some() {
            let Some(rank) = self.data_parallel_rank else {
                bail!("`--data-parallel-external-lb` requires `--data-parallel-rank`");
            };
            if !matches!(engine.data_parallel_size_local, None | Some(1)) {
                bail!("an external data parallel rank requires `--data-parallel-size-local 1`");
            }
            DataParallelTopology {
                start_rank: rank,
                local_size: 1,
                data_parallel_size,
                local_engines_only: true,
            }
        } else if let Some(local_size) = engine.data_parallel_size_local {
            // A start rank outside `--headless` implies hybrid LB, and hybrid LB
            // with every rank on this node is internal LB. Hybrid LB with one
            // local rank is external LB, which connects the same way.
            let hybrid_lb = (self.data_parallel_hybrid_lb
                || (self.data_parallel_start_rank.is_some() && !headless))
                && local_size != data_parallel_size;
            DataParallelTopology {
                start_rank: self.data_parallel_start_rank.unwrap_or(0),
                local_size,
                data_parallel_size,
                local_engines_only: hybrid_lb,
            }
        } else {
            if self.data_parallel_hybrid_lb {
                bail!("`--data-parallel-hybrid-lb` requires `--data-parallel-size-local`");
            }
            // Without a local size, the start rank is ignored.
            DataParallelTopology {
                start_rank: 0,
                local_size: data_parallel_size,
                data_parallel_size,
                local_engines_only: false,
            }
        };

        if topology.start_rank as usize + topology.local_size > data_parallel_size {
            bail!(
                "data parallel ranks [{}, {}) of this node exceed `--data-parallel-size {data_parallel_size}`",
                topology.start_rank,
                topology.start_rank as usize + topology.local_size
            );
        }
        if topology.local_engines_only {
            if topology.local_size == 0 {
                bail!("hybrid and external DP LB require local data parallel ranks");
            }
            // Every node dials the handshake of the frontend owning rank 0.
            if engine.handshake_port.is_none() {
                bail!("hybrid and external DP LB require `--data-parallel-rpc-port`");
            }
        }
        Ok(topology)
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::DataParallelLbArgs;
    use crate::cli::ManagedEngineArgs;

    /// The `vllm-rs serve` flags that the topology is resolved from.
    #[derive(Parser)]
    struct ServeFlags {
        #[command(flatten)]
        engine: ManagedEngineArgs,
        #[command(flatten)]
        data_parallel_lb: DataParallelLbArgs,
        #[arg(long)]
        headless: bool,
    }

    #[test]
    fn resolve_follows_vllm_serve_dp_lb_modes() {
        let cases: &[&[&str]] = &[
            &["--data-parallel-size", "4"],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "2",
                "--data-parallel-rpc-port",
                "13345",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-hybrid-lb",
                "--data-parallel-rpc-port",
                "13345",
            ],
            &[
                "--headless",
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "2",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-rank",
                "3",
                "--data-parallel-rpc-port",
                "1",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-start-rank",
                "2",
            ],
            &["--data-parallel-size", "4", "--data-parallel-external-lb"],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-rank",
                "1",
                "--data-parallel-start-rank",
                "1",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-hybrid-lb",
                "--data-parallel-external-lb",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "2",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-rank",
                "3",
            ],
            &[
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "3",
                "--data-parallel-rpc-port",
                "1",
            ],
        ];
        let resolved = cases
            .iter()
            .map(|flags| {
                let parsed = ServeFlags::try_parse_from(["vllm-rs"].iter().chain(*flags)).unwrap();
                let result = match parsed.data_parallel_lb.resolve(&parsed.engine, parsed.headless)
                {
                    Ok(topology) => format!("{topology:?}"),
                    Err(error) => format!("error: {error}"),
                };
                format!("{}\n  {result}", flags.join(" "))
            })
            .collect::<Vec<_>>()
            .join("\n");

        expect_test::expect![[r#"
        --data-parallel-size 4
          DataParallelTopology { start_rank: 0, local_size: 4, data_parallel_size: 4, local_engines_only: false }
        --data-parallel-size 4 --data-parallel-size-local 2
          DataParallelTopology { start_rank: 0, local_size: 2, data_parallel_size: 4, local_engines_only: false }
        --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 --data-parallel-rpc-port 13345
          DataParallelTopology { start_rank: 2, local_size: 2, data_parallel_size: 4, local_engines_only: true }
        --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-hybrid-lb --data-parallel-rpc-port 13345
          DataParallelTopology { start_rank: 0, local_size: 2, data_parallel_size: 4, local_engines_only: true }
        --headless --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2
          DataParallelTopology { start_rank: 2, local_size: 2, data_parallel_size: 4, local_engines_only: false }
        --data-parallel-size 4 --data-parallel-rank 3 --data-parallel-rpc-port 1
          DataParallelTopology { start_rank: 3, local_size: 1, data_parallel_size: 4, local_engines_only: true }
        --data-parallel-size 4 --data-parallel-start-rank 2
          DataParallelTopology { start_rank: 0, local_size: 4, data_parallel_size: 4, local_engines_only: false }
        --data-parallel-size 4 --data-parallel-external-lb
          error: `--data-parallel-external-lb` requires `--data-parallel-rank`
        --data-parallel-size 4 --data-parallel-rank 1 --data-parallel-start-rank 1
          error: `--data-parallel-rank` and `--data-parallel-start-rank` cannot be set together
        --data-parallel-size 4 --data-parallel-hybrid-lb --data-parallel-external-lb
          error: `--data-parallel-hybrid-lb` and `--data-parallel-external-lb` cannot be enabled together
        --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2
          error: hybrid and external DP LB require `--data-parallel-rpc-port`
        --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-rank 3
          error: an external data parallel rank requires `--data-parallel-size-local 1`
        --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 3 --data-parallel-rpc-port 1
          error: data parallel ranks [3, 5) of this node exceed `--data-parallel-size 4`"#]].assert_eq(&resolved);
    }
}
