// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::net::TcpListener;

use crate::error::Result;
use crate::protocol::handshake::DataParallelRendezvous;

/// Ports listed besides the master port, as many as Python's `ParallelConfig`
/// draws.
const LISTED_PORT_COUNT: usize = 4;

impl DataParallelRendezvous {
    /// Allocate the rendezvous of all `data_parallel_size` ranks on the
    /// handshake host.
    ///
    /// Each launcher's `ParallelConfig` draws its own ports, so engines spawned
    /// by different launchers (another node, or a second launcher on the same
    /// node) only agree on them when the handshake owner sends one set to all
    /// of them. The ports are bound and released here, on the host the engines
    /// dial.
    pub(super) fn allocate(host: &str, data_parallel_size: usize) -> Result<Self> {
        // ZMQ endpoints spell IPv6 hosts in brackets; sockets and torch take
        // the bare address.
        let master_ip =
            host.strip_prefix('[').and_then(|host| host.strip_suffix(']')).unwrap_or(host);

        // Keep every listener bound until all ports are read so that none
        // repeats.
        let master = TcpListener::bind((master_ip, 0))?;
        let listed = (0..LISTED_PORT_COUNT)
            .map(|_| TcpListener::bind((master_ip, 0)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let port = |listener: &TcpListener| listener.local_addr().map(|addr| addr.port());

        Ok(Self {
            data_parallel_master_ip: master_ip.to_string(),
            data_parallel_master_port: port(&master)?,
            data_parallel_master_port_list: listed
                .iter()
                .map(port)
                .collect::<std::io::Result<_>>()?,
            data_parallel_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use crate::protocol::handshake::DataParallelRendezvous;

    #[test]
    fn allocate_sends_bare_ipv6_master_ip() {
        if TcpListener::bind(("::1", 0)).is_err() {
            eprintln!("skipping: IPv6 loopback is unavailable");
            return;
        }

        let rendezvous = DataParallelRendezvous::allocate("[::1]", 2).unwrap();

        assert_eq!(rendezvous.data_parallel_master_ip, "::1");
    }
}
