// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use ipnet::IpNet;

use crate::state::AppState;

/// Uvicorn's `FORWARDED_ALLOW_IPS` trust list, parsed once at startup.
pub(crate) struct TrustedProxies {
    networks: Vec<IpNet>,
    literals: HashSet<String>,
    always_trust: bool,
}

impl TrustedProxies {
    pub(crate) fn new(value: &str) -> Self {
        let mut networks = Vec::new();
        let mut literals = HashSet::new();
        for host in value.split(',').map(str::trim).filter(|host| !host.is_empty()) {
            if let Ok(ip) = host.parse::<IpAddr>() {
                networks.push(IpNet::from(ip));
            } else if let Some((address, _)) = host.split_once('/')
                && address.parse::<IpAddr>().is_ok()
                && let Ok(network) = host.parse::<IpNet>()
                && network.addr() == network.network()
            {
                networks.push(network);
            } else {
                literals.insert(host.to_owned());
            }
        }
        Self {
            networks,
            literals,
            always_trust: value == "*",
        }
    }

    fn contains_ip(&self, ip: IpAddr) -> bool {
        self.always_trust || self.networks.iter().any(|network| network.contains(&ip))
    }

    fn contains_host(&self, host: &[u8]) -> bool {
        self.always_trust
            || std::str::from_utf8(host).is_ok_and(|host| match parse_hop(host) {
                Some(address) => self.contains_ip(address.ip()),
                None => self.literals.contains(host),
            })
    }

    fn client_addr(&self, headers: &HeaderMap) -> Option<SocketAddr> {
        let mut hosts = headers
            .get_all("x-forwarded-for")
            .iter()
            .flat_map(|value| value.as_bytes().split(|byte| *byte == b','))
            .map(<[u8]>::trim_ascii);
        let first = hosts.next()?;
        let host = hosts.rfind(|host| !self.contains_host(host)).unwrap_or(first);
        // Do not skip an invalid hop: that could expose a spoofed address
        // farther left in the chain.
        parse_hop(std::str::from_utf8(host).ok()?)
    }
}

fn parse_hop(host: &str) -> Option<SocketAddr> {
    if let Ok(address) = host.parse::<SocketAddr>() {
        return Some(address);
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(SocketAddr::new(ip, 0));
    }
    let ip = host.strip_prefix('[')?.strip_suffix(']')?.parse::<Ipv6Addr>().ok()?;
    Some(SocketAddr::new(ip.into(), 0))
}

/// Resolve the client address before the access log reads it.
pub(crate) async fn apply_forwarded_client(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let peer = request.extensions().get::<ConnectInfo<SocketAddr>>();
    let proxies = &state.trusted_proxies;
    if (proxies.always_trust
        || peer.is_some_and(|ConnectInfo(peer)| proxies.contains_ip(peer.ip())))
        && let Some(address) = proxies.client_addr(request.headers())
    {
        request.extensions_mut().insert(ConnectInfo(address));
    }
    next.run(request).await
}
