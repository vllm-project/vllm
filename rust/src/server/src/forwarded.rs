// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The client behind trusted reverse proxies, resolved from `X-Forwarded-For`
//! the way uvicorn's `ProxyHeadersMiddleware` does for the Python frontend.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::http::HeaderMap;
use ipnet::{IpNet, Ipv4Net};
use serde::Serialize;

/// Proxies trusted to report the client in `X-Forwarded-For`, parsed from
/// uvicorn's `FORWARDED_ALLOW_IPS`: comma-separated addresses, networks, or
/// other literals, or `*` to trust every peer.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct TrustedProxies {
    source: String,
    #[serde(skip)]
    trust_all: bool,
    #[serde(skip)]
    addresses: Vec<IpAddr>,
    #[serde(skip)]
    networks: Vec<IpNet>,
    #[serde(skip)]
    literals: Vec<String>,
}

impl fmt::Debug for TrustedProxies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TrustedProxies").field(&self.source).finish()
    }
}

impl TrustedProxies {
    /// uvicorn's default when `FORWARDED_ALLOW_IPS` is unset.
    pub const DEFAULT: &str = "127.0.0.1,::1";

    pub fn parse(source: &str) -> Self {
        let mut proxies = Self {
            source: source.to_owned(),
            trust_all: source == "*",
            addresses: Vec::new(),
            networks: Vec::new(),
            literals: Vec::new(),
        };
        if proxies.trust_all {
            return proxies;
        }
        for host in source.split(',').map(str::trim) {
            if host.contains('/') {
                match parse_network(host) {
                    Some(network) => proxies.networks.push(network),
                    None => proxies.literals.push(host.to_owned()),
                }
            } else {
                match host.parse::<IpAddr>() {
                    Ok(address) => proxies.addresses.push(address),
                    Err(_) => proxies.literals.push(host.to_owned()),
                }
            }
        }
        proxies
    }

    fn trusts_address(&self, address: IpAddr) -> bool {
        self.trust_all
            || self.addresses.contains(&address)
            || self.networks.iter().any(|network| network.contains(&address))
    }

    fn trusts(&self, host: &str) -> bool {
        match host.parse::<IpAddr>() {
            Ok(address) => self.trusts_address(address),
            Err(_) => {
                self.trust_all
                    || (!host.is_empty() && self.literals.iter().any(|literal| literal == host))
            }
        }
    }

    /// The client of a request from `peer`, which is absent on a Unix-domain
    /// socket: the `X-Forwarded-For` client when the peer is trusted, else the
    /// peer itself.
    pub fn client(&self, peer: Option<SocketAddr>, headers: &HeaderMap) -> Option<String> {
        let peer_trusted = match peer {
            Some(peer) => self.trusts_address(peer.ip()),
            None => self.trust_all,
        };
        if peer_trusted && let Some(forwarded) = self.forwarded_client(headers) {
            return Some(forwarded);
        }
        peer.map(|peer| peer.to_string())
    }

    /// The rightmost untrusted `X-Forwarded-For` entry, since each proxy
    /// appends the peer it saw; the first entry if every proxy is trusted.
    /// `None` when the header is missing or its chosen entry is empty.
    fn forwarded_client(&self, headers: &HeaderMap) -> Option<String> {
        let values: Vec<String> = (headers.get_all("x-forwarded-for").iter())
            .map(|value| value.as_bytes().iter().map(|&byte| char::from(byte)).collect())
            .collect();
        if values.is_empty() {
            return None;
        }
        let joined = values.join(", ");
        let entries: Vec<&str> = joined.split(',').map(str::trim).collect();
        let chosen = if self.trust_all {
            entries[0]
        } else {
            (entries.iter().rev())
                .find(|entry| !self.trusts(parse_host_port(entry).0))
                .unwrap_or(&entries[0])
        };
        let (host, port) = parse_host_port(chosen);
        if host.is_empty() {
            return None;
        }
        Some(match port {
            Some(port) if host.contains(':') => format!("[{host}]:{port}"),
            Some(port) => format!("{host}:{port}"),
            None => host.to_owned(),
        })
    }
}

/// A network as Python's `ip_network` reads one: a prefix length or, for IPv4,
/// a netmask or hostmask. A network with host bits set is rejected.
fn parse_network(host: &str) -> Option<IpNet> {
    let network = host.parse::<IpNet>().ok().or_else(|| {
        let (address, mask) = host.split_once('/')?;
        let (Ok(address), Ok(mask)) = (address.parse::<Ipv4Addr>(), mask.parse::<Ipv4Addr>())
        else {
            return None;
        };
        let network = Ipv4Net::with_netmask(address, mask);
        network.or_else(|_| Ipv4Net::with_netmask(address, !mask)).ok().map(IpNet::V4)
    })?;
    (network.trunc() == network).then_some(network)
}

/// Split a forwarded entry into its host and port, as uvicorn does: a bare
/// address, IPv4 `host:port`, or bracketed IPv6 `[host]:port`. A malformed
/// entry keeps no port, so trust checks never normalize arbitrary input.
fn parse_host_port(entry: &str) -> (&str, Option<i64>) {
    if let Some(bracketed) = entry.strip_prefix('[') {
        let Some((host, remainder)) = bracketed.split_once(']') else {
            return (entry, None);
        };
        return match remainder.strip_prefix(':') {
            None if remainder.is_empty() => (host, None),
            None => (entry, None),
            Some(port) => (host, port.trim().parse().ok()),
        };
    }
    if entry.matches(':').count() == 1 {
        let (host, port) = entry.rsplit_once(':').expect("one colon");
        return match port.trim().parse() {
            Ok(port) => (host, Some(port)),
            Err(_) => (entry, None),
        };
    }
    (entry, None)
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn client(trusted: &str, peer: Option<&str>, forwarded_for: &[&str]) -> Option<String> {
        let mut headers = HeaderMap::new();
        for value in forwarded_for {
            headers.append("x-forwarded-for", HeaderValue::from_str(value).unwrap());
        }
        let peer = peer.map(|peer| peer.parse().unwrap());
        TrustedProxies::parse(trusted).client(peer, &headers)
    }

    #[test]
    fn resolves_client_like_uvicorn() {
        let default = TrustedProxies::DEFAULT;
        let local = Some("127.0.0.1:5000");
        let cases = [
            // A trusted peer reports the rightmost untrusted entry.
            client(default, local, &["1.2.3.4, 127.0.0.1"]),
            client(default, Some("[::1]:5000"), &["1.2.3.4"]),
            // Entries from several headers join in order.
            client(default, local, &["9.9.9.9", "1.2.3.4"]),
            // An untrusted peer is the client, whatever the header says.
            client(default, Some("10.0.0.5:5000"), &["1.2.3.4"]),
            // Every entry trusted: the first one is the client.
            client(default, local, &["127.0.0.1, ::1"]),
            // An empty header keeps the peer.
            client(default, local, &[""]),
            client(default, local, &[]),
            // Networks, and ports in IPv4 and bracketed IPv6 entries.
            client(
                "10.0.0.0/8",
                Some("10.1.2.3:5000"),
                &["[2001:db8::1]:443, 10.9.9.9:80"],
            ),
            client("10.0.0.0/8", Some("10.1.2.3:5000"), &["1.2.3.4:8080"]),
            // A network with host bits set is only a literal, so it trusts nothing.
            client("10.0.0.1/8", Some("10.1.2.3:5000"), &["1.2.3.4"]),
            // `*` trusts every peer, even on a Unix-domain socket, and takes the first entry.
            client("*", Some("10.0.0.5:5000"), &["1.2.3.4, 5.6.7.8"]),
            client("*", None, &["1.2.3.4"]),
            client(default, None, &["1.2.3.4"]),
            // An empty entry is never trusted, even when listed.
            client("127.0.0.1,", local, &["1.2.3.4, "]),
            // Ports follow Python's `int`.
            client(default, local, &["1.2.3.4: 80"]),
            client(default, local, &["[2001:db8::1]:99999"]),
            // IPv4 networks may also take a netmask or hostmask.
            client("10.0.0.0/255.0.0.0", Some("10.1.2.3:5000"), &["1.2.3.4"]),
            client(
                "10.0.0.0/0.255.255.255",
                Some("10.1.2.3:5000"),
                &["1.2.3.4"],
            ),
            client("10.0.0.1/255.0.0.0", Some("10.1.2.3:5000"), &["1.2.3.4"]),
        ];
        expect_test::expect![[r#"
            [
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "10.0.0.5:5000",
                ),
                Some(
                    "127.0.0.1",
                ),
                Some(
                    "127.0.0.1:5000",
                ),
                Some(
                    "127.0.0.1:5000",
                ),
                Some(
                    "[2001:db8::1]:443",
                ),
                Some(
                    "1.2.3.4:8080",
                ),
                Some(
                    "10.1.2.3:5000",
                ),
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "1.2.3.4",
                ),
                None,
                Some(
                    "127.0.0.1:5000",
                ),
                Some(
                    "1.2.3.4:80",
                ),
                Some(
                    "[2001:db8::1]:99999",
                ),
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "1.2.3.4",
                ),
                Some(
                    "10.1.2.3:5000",
                ),
            ]
        "#]]
        .assert_debug_eq(&cases);
    }
}
