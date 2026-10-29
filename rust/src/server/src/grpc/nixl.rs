// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `MediaItem.nixl_ref`: decoded images that this server reads over NIXL
//! from the host memory of another process.
//!
//! The reader exists only with the `nixl-media` cargo feature. With
//! `--enable-grpc-nixl-media`, server startup starts it and fails when NIXL
//! does not start; without the feature or without a gRPC listener, the flag
//! fails startup.
//!
//! Trust model: the gRPC endpoint has no authentication, and with this
//! feature the server connects to the NIXL agent that a request names and
//! reads the memory that the request describes. The request also carries the
//! UCX worker address and memory keys that NIXL hands to UCX unchecked, and
//! UCX aborts the process on some malformed addresses (for example, an
//! unknown address version). Only trusted peers, such as a co-located
//! frontend, may reach the gRPC port.
//!
//! The server uses the pixels as sent. For the same model input as the encoded
//! image, the client applies what this server's decoder applies to it; see
//! `NixlRef.addr` in `inference.proto`.

use thiserror_ext::AsReport as _;
use tonic::Status;
use vllm_chat::multimodal::{DecodedImage, DeferredImage};

use super::pb;

#[cfg(feature = "nixl-media")]
mod reader;

/// Upper bound on the decoded pixels that the `nixl_ref` items of one request
/// may reference.
pub(super) const MAX_NIXL_MEDIA_BYTES: usize = 256 * 1024 * 1024;

/// Bounds of the canonical `NixlRef.agent_metadata` shape.
const MAX_AGENT_NAME_BYTES: usize = 256;
const MAX_CONNECTION_BYTES: usize = 16 * 1024;
const MAX_SECTIONS: usize = 16;
const MAX_KEY_BYTES: usize = 4 * 1024;

/// Start the NIXL reader for `--enable-grpc-nixl-media`.
#[cfg(feature = "nixl-media")]
pub(crate) async fn start() -> anyhow::Result<()> {
    tokio::task::spawn_blocking(reader::NixlReader::start)
        .await?
        .map_err(anyhow::Error::msg)
}

/// Check that `media[index].nixl_ref` can be read.
pub(super) fn support(index: usize, enabled: bool) -> Result<NixlSupport, Status> {
    let checked = if enabled {
        NixlSupport::check()
    } else {
        Err("the server was started without --enable-grpc-nixl-media".to_owned())
    };
    checked.map_err(|reason| {
        Status::invalid_argument(format!(
            "media[{index}].nixl_ref is not supported: {reason}"
        ))
    })
}

/// Proof that this server reads `nixl_ref` images: the started reader.
#[cfg(feature = "nixl-media")]
pub(super) struct NixlSupport(&'static reader::NixlReader);

#[cfg(feature = "nixl-media")]
impl NixlSupport {
    /// Server startup starts the reader when `--enable-grpc-nixl-media` is on.
    fn check() -> Result<Self, String> {
        reader::NixlReader::get()
            .map(Self)
            .ok_or_else(|| "the NIXL reader is not started".to_owned())
    }

    /// The load that reads `image` when chat polls it.
    pub(super) fn defer(self, image: NixlImage) -> DeferredImage {
        DeferredImage::new(self.0.read_image(image))
    }
}

/// This build cannot read `nixl_ref` images, so there is no proof.
#[cfg(not(feature = "nixl-media"))]
pub(super) enum NixlSupport {}

#[cfg(not(feature = "nixl-media"))]
impl NixlSupport {
    fn check() -> Result<Self, String> {
        Err("this server was built without the `nixl-media` feature".to_owned())
    }

    pub(super) fn defer(self, _image: NixlImage) -> DeferredImage {
        match self {}
    }
}

/// A validated `nixl_ref` image.
#[derive(Debug)]
// Only the reader reads the fields.
#[cfg_attr(not(feature = "nixl-media"), allow(dead_code))]
pub(super) struct NixlImage {
    pub index: usize,
    pub agent_metadata: Vec<u8>,
    /// `agent_metadata` as NIXL reads it.
    pub agent: AgentMetadata,
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// Byte length of the pixels.
    pub len: usize,
    pub uuid: Option<String>,
}

impl NixlImage {
    pub(super) fn from_proto(
        index: usize,
        reference: pb::NixlRef,
        uuid: Option<String>,
    ) -> Result<Self, Status> {
        let invalid =
            |message: &str| Status::invalid_argument(format!("media[{index}].nixl_ref: {message}"));
        let agent = AgentMetadata::parse(&reference.agent_metadata).map_err(invalid)?;
        let len = DecodedImage::byte_len(reference.width, reference.height, reference.channels)
            .map_err(|error| invalid(&error.to_report_string()))?;
        if reference.addr == 0 {
            return Err(invalid("addr is required"));
        }
        let end = reference
            .addr
            .checked_add(len as u64)
            .ok_or_else(|| invalid("the pixels end past the address space"))?;
        if !agent.sections.iter().any(|section| {
            section.addr <= reference.addr
                && section.addr.checked_add(section.len).is_some_and(|limit| end <= limit)
        }) {
            return Err(invalid(
                "the pixels are not inside a region of agent_metadata",
            ));
        }
        Ok(Self {
            index,
            agent,
            agent_metadata: reference.agent_metadata,
            addr: reference.addr,
            width: reference.width,
            height: reference.height,
            channels: reference.channels,
            len,
            uuid,
        })
    }
}

/// `nixl_mem_t::DRAM_SEG`.
const DRAM_SEG: u32 = 0;
const UCX: &[u8] = b"UCX";
const MEMORY_SECTIONS: &[u8] = b"MemSection";
const SECTION_LIST: &[u8] = b"nixlSDList";

/// NIXL agent metadata in the one shape that the reader accepts: one image's
/// region (`getLocalPartialMD` for the region, joined with the agent's
/// connection info), or `getLocalMD` of an agent with 1 to 16 regions.
///
/// That is the `nixlSerDes` layout that `nixlAgent::loadRemoteMD` reads: the
/// agent name (UTF-8, no NUL, at most 256 bytes), exactly one connection entry
/// for backend `UCX`, and exactly one section list for backend `UCX` with 1 to
/// 16 host-memory regions of device 0 that do not overlap, each with a
/// non-empty address, length, and memory key; nothing after it. Every field is checked for its length
/// before the parser keeps it, so what it allocates does not scale with the
/// input. NIXL mishandles other input: nixl-sys 1.5.1 aborts the process on a
/// name that is not UTF-8, a name with a NUL byte comes back cut short, and
/// NIXL 1.5.1 indexes an array with the memory type of a section without a
/// bound check (an out-of-bounds write). The UCX connection info and the
/// memory keys stay opaque and reach UCX as trusted input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AgentMetadata {
    pub name: String,
    /// The UCX connection info (worker address).
    pub connection: Vec<u8>,
    pub sections: Vec<RemoteSection>,
}

/// One registered host-memory region in agent metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RemoteSection {
    pub addr: u64,
    pub len: u64,
    /// UCX data for the region, such as a packed memory key.
    pub key: Vec<u8>,
}

const MALFORMED: &str = "agent_metadata is not NIXL agent metadata";

impl AgentMetadata {
    /// Parse `nixlSerDes` output: `"nixlSerDes|"`, then fields of a tag, a
    /// native-endian `size_t` length, the value, and `'|'`.
    pub(super) fn parse(metadata: &[u8]) -> Result<Self, &'static str> {
        let mut fields = SerDes(metadata.strip_prefix(b"nixlSerDes|").ok_or(MALFORMED)?);
        let name = fields.field("Agent", MAX_AGENT_NAME_BYTES)?;
        if name.is_empty() {
            return Err("agent name is empty");
        }
        let name = std::str::from_utf8(name).map_err(|_| "agent name is not UTF-8")?;
        if name.contains('\0') {
            return Err("agent name contains a NUL byte");
        }
        if fields.count("Conns")? != 1 || fields.field("t", UCX.len())? != UCX {
            return Err("agent_metadata must have exactly one UCX connection");
        }
        let connection = fields.field("c", MAX_CONNECTION_BYTES)?;
        if connection.is_empty() {
            return Err("the UCX connection info is empty");
        }
        if fields.field("", MEMORY_SECTIONS.len())? != MEMORY_SECTIONS {
            return Err(MALFORMED);
        }
        if fields.count("nixlSecElms")? != 1 || fields.field("bknd", UCX.len())? != UCX {
            return Err("agent_metadata must have exactly one UCX section list");
        }
        if fields.field("nixlDList", SECTION_LIST.len())? != SECTION_LIST {
            return Err(MALFORMED);
        }
        if fields.field("t", size_of::<u32>())? != DRAM_SEG.to_ne_bytes() {
            return Err("agent_metadata sections must be host memory");
        }
        let count = fields.count("n")?;
        if !(1..=MAX_SECTIONS).contains(&count) {
            return Err("agent_metadata must have 1 to 16 sections");
        }
        let mut sections = Vec::with_capacity(count);
        for _ in 0..count {
            // `nixlBasicDesc` (addr, len, devId), then the UCX data.
            let (words, key) = fields
                .field("", 24 + MAX_KEY_BYTES)?
                .split_first_chunk::<24>()
                .ok_or(MALFORMED)?;
            let word = |index: usize| {
                let mut bytes = [0; 8];
                bytes.copy_from_slice(&words[index * 8..index * 8 + 8]);
                u64::from_ne_bytes(bytes)
            };
            let (addr, len, device_id) = (word(0), word(1), word(2));
            if device_id != 0 {
                return Err("agent_metadata sections must be on device 0");
            }
            if addr == 0 || len == 0 || key.is_empty() {
                return Err("an agent_metadata section has an empty address, length, or key");
            }
            if addr.checked_add(len).is_none() {
                return Err("an agent_metadata section ends past the address space");
            }
            sections.push(RemoteSection {
                addr,
                len,
                key: key.to_vec(),
            });
        }
        if !fields.0.is_empty() {
            return Err(MALFORMED);
        }
        // NIXL finds the section of a transfer by address, and overlapping
        // sections make it pick the wrong one.
        let mut regions: Vec<_> = sections.iter().map(|s| (s.addr, s.addr + s.len)).collect();
        regions.sort_unstable();
        if regions.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("agent_metadata sections overlap");
        }
        Ok(Self {
            name: name.to_owned(),
            connection: connection.to_vec(),
            sections,
        })
    }

    /// Metadata with the connection info and no sections, as
    /// `getLocalPartialMD` gives it for an empty list. Loading it first
    /// separates a failed connection, which a later read can retry, from
    /// refused sections, which leave the remote agent unusable in NIXL 1.5.1.
    #[cfg(test)]
    pub(super) fn connection_only(&self) -> Vec<u8> {
        self.serialize(&[])
    }

    /// The canonical `nixlSerDes` form of this agent with `sections`.
    #[cfg(test)]
    fn serialize(&self, sections: &[RemoteSection]) -> Vec<u8> {
        self.serialize_as(&self.name, sections)
    }

    /// [`Self::serialize`] under another agent name. NIXL keys a remote agent
    /// by this name only; the connection uses the UCX worker address.
    #[cfg_attr(not(feature = "nixl-media"), allow(dead_code))]
    pub(super) fn serialize_as(&self, name: &str, sections: &[RemoteSection]) -> Vec<u8> {
        let mut out = b"nixlSerDes|".to_vec();
        let mut field = |tag: &str, value: &[u8]| {
            out.extend_from_slice(tag.as_bytes());
            out.extend_from_slice(&value.len().to_ne_bytes());
            out.extend_from_slice(value);
            out.push(b'|');
        };
        field("Agent", name.as_bytes());
        field("Conns", &1usize.to_ne_bytes());
        field("t", UCX);
        field("c", &self.connection);
        field("", MEMORY_SECTIONS);
        field(
            "nixlSecElms",
            &usize::from(!sections.is_empty()).to_ne_bytes(),
        );
        if !sections.is_empty() {
            field("bknd", UCX);
            field("nixlDList", SECTION_LIST);
            field("t", &DRAM_SEG.to_ne_bytes());
            field("n", &sections.len().to_ne_bytes());
            for section in sections {
                let words = [section.addr, section.len, 0].map(u64::to_ne_bytes);
                field("", &[words.as_flattened(), &section.key].concat());
            }
        }
        out
    }
}

/// The unread rest of `nixlSerDes` output.
struct SerDes<'a>(&'a [u8]);

impl<'a> SerDes<'a> {
    /// The next field, which must have `tag` and at most `max` bytes.
    fn field(&mut self, tag: &str, max: usize) -> Result<&'a [u8], &'static str> {
        let rest = self.0.strip_prefix(tag.as_bytes()).ok_or(MALFORMED)?;
        let (len, rest) = rest.split_first_chunk::<{ size_of::<usize>() }>().ok_or(MALFORMED)?;
        let len = usize::from_ne_bytes(*len);
        if len > max {
            return Err("an agent_metadata field is too long");
        }
        let (value, rest) = rest.split_at_checked(len).ok_or(MALFORMED)?;
        self.0 = rest.strip_prefix(b"|").ok_or(MALFORMED)?;
        Ok(value)
    }

    fn count(&mut self, tag: &str) -> Result<usize, &'static str> {
        self.field(tag, size_of::<usize>())?
            .try_into()
            .map(usize::from_ne_bytes)
            .map_err(|_| MALFORMED)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Canonical agent metadata with one host memory region per `(addr, len)`.
    pub(in crate::grpc) fn metadata(name: &str, regions: &[(u64, u64)]) -> Vec<u8> {
        let sections: Vec<_> = regions
            .iter()
            .map(|&(addr, len)| RemoteSection {
                addr,
                len,
                key: b"rkey".to_vec(),
            })
            .collect();
        AgentMetadata {
            name: name.to_owned(),
            connection: b"worker address".to_vec(),
            sections: Vec::new(),
        }
        .serialize(&sections)
    }

    /// The fields of `metadata("peer", &[(0x1000, 64)])`, in order: 0 Agent,
    /// 1 Conns, 2 t, 3 c, 4 MemSection, 5 nixlSecElms, 6 bknd, 7 nixlDList,
    /// 8 t (memory type), 9 n, 10 the region.
    fn fields() -> Vec<(&'static str, Vec<u8>)> {
        let count = |n: usize| n.to_ne_bytes().to_vec();
        vec![
            ("Agent", b"peer".to_vec()),
            ("Conns", count(1)),
            ("t", b"UCX".to_vec()),
            ("c", b"worker address".to_vec()),
            ("", b"MemSection".to_vec()),
            ("nixlSecElms", count(1)),
            ("bknd", b"UCX".to_vec()),
            ("nixlDList", b"nixlSDList".to_vec()),
            ("t", DRAM_SEG.to_ne_bytes().to_vec()),
            ("n", count(1)),
            ("", region(0x1000, 64, 0, b"rkey")),
        ]
    }

    fn region(addr: u64, len: u64, device_id: u64, key: &[u8]) -> Vec<u8> {
        let words = [addr, len, device_id].map(u64::to_ne_bytes);
        [words.as_flattened(), key].concat()
    }

    fn join(fields: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"nixlSerDes|".to_vec();
        for (tag, value) in fields {
            out.extend_from_slice(tag.as_bytes());
            out.extend_from_slice(&value.len().to_ne_bytes());
            out.extend_from_slice(value);
            out.push(b'|');
        }
        out
    }

    /// `fields()` with field `index` set to `value`.
    fn with(index: usize, value: impl Into<Vec<u8>>) -> Vec<u8> {
        let mut fields = fields();
        fields[index].1 = value.into();
        join(&fields)
    }

    fn reference() -> pb::NixlRef {
        pb::NixlRef {
            agent_metadata: metadata("peer", &[(0x1000, 8192)]),
            addr: 0x1000,
            height: 32,
            width: 64,
            channels: 3,
        }
    }

    #[test]
    fn nixl_image_takes_the_reference_fields() {
        let mut image = NixlImage::from_proto(3, reference(), Some("id".to_owned())).unwrap();
        assert_eq!(image.agent_metadata, reference().agent_metadata);
        image.agent_metadata.clear();
        expect_test::expect![[r#"
            NixlImage {
                index: 3,
                agent_metadata: [],
                agent: AgentMetadata {
                    name: "peer",
                    connection: [
                        119,
                        111,
                        114,
                        107,
                        101,
                        114,
                        32,
                        97,
                        100,
                        100,
                        114,
                        101,
                        115,
                        115,
                    ],
                    sections: [
                        RemoteSection {
                            addr: 4096,
                            len: 8192,
                            key: [
                                114,
                                107,
                                101,
                                121,
                            ],
                        },
                    ],
                },
                addr: 4096,
                width: 64,
                height: 32,
                channels: 3,
                len: 6144,
                uuid: Some(
                    "id",
                ),
            }
        "#]]
        .assert_debug_eq(&image);
    }

    #[test]
    fn nixl_image_rejects_incomplete_references() {
        let with = |mutate: fn(&mut pb::NixlRef)| {
            let mut reference = reference();
            mutate(&mut reference);
            reference
        };
        for (reference, message) in [
            (
                with(|r| r.agent_metadata.clear()),
                "media[0].nixl_ref: agent_metadata is not NIXL agent metadata",
            ),
            (with(|r| r.addr = 0), "media[0].nixl_ref: addr is required"),
            (
                with(|r| r.addr = u64::MAX - 100),
                "media[0].nixl_ref: the pixels end past the address space",
            ),
            (
                // The pixels start inside the region and end past it.
                with(|r| r.addr = 0x1000 + 4096),
                "media[0].nixl_ref: the pixels are not inside a region of agent_metadata",
            ),
            (
                with(|r| r.addr = 0x800),
                "media[0].nixl_ref: the pixels are not inside a region of agent_metadata",
            ),
            (
                with(|r| r.width = 0),
                "media[0].nixl_ref: invalid decoded image: 0x32 has no pixels",
            ),
            (
                with(|r| r.channels = 0),
                "media[0].nixl_ref: invalid decoded image: 0 channels is not 1, 2, 3, or 4",
            ),
        ] {
            let error = NixlImage::from_proto(0, reference, None).unwrap_err();
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert_eq!(error.message(), message);
        }
    }

    #[test]
    fn canonical_agent_metadata_round_trips() {
        let bytes = join(&fields());
        let agent = AgentMetadata::parse(&bytes).unwrap();
        assert_eq!(agent.serialize(&agent.sections), bytes);
        assert_eq!(bytes, metadata("peer", &[(0x1000, 64)]));

        let mut adjacent = fields();
        adjacent[9].1 = 2usize.to_ne_bytes().to_vec();
        adjacent.push(("", region(0x1000 + 64, 64, 0, b"rkey")));
        assert_eq!(
            AgentMetadata::parse(&join(&adjacent)).unwrap().sections.len(),
            2
        );

        let mut sixteen = fields();
        sixteen[9].1 = 16usize.to_ne_bytes().to_vec();
        sixteen.extend((1..16).map(|i| ("", region(0x1000 * (i + 1), 64, 0, b"rkey"))));
        assert_eq!(
            AgentMetadata::parse(&join(&sixteen)).unwrap().sections.len(),
            16
        );

        let connection = AgentMetadata::parse(&agent.connection_only());
        assert_eq!(
            connection,
            Err("agent_metadata must have exactly one UCX section list"),
            "a request cannot carry metadata without sections"
        );
    }

    /// Everything but the canonical shape is refused before NIXL sees it:
    /// nixl-sys 1.5.1 aborts on a name that is not UTF-8, and NIXL 1.5.1
    /// writes out of bounds for an unknown memory type.
    #[test]
    fn agent_metadata_outside_the_canonical_shape_is_refused() {
        let count = |n: usize| n.to_ne_bytes().to_vec();
        let too_many = {
            let mut fields = fields();
            fields[9].1 = count(17);
            fields.extend((1..17).map(|i| ("", region(0x1000 * (i + 1), 64, 0, b"rkey"))));
            join(&fields)
        };
        // The second region starts inside the first.
        let overlapping = {
            let mut fields = fields();
            fields[9].1 = count(2);
            fields.push(("", region(0x1000 + 32, 64, 0, b"rkey")));
            join(&fields)
        };
        let huge_backend = {
            let mut out = join(&fields()[..6]);
            out.extend_from_slice(b"bknd");
            out.extend_from_slice(&(1usize << 40).to_ne_bytes());
            out.extend_from_slice(b"UCX|");
            out
        };
        let mut trailing = join(&fields());
        trailing.push(0);
        let full = join(&fields());
        let too_long = |n: usize| vec![b'a'; n];
        let not_ucx_connection = "agent_metadata must have exactly one UCX connection";
        let not_ucx_sections = "agent_metadata must have exactly one UCX section list";
        let section_count = "agent_metadata must have 1 to 16 sections";
        let empty_region = "an agent_metadata section has an empty address, length, or key";
        let long_field = "an agent_metadata field is too long";
        for (metadata, message) in [
            (with(0, b"\xff\xfe".as_slice()), "agent name is not UTF-8"),
            (
                with(0, b"a\0b".as_slice()),
                "agent name contains a NUL byte",
            ),
            (with(0, b"".as_slice()), "agent name is empty"),
            (with(0, too_long(MAX_AGENT_NAME_BYTES + 1)), long_field),
            (with(1, count(0)), not_ucx_connection),
            (with(1, count(2)), not_ucx_connection),
            (with(2, b"GDS".as_slice()), not_ucx_connection),
            (with(2, b"UCX_MO".as_slice()), long_field),
            (with(3, b"".as_slice()), "the UCX connection info is empty"),
            (with(3, too_long(MAX_CONNECTION_BYTES + 1)), long_field),
            (with(5, count(0)), not_ucx_sections),
            (with(5, count(2)), not_ucx_sections),
            (with(6, b"GDS".as_slice()), not_ucx_sections),
            (huge_backend, long_field),
            (with(7, b"nixlXList".as_slice()), MALFORMED),
            // VRAM_SEG, and memory types that NIXL 1.5.1 indexes out of bounds.
            (
                with(8, 1u32.to_ne_bytes()),
                "agent_metadata sections must be host memory",
            ),
            (
                with(8, 5u32.to_ne_bytes()),
                "agent_metadata sections must be host memory",
            ),
            (
                with(8, u32::MAX.to_ne_bytes()),
                "agent_metadata sections must be host memory",
            ),
            (with(9, count(0)), section_count),
            (with(9, count(usize::MAX)), section_count),
            (too_many, section_count),
            (
                with(10, region(0x1000, 64, 1, b"rkey")),
                "agent_metadata sections must be on device 0",
            ),
            (with(10, region(0, 64, 0, b"rkey")), empty_region),
            (with(10, region(0x1000, 0, 0, b"rkey")), empty_region),
            (with(10, region(0x1000, 64, 0, b"")), empty_region),
            (
                with(10, region(0x1000, 64, 0, &too_long(MAX_KEY_BYTES + 1))),
                long_field,
            ),
            (
                with(10, region(u64::MAX - 8, 64, 0, b"rkey")),
                "an agent_metadata section ends past the address space",
            ),
            (overlapping, "agent_metadata sections overlap"),
            (with(10, [0; 23]), MALFORMED),
            (full[..full.len() - 1].to_vec(), MALFORMED),
            (trailing, MALFORMED),
            (b"notNixl".to_vec(), MALFORMED),
        ] {
            assert_eq!(AgentMetadata::parse(&metadata), Err(message));
        }
        assert!(AgentMetadata::parse(&with(0, too_long(MAX_AGENT_NAME_BYTES))).is_ok());
    }
}
