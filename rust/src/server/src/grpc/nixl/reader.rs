// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The NIXL agent that reads `nixl_ref` pixels. Host memory (DRAM) only, on
//! both sides, and the UCX backend only.
//!
//! Leave `NIXL_ETCD_ENDPOINTS` unset for vllm-rs: with it, NIXL starts its
//! etcd thread, which busy-yields one core, and the reader does not use etcd.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use nixl_sys::{
    Agent, AgentConfig, MemType, MemoryRegion, NixlDescriptor, NixlError, RegistrationHandle,
    XferDescList, XferOp, XferRequest, XferStatus,
};
use thiserror_ext::AsReport as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use vllm_chat::multimodal::DecodedImage;

use super::{NixlImage, RemoteSection};

/// Upper bound on the UCX progress thread's `poll()` sleep. It still wakes at
/// once on UCX events. NIXL's default of 0 spins one core for the process life.
const PROGRESS_THREAD_DELAY_US: u64 = 1_000;

/// NIXL keeps every section that it loads for a remote agent, and owners
/// register one section per image. Past this, the next load first unloads.
const MAX_SECTIONS_PER_REMOTE: usize = 4096;

/// Past this, the least recently used idle remote agent is unloaded.
const MAX_REMOTE_AGENTS: usize = 64;

/// See [`NixlReader`]. Each name leaves one UCX connection in NIXL.
const MAX_NAMES_PER_REMOTE: u32 = 8;

const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024;

/// Buffers are sized in powers of two so that images of similar size share them.
const MIN_BUFFER_BYTES: usize = 64 * 1024;

const MAX_CONCURRENT_READS: usize = 16;

/// Receive buffers plus pixel copies of the reads in flight and of stuck
/// reads. One request always fits.
const MAX_IN_FLIGHT_KIB: usize = 1024 * 1024;
const _: () = assert!(2 * super::MAX_NIXL_MEDIA_BYTES / 1024 <= MAX_IN_FLIGHT_KIB);

/// Covers the waits for the owner, the limits, and a blocking thread, the load,
/// and the transfer.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Least time a read needs left to post its transfer, and least time it then
/// waits for it: the post can block on a NIXL lock, and a transfer that times
/// out at its first poll becomes a stuck read. A healthy one takes 0.4 to 2 ms.
const MIN_TRANSFER_TIME: Duration = Duration::from_secs(1);

const MAX_POLL_SLEEP: Duration = Duration::from_micros(200);

const STUCK_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, thiserror::Error)]
enum ReadError {
    #[error("NIXL error: {0}")]
    Nixl(#[from] NixlError),
    #[error(
        "NIXL could not load the connection info of remote agent `{name}`, a later read tries \
         again: {error}"
    )]
    Connect { name: Arc<str>, error: NixlError },
    #[error("NIXL refused the memory sections of remote agent `{name}`: {error}")]
    Refused { name: Arc<str>, error: NixlError },
    #[error("{0}")]
    Unusable(String),
    #[error("the read did not finish in {0:?}")]
    Timeout(Duration),
    #[error("the request was cancelled")]
    Cancelled,
    #[error("reader state is poisoned")]
    Poisoned,
}

/// Limits that tests lower.
struct Limits {
    max_sections_per_remote: usize,
    max_remote_agents: usize,
    read_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_sections_per_remote: MAX_SECTIONS_PER_REMOTE,
            max_remote_agents: MAX_REMOTE_AGENTS,
            read_timeout: READ_TIMEOUT,
        }
    }
}

/// Reads decoded images that other processes registered with NIXL.
///
/// One agent per process, with a UCX backend. Reads of one owner (one remote
/// agent) run one at a time in arrival order, because NIXL erases all sections
/// of a remote agent when a load of its metadata fails. A read takes its
/// owner's turn before any global permit, so a queue on one owner does not
/// starve the others. The waits and the transfer share one deadline. No lock
/// that async code takes is held across a NIXL call.
///
/// The reader works around NIXL 1.5.2:
/// - After NIXL refuses an agent's sections, it keeps the UCX connection under
///   that name and refuses every later load of it. So the reader loads each
///   owner as `<agent>#<n>` and moves to `n + 1` after a refusal, for the
///   life of the process; past `MAX_NAMES_PER_REMOTE` the owner is unusable.
///   It avoids the refusals it can predict: request conversion accepts only
///   the canonical metadata shape, it loads the connection info alone first,
///   and it unloads the agent before a load whose sections conflict with
///   loaded ones. A failed connection loads nothing, so a later read retries
///   it.
/// - NIXL finds the section of a transfer by address (the loaded section at or
///   before it), so loaded sections of one agent must not overlap.
/// - A transfer that fails is released, as in vLLM's NIXL KV connector, but
///   NIXL can report the error while its READ still runs. So its buffer goes
///   back to the pool even past the pool limit, and a late write can only
///   reach a receive buffer.
/// - nixl-sys takes one process-wide lock around loads, posts, status polls,
///   and releases, so reads of different owners overlap only in transfers.
///
/// Every unload runs under the owner's turn, and a transfer that is still in
/// progress at its deadline keeps the turn until NIXL reports that it ended.
pub(crate) struct NixlReader {
    agent: Agent,
    owners: Mutex<HashMap<Arc<str>, Arc<Owner>>>,
    /// Name index of each owner past `#0`, kept after it leaves `owners`.
    /// Locked last: under `owners` or an owner's state, never the reverse.
    name_indices: Mutex<HashMap<Arc<str>, u32>>,
    buffers: Mutex<BufferPool<LocalBuffer>>,
    /// Transfers that failed or timed out and may still write to their buffer.
    stuck: Mutex<Vec<StuckRead>>,
    stuck_added: Condvar,
    reads: Arc<Semaphore>,
    kib: Arc<Semaphore>,
    limits: Limits,
}

/// The NIXL agent of one owner process.
struct Owner {
    name: Arc<str>,
    /// One read at a time, first come first served. Closed while a stuck
    /// transfer holds it.
    turn: Arc<Semaphore>,
    /// Why NIXL cannot load the remote agent again in this process.
    unusable: OnceLock<String>,
    /// Only the holder of the turn, or an unloader of an idle owner, locks it.
    state: Mutex<OwnerState>,
}

#[derive(Default)]
struct OwnerState {
    /// Which name NIXL holds the remote agent under; see [`nixl_name`].
    name_index: u32,
    /// The UCX connection info that NIXL holds for the remote agent.
    connection: Option<Vec<u8>>,
    /// The loaded sections by address. They do not overlap.
    sections: BTreeMap<u64, RemoteSection>,
    last_used: Option<Instant>,
}

fn nixl_name(owner: &Owner, state: &OwnerState) -> String {
    format!("{}#{}", owner.name, state.name_index)
}

/// Permits that a read holds until its transfer ends.
struct ReadPermits {
    _turn: OwnedSemaphorePermit,
    _kib: OwnedSemaphorePermit,
}

/// A transfer that may still be active. It keeps its buffer, its turn on the
/// owner, and its share of the memory limit until NIXL reports that it ended.
struct StuckRead {
    request: XferRequest,
    owner: Arc<Owner>,
    buffer: LocalBuffer,
    _permits: ReadPermits,
}

static GLOBAL: OnceLock<Result<NixlReader, String>> = OnceLock::new();
static STUCK_POLLER: OnceLock<Result<(), String>> = OnceLock::new();

impl NixlReader {
    /// Start the process-wide reader and its stuck-read poller. Blocks. Only
    /// the first call starts them; later calls return its result.
    pub(super) fn start() -> Result<(), String> {
        let reader = GLOBAL
            .get_or_init(|| {
                let name = format!("vllm-nixl-media-{}", uuid::Uuid::new_v4());
                let reader = Self::new(&name, Limits::default());
                if reader.is_ok() {
                    tracing::info!(agent = %name, "NIXL media reader is ready");
                }
                reader
            })
            .as_ref()
            .map_err(Clone::clone)?;
        // Without the poller, stuck reads would never release their memory.
        STUCK_POLLER
            .get_or_init(|| {
                std::thread::Builder::new()
                    .name("nixl-media-stuck".to_owned())
                    .spawn(move || reader.poll_stuck_reads_forever())
                    .map(|_| ())
                    .map_err(|error| {
                        format!(
                            "failed to start the stuck-read poller: {}",
                            error.as_report()
                        )
                    })
            })
            .clone()
    }

    /// The process-wide reader, after `start` succeeded.
    pub(super) fn get() -> Option<&'static Self> {
        // A failed start never starts the poller.
        STUCK_POLLER.get()?.as_ref().ok()?;
        GLOBAL.get()?.as_ref().ok()
    }

    fn new(name: &str, limits: Limits) -> Result<Self, String> {
        if nixl_sys::is_stub() {
            // Every other NIXL call would abort the process.
            return Err("the NIXL library (libnixl_capi.so) cannot be loaded".to_owned());
        }
        let config = AgentConfig {
            pthr_delay_us: PROGRESS_THREAD_DELAY_US,
            ..AgentConfig::default()
        };
        let agent = Agent::new_configured(name, &config)
            .map_err(|error| format!("failed to create the NIXL agent: {}", error.as_report()))?;
        agent
            .get_plugin_params("UCX")
            .and_then(|(_mems, params)| agent.create_backend("UCX", &params))
            .map_err(|error| {
                format!(
                    "failed to create the NIXL UCX backend: {}",
                    error.as_report()
                )
            })?;
        Ok(Self {
            agent,
            owners: Mutex::default(),
            name_indices: Mutex::default(),
            buffers: Mutex::new(BufferPool::new(MAX_POOLED_BYTES)),
            stuck: Mutex::default(),
            stuck_added: Condvar::new(),
            reads: Arc::new(Semaphore::new(MAX_CONCURRENT_READS)),
            kib: Arc::new(Semaphore::new(MAX_IN_FLIGHT_KIB)),
            limits,
        })
    }

    /// Wait for the owner's turn and the global limits, then read on the
    /// blocking pool, all before one deadline. Dropping the future while it
    /// waits releases what it holds. Once the read starts, the future
    /// returns only after it ends: at its next cancellation or deadline check,
    /// or when its transfer finishes or becomes stuck.
    pub(super) async fn read_image(
        &'static self,
        image: NixlImage,
    ) -> vllm_chat::Result<DecodedImage> {
        let timeout = self.limits.read_timeout;
        let deadline = tokio::time::Instant::now() + timeout;
        let index = image.index;
        let failed = move |message: String| {
            vllm_chat::Error::MediaUnavailable(format!("media[{index}].nixl_ref: {message}"))
        };
        let owner = self.owner(&image.agent.name);
        if let Some(reason) = owner.unusable.get() {
            return Err(failed(reason.clone()));
        }
        let waited = |what: &str| {
            failed(format!(
                "the read did not start in {timeout:?}: it waited for {what}"
            ))
        };
        let turn = tokio::time::timeout_at(deadline, Arc::clone(&owner.turn).acquire_owned())
            .await
            .map_err(|_| {
                waited(&format!(
                    "the earlier reads of remote agent `{}`",
                    owner.name
                ))
            })?
            .map_err(|_| {
                failed(format!(
                    "remote agent `{}` is blocked by a transfer that did not end",
                    owner.name
                ))
            })?;
        let closed = |_| failed("the NIXL reader is shut down".to_owned());
        let read_permit =
            tokio::time::timeout_at(deadline, Arc::clone(&self.reads).acquire_owned())
                .await
                .map_err(|_| waited("one of the reads in flight across all requests"))?
                .map_err(closed)?;
        // At most 2 x 256 MiB, so it fits u32 and `MAX_IN_FLIGHT_KIB`.
        let kib = (buffer_capacity(image.len) + image.len).div_ceil(1024) as u32;
        let kib = tokio::time::timeout_at(deadline, Arc::clone(&self.kib).acquire_many_owned(kib))
            .await
            .map_err(|_| waited("memory that other reads hold"))?
            .map_err(closed)?;

        // Dropping this future (client gone, gRPC deadline) cancels the read.
        let cancelled = CancellationToken::new();
        let _cancel_on_drop = cancelled.clone().drop_guard();
        let permits = ReadPermits {
            _turn: turn,
            _kib: kib,
        };
        let std_deadline = deadline.into_std();
        // Whoever sets it first decides: the blocking thread that starts the
        // read, or this future when it gives up on a read that never started.
        let claimed = Arc::new(AtomicBool::new(false));
        let claimed_by_read = Arc::clone(&claimed);
        let mut task = tokio::task::spawn_blocking(move || {
            if claimed_by_read.swap(true, Ordering::AcqRel) {
                return Err(failed("the read was given up before it started".to_owned()));
            }
            let pixels = self
                .read(&image, &owner, permits, std_deadline, &cancelled)
                .map_err(|error| failed(format!("NIXL read failed: {}", error.as_report())))?;
            drop(read_permit);
            DecodedImage::new(
                pixels,
                image.width,
                image.height,
                image.channels,
                image.uuid,
            )
        });
        let finished = match tokio::time::timeout_at(deadline, &mut task).await {
            Ok(finished) => finished,
            // The read never starts, so it cannot touch the owner's memory.
            // Its permits go back when a blocking thread drops it.
            Err(_) if !claimed.swap(true, Ordering::AcqRel) => {
                return Err(waited("a blocking thread"));
            }
            // The read runs, and stops at its own deadline checks.
            Err(_) => task.await,
        };
        finished.map_err(|error| failed(format!("read task failed: {error}")))?
    }

    fn lock_owners(&self) -> std::sync::MutexGuard<'_, HashMap<Arc<str>, Arc<Owner>>> {
        self.owners.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_name_indices(&self) -> std::sync::MutexGuard<'_, HashMap<Arc<str>, u32>> {
        self.name_indices.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The owner of `name`, created on first use under the agent's current
    /// name index.
    fn owner(&self, name: &str) -> Arc<Owner> {
        let mut owners = self.lock_owners();
        if let Some(owner) = owners.get(name) {
            return Arc::clone(owner);
        }
        let name_index = self.lock_name_indices().get(name).copied().unwrap_or(0);
        let owner = Arc::new(Owner {
            name: name.into(),
            turn: Arc::new(Semaphore::new(1)),
            unusable: OnceLock::new(),
            state: Mutex::new(OwnerState {
                name_index,
                ..OwnerState::default()
            }),
        });
        owner.mark_unusable_past_last_name(name_index);
        owners.insert(Arc::clone(&owner.name), Arc::clone(&owner));
        owner
    }

    /// Move the owner to its next name, because NIXL keeps the current one.
    /// The caller holds the owner's state.
    fn burn_name(&self, owner: &Owner, state: &mut OwnerState) {
        state.name_index += 1;
        self.lock_name_indices().insert(Arc::clone(&owner.name), state.name_index);
        owner.mark_unusable_past_last_name(state.name_index);
    }

    /// Copy the image's pixels out of the owner's memory. The caller holds
    /// the owner's turn.
    fn read(
        &self,
        image: &NixlImage,
        owner: &Arc<Owner>,
        permits: ReadPermits,
        deadline: Instant,
        cancelled: &CancellationToken,
    ) -> Result<Vec<u8>, ReadError> {
        let started = Instant::now();
        // Fails when the read is cancelled, or when no more than `reserve`
        // is left before the deadline.
        let go_on = |reserve: Duration| {
            if cancelled.is_cancelled() {
                return Err(ReadError::Cancelled);
            }
            if deadline.saturating_duration_since(Instant::now()) <= reserve {
                return Err(ReadError::Timeout(self.limits.read_timeout));
            }
            Ok(())
        };
        go_on(Duration::ZERO)?;
        self.unload_idle_owners();
        go_on(Duration::ZERO)?;
        self.load_remote(owner, image)?;
        let loaded = started.elapsed();

        let mut buffer = self.take_buffer(image.len)?;
        let mut local = XferDescList::new(MemType::Dram)?;
        local.add_desc(buffer.data.as_mut_ptr() as usize, image.len, 0);
        let mut remote = XferDescList::new(MemType::Dram)?;
        remote.add_desc(image.addr as usize, image.len, 0);
        let name = nixl_name(owner, &owner.lock_state());
        let request = match self.agent.create_xfer_req(XferOp::Read, &local, &remote, &name, None) {
            Ok(request) => request,
            Err(error) => {
                // NIXL did not find the image in the loaded sections, so
                // they are not what the reader expects. Start over.
                self.unload(owner, &mut owner.lock_state());
                self.put_buffer(buffer);
                return Err(error.into());
            }
        };
        if let Err(error) = go_on(MIN_TRANSFER_TIME) {
            drop(request);
            self.put_buffer(buffer);
            return Err(error);
        }
        // From here on the transfer may write to `buffer` until NIXL reports
        // that it ended, so an error hands everything to the stuck list.
        let finished = match self.agent.post_xfer_req(&request, None) {
            // `post_xfer_req` answers false when the transfer already finished.
            Ok(false) => Ok(()),
            Ok(true) => self.wait_for(&request, deadline.max(Instant::now() + MIN_TRANSFER_TIME)),
            Err(error) => Err(error.into()),
        };
        if let Err(error) = finished {
            self.set_aside(StuckRead {
                request,
                owner: Arc::clone(owner),
                buffer,
                _permits: permits,
            });
            return Err(error);
        }
        drop(request);

        let pixels = buffer.data[..image.len].to_vec();
        self.put_buffer(buffer);
        drop(permits);
        tracing::debug!(
            bytes = image.len,
            metadata_bytes = image.agent_metadata.len(),
            load_ms = loaded.as_secs_f64() * 1e3,
            read_ms = started.elapsed().as_secs_f64() * 1e3,
            "nixl media read"
        );
        Ok(pixels)
    }

    fn wait_for(&self, request: &XferRequest, deadline: Instant) -> Result<(), ReadError> {
        let mut pause = Duration::from_micros(10);
        loop {
            if self.agent.get_xfer_status(request)? == XferStatus::Success {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(ReadError::Timeout(self.limits.read_timeout));
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(MAX_POLL_SLEEP);
        }
    }

    /// Unload the least recently used owners that no read holds or waits for,
    /// until at most the limit remain. The map lock is not held across NIXL
    /// calls; the unloader holds the owner's turn instead.
    fn unload_idle_owners(&self) {
        let excess = self.lock_owners().len().saturating_sub(self.limits.max_remote_agents);
        for _ in 0..excess {
            let (owner, _turn) = {
                let owners = self.lock_owners();
                if owners.len() <= self.limits.max_remote_agents {
                    return;
                }
                // Only the map holds an idle owner, and new reads need the map.
                // An unusable owner stays, so that its reads keep failing with
                // the reason.
                let Some((owner, turn)) = owners
                    .values()
                    .filter(|owner| Arc::strong_count(owner) == 1 && owner.unusable.get().is_none())
                    .min_by_key(|owner| owner.lock_state().last_used)
                    .and_then(|owner| {
                        let turn = Arc::clone(&owner.turn).try_acquire_owned().ok()?;
                        Some((Arc::clone(owner), turn))
                    })
                else {
                    return;
                };
                (owner, turn)
            };
            self.unload(&owner, &mut owner.lock_state());
            let mut owners = self.lock_owners();
            // A read that took the owner meanwhile loads it again on its turn.
            if Arc::strong_count(&owner) == 2 {
                owners.remove(&owner.name);
            }
        }
    }

    /// Load the image's metadata into its remote agent, unless NIXL already
    /// has all of its sections. The caller holds the owner's turn.
    fn load_remote(&self, owner: &Owner, image: &NixlImage) -> Result<(), ReadError> {
        let mut state = owner.lock_state();
        state.last_used = Some(Instant::now());
        if let Some(reason) = owner.unusable.get() {
            return Err(ReadError::Unusable(reason.clone()));
        }
        let agent = &image.agent;
        let mut conflict =
            state.connection.as_ref().is_some_and(|loaded| *loaded != agent.connection);
        let mut known = state.connection.is_some();
        for section in &agent.sections {
            // Loaded sections do not overlap, so only the last one that
            // starts before the end of this section can overlap it.
            let end = section.addr + section.len;
            match state.sections.range(..end).next_back() {
                Some((_, loaded)) if loaded.addr + loaded.len > section.addr => {
                    // The same section with another key, or another section
                    // that overlaps this one (the owner freed memory and
                    // registered new memory there).
                    conflict |= loaded != section;
                }
                _ => known = false,
            }
        }
        if known && !conflict {
            return Ok(());
        }
        if conflict
            || state.sections.len() + agent.sections.len() > self.limits.max_sections_per_remote
        {
            self.unload(owner, &mut state);
        }
        let name = nixl_name(owner, &state);
        if state.connection.is_none() {
            if let Err(error) = self.agent.load_remote_md(&agent.serialize_as(&name, &[])) {
                // Drop what NIXL may hold for the agent, so the next read
                // starts clean.
                self.invalidate(&name);
                state.sections.clear();
                return Err(ReadError::Connect {
                    name: Arc::clone(&owner.name),
                    error,
                });
            }
            state.connection = Some(agent.connection.clone());
        }
        if let Err(error) = self.agent.load_remote_md(&agent.serialize_as(&name, &agent.sections)) {
            // NIXL erased the agent but kept its UCX connection under this
            // name, so the next load uses the next name.
            state.connection = None;
            state.sections.clear();
            self.burn_name(owner, &mut state);
            return Err(ReadError::Refused {
                name: Arc::clone(&owner.name),
                error,
            });
        }
        let sections = agent.sections.iter().map(|section| (section.addr, section.clone()));
        state.sections.extend(sections);
        Ok(())
    }

    /// Unload the owner's remote agent from NIXL and forget its sections. The
    /// caller holds the owner's turn.
    fn unload(&self, owner: &Owner, state: &mut OwnerState) {
        if state.connection.take().is_some() {
            self.invalidate(&nixl_name(owner, state));
        }
        state.sections.clear();
    }

    /// The caller holds that agent's turn.
    fn invalidate(&self, name: &str) {
        if let Err(error) = self.agent.invalidate_remote_md(name) {
            // A failed load already erased the agent, so NIXL may not know it.
            tracing::debug!(remote = %name, error = %error.as_report(), "NIXL remote agent was not loaded");
        }
    }

    fn set_aside(&self, stuck: StuckRead) {
        tracing::warn!(
            remote = %stuck.owner.name,
            bytes = stuck.buffer.data.len(),
            "NIXL read set aside until its transfer ends"
        );
        // Reads that wait for this owner fail now instead of timing out.
        stuck.owner.turn.close();
        self.lock_stuck().push(stuck);
        self.stuck_added.notify_all();
    }

    /// Release the stuck reads whose transfers ended, and return how many
    /// remain.
    fn reclaim_stuck_reads(&self) -> usize {
        let mut reads = self.lock_stuck();
        let mut ended = Vec::new();
        let mut active = Vec::new();
        for read in std::mem::take(&mut *reads) {
            match self.agent.get_xfer_status(&read.request) {
                Ok(XferStatus::Success) => ended.push((read, None)),
                Ok(XferStatus::InProgress) => active.push(read),
                Err(error) => ended.push((read, Some(error))),
            }
        }
        *reads = active;
        let remaining = reads.len();
        drop(reads);
        for (read, error) in ended {
            let StuckRead {
                request,
                owner,
                buffer,
                _permits,
            } = read;
            let bytes = buffer.data.len();
            drop(request);
            self.unload(&owner, &mut owner.lock_state());
            if let Some(error) = error {
                tracing::warn!(remote = %owner.name, error = %error.as_report(), "NIXL read failed");
                self.keep_buffer(buffer);
            } else {
                self.put_buffer(buffer);
            }
            // Start the owner over with a fresh turn. The closed turn keeps
            // reads away until the owner leaves the map.
            let mut owners = self.lock_owners();
            if owners.get(&owner.name).is_some_and(|current| Arc::ptr_eq(current, &owner)) {
                owners.remove(&owner.name);
            }
            drop(owners);
            tracing::info!(remote = %owner.name, bytes, "released a stuck NIXL read");
        }
        remaining
    }

    fn poll_stuck_reads_forever(&self) {
        loop {
            drop(
                self.stuck_added
                    .wait_while(self.lock_stuck(), |reads| reads.is_empty())
                    .unwrap_or_else(PoisonError::into_inner),
            );
            std::thread::sleep(STUCK_POLL_INTERVAL);
            self.reclaim_stuck_reads();
        }
    }

    fn take_buffer(&self, len: usize) -> Result<LocalBuffer, ReadError> {
        let capacity = buffer_capacity(len);
        if let Some(buffer) = self.buffers.lock().map_err(|_| ReadError::Poisoned)?.take(capacity) {
            return Ok(buffer);
        }
        let mut data = vec![0u8; capacity].into_boxed_slice();
        let handle = self.agent.register_memory(
            &DramRegion {
                ptr: data.as_mut_ptr(),
                len: capacity,
            },
            None,
        )?;
        Ok(LocalBuffer {
            _handle: handle,
            data,
        })
    }

    fn put_buffer(&self, buffer: LocalBuffer) {
        if let Ok(mut pool) = self.buffers.lock() {
            // A buffer that does not fit drops here, which deregisters it.
            let _ = pool.put(buffer.data.len(), buffer);
        }
    }

    fn keep_buffer(&self, buffer: LocalBuffer) {
        let mut pool = self.buffers.lock().unwrap_or_else(PoisonError::into_inner);
        pool.keep(buffer.data.len(), buffer);
    }

    fn lock_stuck(&self) -> std::sync::MutexGuard<'_, Vec<StuckRead>> {
        self.stuck.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Owner {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, OwnerState> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn mark_unusable_past_last_name(&self, name_index: u32) {
        if name_index >= MAX_NAMES_PER_REMOTE {
            let _ = self.unusable.set(format!(
                "NIXL refused remote agent `{}` {MAX_NAMES_PER_REMOTE} times; the owner must \
                 restart under a new agent name",
                self.name
            ));
        }
    }
}

/// A registered host buffer that transfers land in.
struct LocalBuffer {
    // Fields drop in order: deregister before the memory is freed.
    _handle: RegistrationHandle,
    data: Box<[u8]>,
}

/// Describes a `LocalBuffer` to NIXL.
#[derive(Debug)]
struct DramRegion {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: a `DramRegion` only lives during a `register_memory` call on a live
// buffer, and NIXL treats the region as opaque memory.
unsafe impl Send for DramRegion {}
// SAFETY: see above.
unsafe impl Sync for DramRegion {}

impl MemoryRegion for DramRegion {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for DramRegion {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

fn buffer_capacity(len: usize) -> usize {
    len.max(MIN_BUFFER_BYTES).next_power_of_two()
}

/// Free buffers by capacity, bounded by their total size.
struct BufferPool<B> {
    free: HashMap<usize, Vec<B>>,
    pooled_bytes: usize,
    max_bytes: usize,
}

impl<B> BufferPool<B> {
    fn new(max_bytes: usize) -> Self {
        Self {
            free: HashMap::new(),
            pooled_bytes: 0,
            max_bytes,
        }
    }

    fn take(&mut self, capacity: usize) -> Option<B> {
        let buffer = self.free.get_mut(&capacity)?.pop()?;
        self.pooled_bytes -= capacity;
        Some(buffer)
    }

    /// Keep `buffer` for reuse, or hand it back when the pool is full.
    fn put(&mut self, capacity: usize, buffer: B) -> Option<B> {
        if self.pooled_bytes + capacity > self.max_bytes {
            return Some(buffer);
        }
        self.keep(capacity, buffer);
        None
    }

    /// Keep `buffer` for reuse even past the byte limit.
    fn keep(&mut self, capacity: usize, buffer: B) {
        self.pooled_bytes += capacity;
        self.free.entry(capacity).or_default().push(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_sized_in_powers_of_two() {
        assert_eq!(buffer_capacity(1), MIN_BUFFER_BYTES);
        assert_eq!(buffer_capacity(MIN_BUFFER_BYTES + 1), 2 * MIN_BUFFER_BYTES);
        // 512 x 512 RGB
        assert_eq!(buffer_capacity(786_432), 1 << 20);
    }

    #[test]
    fn buffer_pool_stays_under_its_byte_limit() {
        let mut pool = BufferPool::new(3 * 1024);
        assert!(pool.put(1024, "a").is_none());
        assert!(pool.put(2048, "b").is_none());
        assert_eq!(
            pool.put(1024, "c"),
            Some("c"),
            "a full pool hands the buffer back"
        );
        assert_eq!(pool.take(2048), Some("b"));
        assert_eq!(pool.take(2048), None);
        assert!(pool.put(1024, "c").is_none());
        assert_eq!(pool.pooled_bytes, 2048);
        pool.keep(2048, "d");
        assert_eq!(pool.pooled_bytes, 4096, "a kept buffer may pass the limit");
    }

    /// A process-local peer that owns host memory, like a gRPC client that decodes images.
    struct Peer {
        name: String,
        agent: Agent,
        buffers: Vec<(RegistrationHandle, Box<[u8]>)>,
    }

    impl Peer {
        fn new() -> Self {
            let name = format!("peer-{}", uuid::Uuid::new_v4());
            let agent = Agent::new_configured(
                &name,
                &AgentConfig {
                    pthr_delay_us: PROGRESS_THREAD_DELAY_US,
                    ..AgentConfig::default()
                },
            )
            .unwrap();
            let (_mems, params) = agent.get_plugin_params("UCX").unwrap();
            agent.create_backend("UCX", &params).unwrap();
            Self {
                name,
                agent,
                buffers: Vec::new(),
            }
        }

        /// Register `pixels` as a 1-row RGB image, with `getLocalMD` metadata.
        fn publish(&mut self, pixels: &[u8]) -> NixlImage {
            self.publish_with(pixels, |agent, _| agent.get_local_md().unwrap())
        }

        /// Like `publish`, with the connection info and only this buffer's
        /// section, as a client that registers one region per image sends it.
        fn publish_region(&mut self, pixels: &[u8]) -> NixlImage {
            self.publish_with(pixels, region_metadata)
        }

        fn publish_with(
            &mut self,
            pixels: &[u8],
            metadata: impl FnOnce(&Agent, &DramRegion) -> Vec<u8>,
        ) -> NixlImage {
            let mut data: Box<[u8]> = pixels.into();
            let region = DramRegion {
                ptr: data.as_mut_ptr(),
                len: data.len(),
            };
            let handle = self.agent.register_memory(&region, None).unwrap();
            let agent_metadata = metadata(&self.agent, &region);
            self.buffers.push((handle, data));
            let image = image(agent_metadata, region.ptr as u64, pixels.len());
            // Request conversion parses real NIXL metadata the same way, and
            // the canonical form is NIXL's own.
            assert_eq!(image.agent.name, self.name);
            assert_eq!(
                image.agent.serialize(&image.agent.sections),
                image.agent_metadata
            );
            image
        }
    }

    /// A 1-row RGB image of the `len` bytes at `addr`.
    fn image(agent_metadata: Vec<u8>, addr: u64, len: usize) -> NixlImage {
        NixlImage {
            index: 0,
            agent: super::super::AgentMetadata::parse(&agent_metadata).unwrap(),
            agent_metadata,
            addr,
            width: (len / 3) as u32,
            height: 1,
            channels: 3,
            len,
            uuid: None,
        }
    }

    /// The connection info of `agent` and the section of `region`.
    fn region_metadata(agent: &Agent, region: &DramRegion) -> Vec<u8> {
        let partial =
            |descs: &nixl_sys::RegDescList<'_>| agent.get_local_partial_md(descs, None).unwrap();
        let connections = partial(&nixl_sys::RegDescList::new(MemType::Dram).unwrap());
        let mut descs = nixl_sys::RegDescList::new(MemType::Dram).unwrap();
        descs.add_storage_desc(region).unwrap();
        let section = partial(&descs);
        let metadata = [
            &connections[..memory_section_offset(&connections)],
            &section[memory_section_offset(&section)..],
        ]
        .concat();
        // The reader loads the connection info alone in NIXL's own form.
        let agent = super::super::AgentMetadata::parse(&metadata).unwrap();
        assert_eq!(agent.serialize(&[]), connections);
        metadata
    }

    /// Byte offset of the `MemSection` field of agent metadata.
    fn memory_section_offset(metadata: &[u8]) -> usize {
        let mut fields = super::super::SerDes(&metadata[b"nixlSerDes|".len()..]);
        fields.field("Agent", 256).unwrap();
        for _ in 0..fields.count("Conns").unwrap() {
            fields.field("t", 64).unwrap();
            fields.field("c", 1 << 20).unwrap();
        }
        metadata.len() - fields.0.len()
    }

    fn with_metadata(image: &NixlImage, agent: super::super::AgentMetadata) -> NixlImage {
        NixlImage {
            agent_metadata: agent.serialize(&agent.sections),
            agent,
            uuid: None,
            ..*image
        }
    }

    fn reader(max_sections_per_remote: usize) -> &'static NixlReader {
        reader_with(Limits {
            max_sections_per_remote,
            max_remote_agents: 2,
            read_timeout: READ_TIMEOUT,
        })
    }

    fn reader_with(limits: Limits) -> &'static NixlReader {
        let name = format!("reader-{}", uuid::Uuid::new_v4());
        Box::leak(Box::new(NixlReader::new(&name, limits).unwrap()))
    }

    /// Read `image` and return the content hash, which covers every pixel.
    async fn read(reader: &'static NixlReader, image: &NixlImage) -> vllm_chat::Result<String> {
        Ok(reader.read_image(image.clone()).await?.hash().to_owned())
    }

    /// The content hash of `pixels` as a 1-row RGB image.
    fn hash(pixels: &[u8]) -> String {
        let width = (pixels.len() / 3) as u32;
        DecodedImage::new(pixels.to_vec(), width, 1, 3, None).unwrap().hash().to_owned()
    }

    fn pixels(seed: u8) -> Vec<u8> {
        (0..64 * 48 * 3).map(|i| (i as u8).wrapping_mul(seed + 1)).collect()
    }

    fn loaded_sections(reader: &NixlReader, name: &str) -> usize {
        reader.owner(name).lock_state().sections.len()
    }

    async fn permits(reader: &NixlReader, owner: &Owner) -> ReadPermits {
        ReadPermits {
            _turn: Arc::clone(&owner.turn).acquire_owned().await.unwrap(),
            _kib: Arc::clone(&reader.kib).acquire_many_owned(1).await.unwrap(),
        }
    }

    /// Reads real NIXL memory in one process. Needs the NIXL library with its
    /// UCX plugin: `cargo test -p vllm-server --features nixl-media -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_reads_return_the_published_pixels() {
        let reader = reader(2);
        let mut peer = Peer::new();
        for seed in 0..4 {
            let expected = pixels(seed);
            let image = peer.publish(&expected);
            assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
            // Its sections are loaded now, so this read skips the load.
            assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
        }
        // Each metadata carries one more section; past the limit of two, the
        // reader unloads before it loads, so only the last load remains.
        assert_eq!(loaded_sections(reader, &peer.name), 4);
        assert!(reader.stuck.lock().unwrap().is_empty());
        assert_eq!(reader.reads.available_permits(), MAX_CONCURRENT_READS);
        assert_eq!(reader.kib.available_permits(), MAX_IN_FLIGHT_KIB);
        assert!(reader.buffers.lock().unwrap().pooled_bytes > 0);
    }

    /// Per-image metadata loads the connection once and adds one section per image.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_reads_accept_per_region_metadata() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let images: Vec<_> = (0..3)
            .map(|seed| {
                let expected = pixels(seed);
                let image = peer.publish_region(&expected);
                assert_eq!(image.agent.sections.len(), 1);
                (expected, image)
            })
            .collect();
        for (expected, image) in &images {
            assert_eq!(read(reader, image).await.unwrap(), hash(expected));
        }
        assert_eq!(loaded_sections(reader, &peer.name), 3);
    }

    /// A section that comes back with another memory key (new memory at a
    /// freed address) makes the reader unload before it loads, because NIXL
    /// refuses such a load. UCX reads a packed key without its length, so a key
    /// with one more byte still reads the same memory.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_reads_unload_before_a_conflicting_load() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let expected = pixels(1);
        let image = peer.publish_region(&expected);
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));

        let mut agent = image.agent.clone();
        agent.sections[0].key.push(0);
        let rekeyed = with_metadata(&image, agent);
        assert_eq!(read(reader, &rekeyed).await.unwrap(), hash(&expected));
        let owner = reader.owner(&peer.name);
        let state = owner.lock_state();
        let keys: Vec<_> = state.sections.values().map(|section| &section.key).collect();
        assert_eq!(keys, [&rekeyed.agent.sections[0].key]);
    }

    /// The owner freed a buffer and registered a larger one over it. NIXL finds
    /// the section of a transfer by the loaded section at or before its
    /// address, so the old section would hide the new one: the reader unloads
    /// the agent before it loads the new one.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_a_section_over_a_loaded_one_is_unloaded_first() {
        let reader = reader(1024);
        let peer = Peer::new();
        let mut data: Vec<u8> = (0..9216).map(|i| (i % 251) as u8).collect();
        let old = DramRegion {
            ptr: data.as_mut_ptr().wrapping_add(1536),
            len: 1536,
        };
        let old_handle = peer.agent.register_memory(&old, None).unwrap();
        let first = image(region_metadata(&peer.agent, &old), old.ptr as u64, 1536);
        assert_eq!(read(reader, &first).await.unwrap(), hash(&data[1536..3072]));
        drop(old_handle);

        let new = DramRegion {
            ptr: data.as_mut_ptr(),
            len: data.len(),
        };
        let _new_handle = peer.agent.register_memory(&new, None).unwrap();
        let next = image(
            region_metadata(&peer.agent, &new),
            new.ptr as u64 + 2304,
            1536,
        );
        assert_eq!(read(reader, &next).await.unwrap(), hash(&data[2304..3840]));
        assert_eq!(loaded_sections(reader, &peer.name), 1);
    }

    /// A read that has not started when its deadline passes, here because
    /// the blocking pool is busy, fails at the deadline and touches nothing.
    #[test]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    fn loopback_a_read_waiting_for_a_blocking_thread_times_out() {
        let reader = reader_with(Limits {
            read_timeout: Duration::from_millis(200),
            ..Limits::default()
        });
        let mut peer = Peer::new();
        let image = peer.publish_region(&pixels(1));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (release, hold) = std::sync::mpsc::channel::<()>();
            let (started, blocker_started) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                hold.recv().unwrap();
            });
            blocker_started.await.unwrap();

            let started = Instant::now();
            let error = tokio::time::timeout(Duration::from_secs(5), read(reader, &image))
                .await
                .expect("the read outlived its deadline")
                .unwrap_err();
            assert!(error.to_string().contains("a blocking thread"), "{error}");
            assert!(started.elapsed() >= Duration::from_millis(200));

            release.send(()).unwrap();
            blocker.await.unwrap();
            // The given-up read drops its permits once a thread picks it up.
            tokio::time::timeout(Duration::from_secs(5), async {
                while reader.kib.available_permits() != MAX_IN_FLIGHT_KIB
                    || reader.reads.available_permits() != MAX_CONCURRENT_READS
                {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("permits came back");
        });
        assert_eq!(loaded_sections(reader, &peer.name), 0);
        assert!(reader.stuck.lock().unwrap().is_empty());
    }

    /// A read past its deadline does not load. One with less than
    /// `MIN_TRANSFER_TIME` left loads but does not post, so it cannot become a
    /// stuck read.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_a_read_near_its_deadline_does_not_post() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let expected = pixels(1);
        let image = peer.publish_region(&expected);
        let owner = reader.owner(&peer.name);
        for (deadline, loaded) in [
            (Instant::now(), 0),
            (Instant::now() + MIN_TRANSFER_TIME / 2, 1),
        ] {
            let permits = permits(reader, &owner).await;
            let error = reader
                .read(&image, &owner, permits, deadline, &CancellationToken::new())
                .unwrap_err();
            assert!(matches!(error, ReadError::Timeout(_)), "{error}");
            assert_eq!(loaded_sections(reader, &peer.name), loaded);
        }
        assert!(reader.stuck.lock().unwrap().is_empty());
        assert_eq!(reader.kib.available_permits(), MAX_IN_FLIGHT_KIB);
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
    }

    /// NIXL keeps a refused name for the life of the process, so an owner that
    /// leaves the map, here when the reader reclaims its ended stuck read,
    /// comes back under its next name.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_a_reclaimed_owner_keeps_its_next_name() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let expected = pixels(1);
        let image = peer.publish_region(&expected);
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
        // The reader forgets the loaded section, so it loads the same section
        // with another key, which NIXL refuses.
        reader.owner(&peer.name).lock_state().sections.clear();
        let mut agent = image.agent.clone();
        agent.sections[0].key.push(0);
        let error = read(reader, &with_metadata(&image, agent)).await.unwrap_err();
        assert!(error.to_report_string().contains("refused"), "{error}");
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));

        let owner = reader.owner(&peer.name);
        assert_eq!(owner.lock_state().name_index, 1);
        let permits = permits(reader, &owner).await;
        let mut buffer = reader.take_buffer(image.len).unwrap();
        let mut local = XferDescList::new(MemType::Dram).unwrap();
        local.add_desc(buffer.data.as_mut_ptr() as usize, image.len, 0);
        let mut remote = XferDescList::new(MemType::Dram).unwrap();
        remote.add_desc(image.addr as usize, image.len, 0);
        let name = nixl_name(&owner, &owner.lock_state());
        let request = reader
            .agent
            .create_xfer_req(XferOp::Read, &local, &remote, &name, None)
            .unwrap();
        if reader.agent.post_xfer_req(&request, None).unwrap() {
            reader.wait_for(&request, Instant::now() + Duration::from_secs(5)).unwrap();
        }
        reader.set_aside(StuckRead {
            request,
            owner: Arc::clone(&owner),
            buffer,
            _permits: permits,
        });
        drop(owner);
        assert_eq!(reader.reclaim_stuck_reads(), 0);
        assert!(!reader.lock_owners().contains_key(peer.name.as_str()));
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
        assert_eq!(reader.owner(&peer.name).lock_state().name_index, 1);
    }

    /// A failed connection does not make the owner unusable: the next read
    /// loads the connection again.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_a_failed_connection_is_retried() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let expected = pixels(1);
        let image = peer.publish_region(&expected);
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));

        // NIXL still holds the connection info, which it refuses to replace.
        reader.owner(&peer.name).lock_state().connection = None;
        let mut agent = image.agent.clone();
        agent.connection.push(0);
        let error = read(reader, &with_metadata(&image, agent)).await.unwrap_err();
        assert!(
            error.to_string().contains("a later read tries again"),
            "{error}"
        );
        assert!(reader.owner(&peer.name).unusable.get().is_none());
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
    }

    /// When NIXL cannot create the transfer, the read fails before any
    /// transfer starts, unloads the agent, and releases what it held.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_read_of_unregistered_memory_fails_cleanly() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let expected = pixels(1);
        let image = peer.publish(&expected);
        let outside = NixlImage {
            addr: image.addr + image.len as u64,
            ..peer.publish(&pixels(2))
        };
        assert!(read(reader, &outside).await.is_err());
        assert!(reader.stuck.lock().unwrap().is_empty());
        assert_eq!(loaded_sections(reader, &peer.name), 0);
        assert_eq!(read(reader, &image).await.unwrap(), hash(&expected));
        assert_eq!(loaded_sections(reader, &peer.name), 1);
        assert_eq!(reader.kib.available_permits(), MAX_IN_FLIGHT_KIB);
    }

    /// Reads of an unusable owner fail at once with the reason, and a blocked
    /// owner fails its waiting reads.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_unusable_and_blocked_owners_fail_fast() {
        let reader = reader(1024);
        let mut peer = Peer::new();
        let image = peer.publish(&pixels(1));
        reader.owner(&peer.name).unusable.set("refused earlier".to_owned()).unwrap();
        let error = read(reader, &image).await.unwrap_err();
        assert!(error.to_string().contains("refused earlier"), "{error}");

        let mut other = Peer::new();
        let image = other.publish(&pixels(2));
        let owner = reader.owner(&other.name);
        let held = Arc::clone(&owner.turn).acquire_owned().await.unwrap();
        let waiting = tokio::spawn(async move { read(reader, &image).await });
        tokio::task::yield_now().await;
        owner.turn.close();
        let error = waiting.await.unwrap().unwrap_err();
        assert!(
            error.to_string().contains("blocked by a transfer"),
            "{error}"
        );
        drop(held);
        assert_eq!(reader.reads.available_permits(), MAX_CONCURRENT_READS);
    }

    /// Reads queued on one owner hold no global permit, so another owner's
    /// read goes through; dropping the queued reads releases everything.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_queue_on_one_owner_does_not_starve_another() {
        let reader = reader(1024);
        let mut busy = Peer::new();
        let busy_image = busy.publish(&pixels(1));
        let held = Arc::clone(&reader.owner(&busy.name).turn).acquire_owned().await.unwrap();
        let queued: Vec<_> = (0..2 * MAX_CONCURRENT_READS)
            .map(|_| {
                let image = busy_image.clone();
                tokio::spawn(async move { read(reader, &image).await })
            })
            .collect();
        tokio::task::yield_now().await;

        let mut other = Peer::new();
        let expected = pixels(2);
        let image = other.publish(&expected);
        let read_other = tokio::time::timeout(Duration::from_secs(5), read(reader, &image));
        assert_eq!(
            read_other.await.expect("other owner starved").unwrap(),
            hash(&expected)
        );

        for task in queued {
            task.abort();
            let _ = task.await;
        }
        drop(held);
        assert_eq!(reader.reads.available_permits(), MAX_CONCURRENT_READS);
        assert_eq!(reader.kib.available_permits(), MAX_IN_FLIGHT_KIB);
        assert_eq!(read(reader, &busy_image).await.unwrap(), hash(&pixels(1)));
    }

    /// Past two remote agents, a read first unloads the least recently used
    /// idle one, in NIXL too, and a later read of that one loads it again.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_the_least_recently_used_owner_is_unloaded() {
        let reader = reader(1024);
        let mut peers: Vec<_> = (0..3).map(|_| Peer::new()).collect();
        let images: Vec<_> = peers
            .iter_mut()
            .enumerate()
            .map(|(seed, peer)| {
                let expected = pixels(seed as u8);
                let image = peer.publish_region(&expected);
                (expected, image)
            })
            .collect();
        let owners = || {
            let mut names: Vec<_> =
                reader.lock_owners().keys().map(|name| name.to_string()).collect();
            names.sort();
            names
        };
        let names = |indices: &[usize]| {
            let mut names: Vec<_> = indices.iter().map(|&i| peers[i].name.clone()).collect();
            names.sort();
            names
        };
        for (expected, image) in &images {
            assert_eq!(read(reader, image).await.unwrap(), hash(expected));
        }
        assert_eq!(owners(), names(&[1, 2]));
        assert!(
            reader.agent.invalidate_remote_md(&format!("{}#0", peers[0].name)).is_err(),
            "NIXL still holds the unloaded agent"
        );

        let (expected, image) = &images[0];
        assert_eq!(read(reader, image).await.unwrap(), hash(expected));
        assert_eq!(owners(), names(&[0, 2]));
        assert!(reader.agent.invalidate_remote_md(&format!("{}#0", peers[1].name)).is_err());
    }

    /// When NIXL does not start, the start that server startup runs fails
    /// with the reason. The start is process-wide, so the test runs itself in
    /// a child process without NIXL plugins.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL library"]
    async fn a_failed_start_reports_the_reason() {
        const CHILD: &str = "VLLM_NIXL_FAILED_START_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let error = super::super::start().await.unwrap_err();
            assert!(
                format!("{error:#}").contains("failed to create the NIXL UCX backend"),
                "{error:#}"
            );
            assert!(NixlReader::get().is_none());
            return;
        }
        let test = concat!(module_path!(), "::a_failed_start_reports_the_reason");
        let (_crate, test) = test.split_once("::").unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", test])
            .env(CHILD, "1")
            .env("NIXL_PLUGIN_DIR", "/nonexistent/nixl-plugins")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        assert!(
            child.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&child.stderr)
        );
    }

    /// Concurrent reads of two owners while the reader unloads remote agents
    /// between them.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs the NIXL runtime with the UCX plugin"]
    async fn loopback_reads_stay_correct_under_concurrency() {
        let reader = reader(3);
        let mut published = Vec::new();
        let mut peers = Vec::new();
        for peer in 0..2u8 {
            let mut owner = Peer::new();
            for seed in 0..4u8 {
                let expected: Vec<u8> = (0..32 * 32 * 3).map(|i| (i as u8) ^ seed ^ peer).collect();
                let image = owner.publish(&expected);
                published.push((expected, image));
            }
            peers.push(owner);
        }
        let published = Arc::new(published);
        let tasks: Vec<_> = (0..4)
            .map(|task| {
                let published = Arc::clone(&published);
                tokio::spawn(async move {
                    for round in 0..16 {
                        let (expected, image) = &published[(task * 3 + round) % published.len()];
                        assert_eq!(read(reader, image).await.unwrap(), hash(expected));
                    }
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(reader.reads.available_permits(), MAX_CONCURRENT_READS);
    }
}
