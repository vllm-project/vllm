// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `Scheduler.schedule()` and its helpers.

use std::collections::VecDeque;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};

use super::connector::{KvConnectorBlockState, Requests};
use super::{PauseState, Scheduler, WaitingQueue};
use crate::kv_cache::{
    AllocateSlotsArgs, KvCacheBlocks, get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
};
use crate::sched::multimodal::{MultiModalData, get_mm_features_in_window};
use crate::sched::output::{
    CachedRequestData, EncoderInputUpdate, IterationDetails, NewRequestData, SchedulerOutput,
};
use crate::sched::py_int_set::PyIntSet;
use crate::sched::request::PrefillStatsExt;
use crate::sched::request::{EngineCoreEventType, Request, RequestStatus};
use crate::sched::request_queue::SchedulingPolicy;
use crate::sched::stats::KvFetchStage;

/// Requests the waiting-queue scan skipped this step (`skip_request`), to
/// re-queue ahead of older skipped items.
#[derive(Default)]
struct SkippedRequests {
    waiting: VecDeque<usize>,
    kv_holding: VecDeque<usize>,
    async_kv: VecDeque<usize>,
    /// Async loads started this step: newer than any block-holder.
    started_kv_loads: Vec<usize>,
}

/// What `_try_schedule_encoder_inputs` returns.
struct EncoderSchedule {
    encoder_inputs_to_schedule: Vec<usize>,
    num_new_tokens: i64,
    encoder_compute_budget: usize,
    external_load_encoder_input: Vec<usize>,
    duplicate_encoder_inputs: Vec<usize>,
}

impl Scheduler {
    /// Whether the drafts proposed for `request` after this step are sure to
    /// go unused.
    ///
    /// Drafts proposed after a step are verified in the request's next decode
    /// step. They go unused when no such step follows, in two cases:
    ///
    /// 1. This step is a non-final prefill chunk. It samples no token, and the
    ///    scheduler drops drafts proposed after a prefill chunk.
    /// 2. This step samples the request's last output token: even if every
    ///    draft is rejected, the request reaches `max_tokens` and finishes.
    ///
    /// Stop tokens and stop strings also end requests, but they cannot be known
    /// in advance. This returns True only when it is sure.
    ///
    /// The model runner proposes no drafts (K=0) in a step where this is True
    /// for every scheduled request on every DP rank.
    ///
    /// Args:
    ///     request: A request scheduled in this step.
    ///     num_computed_tokens: Tokens computed before this step, including
    ///         prefix-cache and KV-connector hits for a newly admitted request.
    ///     num_new_tokens: Tokens scheduled for this step.
    fn drafts_unused(request: &Request, num_computed_tokens: usize, num_new_tokens: usize) -> bool {
        // Case 1: a non-final prefill chunk stops before the request's last
        // known token, so this step samples nothing.
        if num_computed_tokens + num_new_tokens < request.num_tokens() {
            return true;
        }

        // Case 2: this step samples the request's last output token. Drafts may
        // all be rejected, so count only the output tokens the request is sure
        // to have once this step finishes: the verified ones, plus one from each
        // step that samples. This step samples one.
        let mut min_num_output_tokens = request.num_output_tokens() + 1;
        if request.num_output_placeholders > 0 {
            // Async scheduling: a step that samples is still in flight, and it
            // adds one more.
            min_num_output_tokens += 1;
        }
        // Reaching max_tokens finishes the request, so no step verifies the drafts.
        min_num_output_tokens >= request.max_tokens
    }

    /// Clip a prefill chunk so it ends where Mamba state must be cached.
    ///
    /// In "align" cache mode reusable SSM states are materialized at block
    /// boundaries, plus mandatory early stops (the prompt's partial-tail hash
    /// boundary, a detected shared-prefix junction). If a block is larger
    /// than the configured prefill chunk limit, intermediate chunks keep
    /// private running state until they reach the next cacheable position.
    fn mamba_block_aligned_split(
        &self,
        request: &Request,
        num_new_tokens: i64,
        num_new_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
    ) -> i64 {
        let start = (request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens) as i64;
        // Split only during prefill: `request.num_tokens - 1` extends this to
        // resumed requests replaying their output tokens.
        let num_prompt_tokens = request.num_prompt_tokens as i64;
        let prefill_end = num_prompt_tokens.max(request.num_tokens() as i64 - 1);
        if start >= prefill_end {
            return num_new_tokens;
        }

        let block_size = self.cache_block_size as i64;
        // The last block-aligned position whose state can be cached. With
        // Eagle, FullAttn prunes the last matching block, so back off one
        // block to avoid a Mamba cache miss.
        let ckpt_block_size = if self.mamba_partial_cache_hit {
            self.hash_block_size
        } else {
            self.block_size
        };
        let mut last_cache_position = get_mamba_prefill_checkpoint_position(
            request.num_tokens(),
            self.cache_block_size,
            self.use_eagle_block_drop,
            self.num_hash_lookahead_tokens,
        ) as i64;

        let mut end = start + num_new_tokens;
        let checkpoint_position = get_mamba_prefill_checkpoint_position(
            prefill_end as usize,
            ckpt_block_size,
            self.use_eagle_block_drop,
            self.num_hash_lookahead_tokens,
        );
        let use_internal_checkpoint = self.mamba_has_prefill_checkpoint_blocks
            && end >= prefill_end
            && is_mamba_prefill_checkpoint_valid(
                start as usize,
                end as usize,
                checkpoint_position,
                self.hash_block_size,
                self.cache_block_size,
                self.mamba_prefill_checkpoint_alignment,
            );
        if use_internal_checkpoint {
            last_cache_position = 0;
        }
        // Invariant: slot p holds the state after exactly (p + 1) * block_size
        // tokens. State is written at chunk ends, so chunk ends must be block
        // aligned. Exempt: the prompt's last chunk, whose slot decode advances
        // to the boundary. A block too wide for one chunk advances sub-block
        // and re-aligns at the next boundary.
        if end < prefill_end {
            let mut max_prefill_tokens = self.max_num_scheduled_tokens as i64;
            let long_prefill_threshold = self.long_prefill_token_threshold as i64;
            if long_prefill_threshold > 0 {
                max_prefill_tokens = max_prefill_tokens.min(long_prefill_threshold);
            }
            let aligned_end = end.div_euclid(block_size) * block_size;
            if aligned_end > start || block_size <= max_prefill_tokens {
                end = aligned_end;
            }
        }

        let next_block_boundary = (start.div_euclid(block_size) + 1) * block_size;
        let tail_boundary = if self.mamba_partial_cache_hit && !use_internal_checkpoint {
            get_mamba_prefill_checkpoint_position(
                request.num_prompt_tokens,
                self.hash_block_size,
                self.use_eagle_block_drop,
                self.num_hash_lookahead_tokens,
            ) as i64
        } else {
            0
        };
        let junction = request.shared_prefix_boundary as i64;
        // Block-floored: a sub-block junction's state is not separately cacheable.
        let block_floored = start + (junction - start).div_euclid(block_size) * block_size;
        // Past the prompt the manager writes nothing, so fall back to the
        // block-floored stop rather than dropping it: a resumed request replaying
        // output tokens can still observe a junction there.
        let junction_stop = if self.mamba_shared_prefix_checkpoint && junction <= num_prompt_tokens
        {
            junction
        } else {
            block_floored
        };
        let stops = [
            // Same invariant: a chunk starting mid-block stops at the boundary
            // rather than running past it.
            if start % block_size != 0 && !use_internal_checkpoint {
                next_block_boundary
            } else {
                0
            },
            // Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            // Stop at the prompt checkpoint unless the backend saves it internally.
            if last_cache_position < tail_boundary && tail_boundary < num_prompt_tokens {
                tail_boundary
            } else {
                0
            },
            // Marconi shared-prefix junction: cache its state so sibling
            // requests sharing the prefix can reuse it.
            if start < junction && junction < end {
                junction_stop
            } else {
                0
            },
        ];
        // Stop at the earliest mandatory position strictly inside the chunk.
        let end = stops.into_iter().filter(|&s| start < s && s < end).min().unwrap_or(end);
        (end - start).max(0)
    }

    fn get_local_prefix_cache_hit(
        &mut self,
        h: usize,
        max_cache_hit_length: Option<usize>,
    ) -> (KvCacheBlocks, usize, usize, bool) {
        let request = &self.requests[h];
        if let Some(connector) = &self.connector
            && connector.supports_divergent_local_hybrid_hits()
        {
            return self
                .kv_cache_manager
                .get_computed_blocks_for_connector(request, max_cache_hit_length);
        }

        let (blocks, num_local, shared_prefix_boundary) =
            self.kv_cache_manager.get_computed_blocks(request, max_cache_hit_length);
        (blocks, num_local, shared_prefix_boundary, false)
    }

    /// Never end a prefill chunk within num_prefill_lookahead of the
    /// prefill end.
    ///
    /// At a chunked-prefill boundary, the multi-module MTP drafter consumes
    /// the next num_prefill_lookahead known prefill tokens as draft inputs. A
    /// boundary closer to the end than that would make it fall back to
    /// sampled drafts, permanently polluting the trailing modules' KV caches.
    /// Either finish the prefill or leave at least num_prefill_lookahead for
    /// the next chunk. No-op for eagle-family drafters (lookahead 1).
    fn reserve_prefill_lookahead(
        &self,
        request: &Request,
        num_computed_tokens: usize,
        mut num_new_tokens: i64,
    ) -> i64 {
        let remaining = request.num_tokens() as i64 - num_computed_tokens as i64 - num_new_tokens;
        let num_prefill_lookahead = self.num_prefill_lookahead as i64;
        if 0 < remaining && remaining < num_prefill_lookahead {
            num_new_tokens -= num_prefill_lookahead - remaining;
        }
        num_new_tokens.max(0)
    }

    pub fn schedule(&mut self, throttle_prefills: bool) -> SchedulerOutput {
        self.current_step += 1;
        // NOTE(woosuk) on the scheduling algorithm:
        // There's no "decoding phase" nor "prefill phase" in the scheduler.
        // Each request just has the num_computed_tokens and
        // num_tokens_with_spec. num_tokens_with_spec =
        // len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        // At each step, the scheduler tries to assign tokens to the requests
        // so that each request's num_computed_tokens can catch up its
        // num_tokens_with_spec. This is general enough to cover
        // chunked prefills, prefix caching, speculative decoding,
        // and the "jump decoding" optimization in the future.

        let mut scheduled_new_reqs: Vec<usize> = Vec::new();
        let mut scheduled_running_reqs: Vec<usize> = Vec::new();
        let mut preempted_reqs: Vec<usize> = Vec::new();

        let mut iteration_details = self.enable_iteration_details.then(IterationDetails::default);

        let mut req_to_new_blocks: FxHashMap<usize, KvCacheBlocks> = FxHashMap::default();
        let mut num_scheduled_tokens: IndexMap<String, usize> = IndexMap::new();
        let mut token_budget = self.max_num_scheduled_tokens as i64;
        let draft_slots = self.draft_slots as i64;
        let mut input_budget = self.max_num_batched_tokens as i64;
        if self.pause_state == PauseState::PausedAll {
            // Do not schedule any requests when paused.
            token_budget = 0;
        }

        // Encoder-related.
        let mut scheduled_encoder_inputs: IndexMap<String, Vec<usize>> = IndexMap::new();
        let mut num_scheduled_encoder_inputs = 0;
        let mut encoder_compute_budget = self.max_num_encoder_input_tokens;
        // Spec decode-related.
        let mut scheduled_spec_decode_tokens: IndexMap<String, Vec<i32>> = IndexMap::new();
        // Whether a request scheduled this step can use the drafts proposed after
        // it. Stays set if that request is preempted later in the step.
        let mut drafts_needed = !self.can_skip_drafts;
        // Whether the running batch contains any prefill requests.
        let mut prefill_scheduled = false;
        // Whether any scheduled request has a synchronous connector KV load.
        let mut has_sync_kv_loads = false;

        // For logging.
        let scheduled_timestamp = (self.clock)();

        self.kv_cache_manager.new_step_starts();

        // DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        // all prefill compute unless saturated.
        let defer_prefills = (throttle_prefills && !self.prefill_capacity_bound)
            && self.running.iter().any(|&h| !self.requests[h].is_prefill_chunk);

        // `long_prefill_token_threshold` exists to stop a long prefill from
        // starving other requests of the token budget. When it is the only
        // request there is nobody to starve, so let it use the whole budget.
        let (num_running, num_waiting) = self.get_request_counts();
        let num_eligible_reqs = num_running + num_waiting;
        let mut long_prefill_token_threshold =
            if self.enable_chunked_prefill && num_eligible_reqs > 1 {
                self.long_prefill_token_threshold
            } else {
                0
            };
        if long_prefill_token_threshold > 0 && self.adaptive_long_prefill_threshold {
            // Floor the cap at a fair share of the input budget so it never
            // cuts a request below max_num_batched_tokens / num requests.
            long_prefill_token_threshold =
                long_prefill_token_threshold.max(input_budget as usize / num_eligible_reqs);
        }
        let long_prefill_token_threshold = long_prefill_token_threshold as i64;

        // First, schedule the RUNNING requests.
        let mut req_index = 0;
        while req_index < self.running.len() && token_budget > 0 {
            let h = self.running[req_index];
            if input_budget <= draft_slots {
                break;
            }

            let request = &self.requests[h];
            if request.num_output_placeholders > 0
                // This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                // Since output placeholders are also included in the computed tokens
                // count, we subtract (num_output_placeholders - 1) to remove any draft
                // tokens, so that we can be sure no further steps are needed even if
                // they are all rejected.
                && request.num_computed_tokens + 2
                    >= request.num_prompt_tokens + request.max_tokens + request.num_output_placeholders
            {
                // Async scheduling: Avoid scheduling an extra step when we are sure that
                // the previous step has reached request.max_tokens. We don't schedule
                // partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1;
                continue;
            }

            if self.current_step < request.next_decode_eligible_step {
                // PP+async: enforce `pp_size` steps between same-req decodes
                // to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1;
                continue;
            }

            if defer_prefills && request.is_prefill_chunk {
                // DP prefill balancing: defer this in-progress prefill chunk to a
                // cadence-aligned step; decodes still run to fill this step.
                req_index += 1;
                continue;
            }

            if let Some(ec_connector) = self.ec_connector.as_mut()
                && !request.mm_features.is_empty()
                && !ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            {
                req_index += 1;
                continue;
            }

            let mut num_new_tokens = request.num_tokens_with_spec() as i64
                + request.num_output_placeholders as i64
                - request.num_computed_tokens as i64;
            if 0 < long_prefill_token_threshold && long_prefill_token_threshold < num_new_tokens {
                num_new_tokens = long_prefill_token_threshold;
            }
            num_new_tokens = num_new_tokens.min(token_budget).min(input_budget - draft_slots);

            // Make sure the input position does not exceed the max model len.
            // This is necessary when using spec decoding.
            num_new_tokens = num_new_tokens.min(
                self.max_model_len as i64
                    - request.num_computed_tokens as i64
                    - self.num_sampled_tokens_per_step as i64,
            );

            // Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split {
                num_new_tokens = self.mamba_block_aligned_split(request, num_new_tokens, 0, 0);
            }

            // Schedule encoder inputs.
            let mut encoder_schedule = None;
            if request.has_encoder_inputs() {
                let num_computed_tokens = request.num_computed_tokens;
                let schedule = self.try_schedule_encoder_inputs(
                    h,
                    num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    self.num_prefill_lookahead,
                );
                num_new_tokens = schedule.num_new_tokens;
                encoder_schedule = Some(schedule);
            }

            // Multi-module MTP: avoid ending a prefill chunk within
            // num_prefill_lookahead of the prefill end.
            let request = &self.requests[h];
            let num_new_tokens = self.reserve_prefill_lookahead(
                request,
                request.num_computed_tokens,
                num_new_tokens,
            ) as usize;

            if num_new_tokens == 0 {
                // The request cannot be scheduled because one of the following
                // reasons:
                // 1. No new tokens to schedule. This may happen when
                //    (1) PP>1 and we have already scheduled all prompt tokens
                //    but they are not finished yet.
                //    (2) Async scheduling and the request has reached to either
                //    its max_total_tokens or max_model_len.
                // 2. The encoder budget is exhausted.
                // 3. The encoder cache is exhausted.
                // 4. Insufficient budget for a block-aligned chunk in hybrid
                //    models with mamba cache mode \"align\".
                // 5. Insufficient budget to keep a multi-module MTP prefill
                //    chunk out of the prefill-lookahead window.
                // NOTE(woosuk): Here, by doing `continue` instead of `break`,
                // we do not strictly follow the FCFS scheduling policy and
                // allow the lower-priority requests to be scheduled.
                req_index += 1;
                continue;
            }

            // Schedule newly needed KV blocks for the request.
            let mut new_blocks;
            loop {
                new_blocks = self
                    .kv_cache_manager
                    .allocate_slots(
                        &self.requests[h],
                        num_new_tokens,
                        AllocateSlotsArgs {
                            num_lookahead_tokens: self.num_lookahead_tokens,
                            ..Default::default()
                        },
                    )
                    .expect("allocate_slots arguments are valid");

                if new_blocks.is_some() {
                    // The request can be scheduled.
                    break;
                }

                if let Some(connector) = self.connector.as_mut()
                    && connector.has_pending_block_frees(&mut self.kv_cache_manager)
                {
                    break;
                }

                // The request cannot be scheduled.
                // Preempt the lowest-priority request.
                let preempted_req = if self.policy == SchedulingPolicy::Priority {
                    self.lowest_priority_running()
                } else {
                    *self.running.last().unwrap()
                };

                // A deferred free will not help with immediate allocation.
                if !self.request_blocks_can_be_freed(preempted_req) {
                    break;
                }

                if self.policy == SchedulingPolicy::Priority {
                    let victim_index =
                        self.running.iter().position(|&r| r == preempted_req).unwrap();
                    self.running.remove(victim_index);
                    if victim_index < req_index {
                        req_index -= 1;
                    }

                    if let Some(pos) =
                        scheduled_running_reqs.iter().position(|&r| r == preempted_req)
                    {
                        let preempted_req_id = self.requests[preempted_req].request_id.clone();
                        scheduled_running_reqs.remove(pos);
                        let restored =
                            num_scheduled_tokens.shift_remove(&preempted_req_id).unwrap();
                        token_budget += restored as i64;
                        input_budget += restored as i64 + draft_slots;
                        req_to_new_blocks.remove(&preempted_req);
                        scheduled_spec_decode_tokens.shift_remove(&preempted_req_id);
                        if let Some(preempted_encoder_inputs) =
                            scheduled_encoder_inputs.shift_remove(&preempted_req_id)
                            && !preempted_encoder_inputs.is_empty()
                        {
                            // Restore encoder compute budget if the preempted
                            // request had encoder inputs scheduled in this step.
                            let preempted = &self.requests[preempted_req];
                            let num_embeds_to_restore: usize = preempted_encoder_inputs
                                .iter()
                                .map(|&i| preempted.get_num_encoder_embeds(i))
                                .sum();
                            encoder_compute_budget += num_embeds_to_restore;
                            if iteration_details.is_some() {
                                num_scheduled_encoder_inputs -= preempted_encoder_inputs.len();
                            }
                        }
                    }
                } else {
                    self.running.pop();
                }

                self.preempt_request(
                    preempted_req,
                    scheduled_timestamp,
                    self.requires_kv_delivery,
                );
                preempted_reqs.push(preempted_req);
                if preempted_req == h {
                    // No more request to preempt. Cannot schedule this request.
                    break;
                }
            }

            let Some(new_blocks) = new_blocks else {
                if self.policy == SchedulingPolicy::Priority
                    && self.requests[h].status == RequestStatus::Preempted
                {
                    // The request was the lowest-priority running request and
                    // preempted itself. The requests after it have higher
                    // priority and may still fit (it just freed its blocks);
                    // it was removed from `self.running`, so `req_index`
                    // already points at the next one.
                    continue;
                }
                // Cannot schedule this request.
                break;
            };

            // Schedule the request.
            scheduled_running_reqs.push(h);
            let request = &mut self.requests[h];
            prefill_scheduled |= request.is_prefill_chunk;
            let request_id = request.request_id.clone();
            req_to_new_blocks.insert(h, new_blocks);
            num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);
            token_budget -= num_new_tokens as i64;
            input_budget -= num_new_tokens as i64 + draft_slots;
            req_index += 1;
            drafts_needed = drafts_needed
                || !Self::drafts_unused(request, request.num_computed_tokens, num_new_tokens);

            // Speculative decode related.
            if !request.spec_token_ids.is_empty() {
                let num_scheduled_spec_tokens = num_new_tokens as i64
                    + request.num_computed_tokens as i64
                    - request.num_tokens() as i64
                    - request.num_output_placeholders as i64;
                if num_scheduled_spec_tokens > 0 {
                    let mut spec_token_ids = request.spec_token_ids.clone();
                    spec_token_ids.truncate(num_scheduled_spec_tokens as usize);
                    scheduled_spec_decode_tokens.insert(request_id.clone(), spec_token_ids);
                }

                // New spec tokens will be set in `update_draft_token_ids` before the
                // next step when applicable.
                request.spec_token_ids = Vec::new();
            }

            // Encoder-related.
            if let Some(schedule) = encoder_schedule {
                if !schedule.encoder_inputs_to_schedule.is_empty() {
                    scheduled_encoder_inputs
                        .insert(request_id, schedule.encoder_inputs_to_schedule.clone());
                    if iteration_details.is_some() {
                        num_scheduled_encoder_inputs += schedule.encoder_inputs_to_schedule.len();
                    }
                    encoder_compute_budget = schedule.encoder_compute_budget;
                }
                self.allocate_scheduled_encoder_inputs(h, &schedule);
            }
        }

        // Record the LoRAs of all running requests, including those not
        // scheduled in this step, which can join a later batch.
        let mut scheduled_loras: FxHashSet<i64> = FxHashSet::default();
        if let Some(max_loras) = self.max_loras {
            scheduled_loras = self
                .running
                .iter()
                .filter_map(|&h| self.requests[h].lora_int_id.filter(|&id| id > 0))
                .collect();
            assert!(scheduled_loras.len() <= max_loras);
        }

        // Next, schedule the WAITING requests.
        if preempted_reqs.is_empty() && self.pause_state != PauseState::PausedAll {
            if self.cache_aware_window > 0 && !defer_prefills {
                self.reorder_waiting_by_cached_prefix();
            }

            let mut skipped = SkippedRequests::default();

            let mut num_async_kv_loads = 0;
            let max_async_kv_loads = self.max_async_kv_loads_per_step.unwrap_or(usize::MAX);
            // Cleared once self.async_kv_waiting's head can't start a load.
            let mut take_async_kv = true;
            loop {
                // Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                // in `running` but still hold a model-runner request slot.
                let num_running = self.running.len() + self.num_waiting_for_streaming_input;
                let loads_only = num_running >= self.max_num_active_reqs || token_budget <= 0;
                let request_queue = if !loads_only {
                    // Requests holding KV blocks are always drained first, and are
                    // the only ones admitted under PAUSED_NEW (e.g. async KV loads).
                    let mut request_queue = WaitingQueue::KvHolding;
                    if self.kv_holding_waiting.is_empty()
                        && self.pause_state == PauseState::Unpaused
                    {
                        request_queue = self.next_waiting_queue(take_async_kv);
                    }
                    request_queue
                } else if take_async_kv
                    && self.max_num_inflight_kv_loads.is_some_and(|n| n > 0)
                    && self.pause_state == PauseState::Unpaused
                {
                    // No running slot or token budget left: start async KV loads
                    // only, which take neither until their KV has arrived.
                    WaitingQueue::AsyncKv
                } else {
                    break;
                };
                if self.queue(request_queue).is_empty()
                    || (!loads_only && input_budget <= draft_slots)
                {
                    break;
                }

                let h = self.queue(request_queue).peek_request().unwrap();
                let request_id = self.requests[h].request_id.clone();

                let ready_to_schedule = self.handle_blocked_waiting_request(h);
                if !ready_to_schedule {
                    self.skip_request(request_queue, &mut skipped);
                    continue;
                }

                let request = &self.requests[h];
                if request.num_stale_output_tokens > 0 && !request.drop_stale_output {
                    // Deliverable stale output still in flight: resuming now
                    // could resample a position that output later delivers.
                    // It drains within the pipeline depth.
                    self.skip_request(request_queue, &mut skipped);
                    continue;
                }

                // Check that adding the request still respects the max_loras constraint.
                if let Some(max_loras) = self.max_loras
                    && let Some(lora_int_id) = request.lora_int_id
                    && scheduled_loras.len() == max_loras
                    && !scheduled_loras.contains(&lora_int_id)
                {
                    // Scheduling would exceed max_loras, skip.
                    self.skip_request(request_queue, &mut skipped);
                    continue;
                }

                let mut num_external_computed_tokens = 0;
                let mut load_kv_async = false;
                let (mut connector_prefix_cache_queries, mut connector_prefix_cache_hits) = (0, 0);
                let mut did_prefix_cache_lookup = false;
                let mut window_loaded = false;
                let mut new_computed_blocks;
                let mut num_new_local_computed_tokens;
                let mut num_computed_tokens;

                // Get already-cached tokens.
                if request.num_computed_tokens == 0 {
                    if request.resumable && self.num_hash_lookahead_tokens > 0 {
                        // Resident KV can retain a discarded streaming token's
                        // hash. Recomputing KV must use the current token sequence.
                        let request = &mut self.requests[h];
                        request.block_hashes.clear();
                        request.update_block_hashes();
                    }
                    did_prefix_cache_lookup = true;
                    let max_cache_hit_length = self.aux_output_connector.as_mut().map(|aux| {
                        aux.max_cache_hit(&self.requests[h], &mut self.kv_cache_manager)
                    });
                    let hit_diverged;
                    let shared_prefix_boundary;
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        shared_prefix_boundary,
                        hit_diverged,
                    ) = self.get_local_prefix_cache_hit(h, max_cache_hit_length);
                    self.requests[h].shared_prefix_boundary = shared_prefix_boundary;

                    // Get externally-cached tokens if using a KVConnector.
                    if let Some(connector) = self.connector.as_mut() {
                        let request = &mut self.requests[h];
                        // Present a block-aligned local hit to the connector so
                        // a strictly longer remote hit can supersede a local
                        // sub-block tail without racing its copy-on-write.
                        let partial_tail = num_new_local_computed_tokens % self.block_size;
                        let block_aligned_local = num_new_local_computed_tokens - partial_tail;
                        let (mut ext_tokens, mut load_async) = connector
                            .get_num_new_matched_tokens(
                                request,
                                block_aligned_local,
                                &mut self.kv_cache_manager,
                            );
                        if request.skip_reading_external_kv() {
                            (ext_tokens, load_async) = (Some(0), false);
                        }
                        load_kv_async = load_async;

                        let Some(mut ext_tokens) = ext_tokens else {
                            // The request cannot be scheduled because
                            // the KVConnector couldn't determine
                            // the number of matched tokens.
                            self.skip_request(request_queue, &mut skipped);
                            continue;
                        };

                        if self.prefix_replay_tokens > 0
                            && !(ext_tokens > 0 && self.load_restores_replay_window(h))
                        {
                            // DSV41 SWA bounded replay recomputes the hit's last
                            // window, from the block holding its first
                            // token; the sliding-window groups retire
                            // whole blocks below the window of the hit's
                            // next token. Hits end on a block boundary, or
                            // a hit ending one token short of one would
                            // retire the block the replay starts in. A load
                            // that carries the window replays nothing.
                            ext_tokens -= ext_tokens % self.block_size;
                            load_kv_async = load_kv_async && ext_tokens > 0;
                        }

                        if partial_tail > 0 && ext_tokens > partial_tail {
                            // Remote strictly exceeds the full local hit: drop the
                            // sub-block tail so no CoW is needed, and let the load
                            // cover it. Trim the partial block out of the local
                            // computed blocks so it is not adopted from the cache.
                            new_computed_blocks = self.kv_cache_manager.truncate_computed_blocks(
                                &new_computed_blocks,
                                block_aligned_local,
                            );
                            num_new_local_computed_tokens = block_aligned_local;
                            num_external_computed_tokens = ext_tokens;
                        } else if partial_tail > 0 {
                            // Remote does not exceed the full local hit: keep the
                            // local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0;
                            // Nothing to load remotely -> not an async-load step;
                            // clearing avoids the `load_kv_async` assert below.
                            load_kv_async = false;
                        } else {
                            num_external_computed_tokens = ext_tokens;
                        }

                        if let Some(aux) = self.aux_output_connector.as_mut() {
                            num_external_computed_tokens = aux.cap_external_hit(
                                &self.requests[h],
                                num_new_local_computed_tokens,
                                num_external_computed_tokens,
                                &mut self.kv_cache_manager,
                            );
                            load_kv_async &= num_external_computed_tokens > 0;
                        }

                        if hit_diverged && num_external_computed_tokens == 0 {
                            // No external tokens back the deeper local hit, so its
                            // resume boundary would have no valid Mamba state.
                            // Reconcile to the boundary every group agrees on.
                            let shared_prefix_boundary;
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                shared_prefix_boundary,
                            ) = self
                                .kv_cache_manager
                                .get_computed_blocks(&self.requests[h], max_cache_hit_length);
                            self.requests[h].shared_prefix_boundary = shared_prefix_boundary;
                        }

                        connector_prefix_cache_queries =
                            self.requests[h].num_tokens() - num_new_local_computed_tokens;
                        connector_prefix_cache_hits = num_external_computed_tokens;
                    }

                    // Total computed tokens (local + external).
                    num_computed_tokens =
                        num_new_local_computed_tokens + num_external_computed_tokens;
                    assert!(num_computed_tokens <= self.requests[h].num_tokens());
                    // DSV41 SWA bounded replay: a KV load that carries the
                    // window is taken as is; any other hit no longer than the
                    // replayed window would be recomputed in full and save
                    // nothing.
                    window_loaded =
                        num_external_computed_tokens > 0 && self.load_restores_replay_window(h);
                    if !window_loaded
                        && 0 < num_computed_tokens
                        && num_computed_tokens <= self.prefix_replay_tokens
                    {
                        new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks();
                        num_new_local_computed_tokens = 0;
                        num_external_computed_tokens = 0;
                        connector_prefix_cache_hits = 0;
                        num_computed_tokens = 0;
                        load_kv_async = false;
                    }

                    // Include encoder inputs needed to replay the prefix tail.
                    let mut ec_computed_tokens = num_computed_tokens;
                    if !window_loaded {
                        ec_computed_tokens =
                            num_computed_tokens.saturating_sub(self.prefix_replay_tokens);
                    }
                    if self.ec_transfer_pending(h, ec_computed_tokens) {
                        self.skip_request(request_queue, &mut skipped);
                        continue;
                    }

                    // Track first scheduled prefill, not post-preemption repeat prefills
                    let request = &mut self.requests[h];
                    if request.num_preemptions == 0
                        && let Some(prefill_stats) = &mut request.prefill_stats
                    {
                        assert!(num_computed_tokens <= request.num_prompt_tokens);
                        prefill_stats.set(
                            request.num_prompt_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        );
                    }
                    self.set_kv_fetch_stage(
                        h,
                        load_kv_async.then_some(KvFetchStage::WaitingToStart),
                    );
                } else {
                    // KVTransfer: WAITING reqs have num_computed_tokens > 0
                    // after async KV recvs are completed. A streaming-input
                    // session resumes here too, carrying whatever media its
                    // latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks();
                    num_new_local_computed_tokens = 0;
                    num_computed_tokens = request.num_computed_tokens;

                    if self.ec_transfer_pending(h, num_computed_tokens) {
                        self.skip_request(request_queue, &mut skipped);
                        continue;
                    }
                }

                let mut encoder_schedule = None;
                let mut pad_spec_decode = false;

                // SWA bounded replay: recompute the tail of the hit without
                // rewriting its cached KV. An async load replays once the KV
                // has arrived (_update_waiting_for_remote_kv).
                let mut num_replay_tokens = 0;

                if (load_kv_async
                    && (num_async_kv_loads >= max_async_kv_loads
                        || !self.admits_inflight_kv_load()))
                    || (loads_only && !load_kv_async)
                {
                    // Past an async KV load limit, or needs a running slot while none
                    // is free. It waits at the head of its queue; others may admit.
                    if request_queue != WaitingQueue::AsyncKv {
                        break;
                    }
                    take_async_kv = false;
                    continue;
                }

                let num_new_tokens: usize;
                if load_kv_async {
                    // KVTransfer: loading remote KV, do not allocate for new work.
                    assert!(num_external_computed_tokens > 0);
                    num_new_tokens = 0;
                } else if defer_prefills && num_computed_tokens + 1 < self.requests[h].num_tokens()
                {
                    // DP prefill balancing: defer this step's local prefill
                    // compute to a cadence-aligned step.
                    break;
                } else {
                    let request_token_budget = token_budget.min(input_budget - draft_slots);
                    if did_prefix_cache_lookup {
                        // Fresh admission: local and/or sync external hit.
                        num_replay_tokens =
                            self.mark_prefix_replay(h, num_computed_tokens, window_loaded);
                        num_computed_tokens -= num_replay_tokens;
                    }
                    let request = &self.requests[h];
                    // Number of tokens to be scheduled.
                    // We use `request.num_tokens` instead of
                    // `request.num_prompt_tokens` to consider the resumed
                    // requests, which have output tokens.
                    let mut n = (request.num_tokens() - num_computed_tokens) as i64;

                    // Pad new decode requests to uniform spec decoding size to
                    // preserve full cudagraph for this step.
                    // Not for diffusion where draft tokens can't be padded.
                    if (self.num_spec_tokens > 0 && self.dynamic_sd_lookup.is_none())
                        && self.num_sampled_tokens_per_step > 0
                        && n == 1
                        && !prefill_scheduled
                        && (!scheduled_running_reqs.is_empty() || num_computed_tokens > 0)
                    {
                        let padded_num_tokens = 1 + self.num_spec_tokens;
                        // Pad only when there is room for the sampled token(s).
                        if num_computed_tokens
                            + padded_num_tokens
                            + self.num_sampled_tokens_per_step
                            <= self.max_model_len
                        {
                            if padded_num_tokens as i64 > request_token_budget {
                                // Prefer to not schedule than schedule un-padded.
                                break;
                            }
                            n = padded_num_tokens as i64;
                            pad_spec_decode = true;
                        }
                    }

                    if 0 < long_prefill_token_threshold && long_prefill_token_threshold < n {
                        n = long_prefill_token_threshold;
                    }

                    // chunked prefill has to be enabled explicitly to allow
                    // pooling requests to be chunked
                    if !self.enable_chunked_prefill
                        && n > request_token_budget
                        && !self.is_mm_encoder_only
                    {
                        // If chunked_prefill is disabled,
                        // we can stop the scheduling here.
                        break;
                    }

                    n = n.min(request_token_budget);
                    assert!(n > 0);

                    // Apply Mamba alignment before encoder caps.
                    if self.need_mamba_block_aligned_split {
                        n = self.mamba_block_aligned_split(
                            request,
                            n,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        );
                        if n == 0 {
                            break;
                        }
                        if pad_spec_decode && n != 1 + self.num_spec_tokens as i64 {
                            // Alignment clipped the placeholder rows. The split
                            // aligns prefill chunks, but the padded tail rows are
                            // speculative positions, not prefill tokens. A padded
                            // request must keep all 1 + num_spec rows or the
                            // sampler's row count stops matching its query rows,
                            // so drop the padding instead of shortening it.
                            n = 1;
                            pad_spec_decode = false;
                        }
                    }

                    // Schedule encoder inputs.
                    if request.has_encoder_inputs() {
                        let schedule = self.try_schedule_encoder_inputs(
                            h,
                            num_computed_tokens,
                            n,
                            encoder_compute_budget,
                            self.num_prefill_lookahead,
                        );
                        n = schedule.num_new_tokens;
                        encoder_schedule = Some(schedule);
                    }

                    // Multi-module MTP: avoid ending a prefill chunk within
                    // num_prefill_lookahead of the prefill end.
                    let request = &self.requests[h];
                    n = self.reserve_prefill_lookahead(request, num_computed_tokens, n);

                    if request.has_encoder_inputs()
                        && !self.enable_chunked_prefill
                        && !self.is_mm_encoder_only
                        && n > 0
                        && n < (request.num_tokens() - num_computed_tokens) as i64
                    {
                        // Encoder cache pressure must not split a prompt that
                        // requires full prefill. Release references acquired by hits.
                        self.encoder_cache_manager.free(request);
                        n = 0;
                    }

                    if n == 0 {
                        // The request cannot be scheduled. Requeue it so it does
                        // not stall the requests behind it.
                        self.skip_request(request_queue, &mut skipped);
                        continue;
                    }
                    num_new_tokens = n as usize;
                }

                // During async KV load, no forward pass is run yet.
                // Allocate speculative lookahead slots later to avoid
                // mismatching local and remote block counts.
                let limit_lookahead_tokens = load_kv_async && self.num_lookahead_tokens > 0;
                let effective_lookahead_tokens = if limit_lookahead_tokens {
                    0
                } else {
                    self.num_lookahead_tokens
                };

                // Determine if we need to allocate cross-attention blocks.
                let request = &self.requests[h];
                let mut num_encoder_tokens = 0;
                if self.is_encoder_decoder
                    && request.has_encoder_inputs()
                    && let Some(schedule) = &encoder_schedule
                    && !schedule.encoder_inputs_to_schedule.is_empty()
                {
                    num_encoder_tokens = schedule
                        .encoder_inputs_to_schedule
                        .iter()
                        .map(|&i| request.get_num_encoder_embeds(i))
                        .sum();
                }

                let mut reserved_blocks = 0;
                if load_kv_async {
                    // An async load holds its blocks for the whole transfer with
                    // no forward progress and isn't preemptible here. Admit it
                    // only if it fits in (free - other in-flight reservations)
                    // plus its own spec decode step blocks, to avoid deadlock and
                    // predictable preemptions.
                    reserved_blocks =
                        self.inflight_prefill_reserved_blocks() + self.spec_decode_step_blocks;
                }

                // Replayed tokens are already counted in the adopted hit; a
                // chunk that ends inside the replayed range needs no new slots.
                let num_tokens_past_hit = num_new_tokens.saturating_sub(num_replay_tokens);
                let skip_zeroing_group_ids = match self.connector.as_mut() {
                    Some(connector) if load_kv_async => {
                        connector.get_loaded_kv_cache_group_ids(&self.requests[h])
                    }
                    _ => Vec::new(),
                };
                let has_scheduled_reqs = !self.running.is_empty();
                let new_blocks = self
                    .kv_cache_manager
                    .allocate_slots(
                        &self.requests[h],
                        num_tokens_past_hit,
                        AllocateSlotsArgs {
                            num_new_computed_tokens: num_new_local_computed_tokens,
                            new_computed_blocks: Some(&new_computed_blocks),
                            num_lookahead_tokens: effective_lookahead_tokens,
                            num_external_computed_tokens,
                            delay_cache_blocks: load_kv_async,
                            num_encoder_tokens,
                            full_sequence_must_fit: self.scheduler_reserve_full_isl,
                            // TODO: Python passes a negative reservation as is
                            // (Mamba "align" can count negative blocks); the KV
                            // cache manager takes a `usize`.
                            reserved_blocks: reserved_blocks.max(0) as usize,
                            has_scheduled_reqs: Some(has_scheduled_reqs),
                            skip_zeroing_group_ids: &skip_zeroing_group_ids,
                        },
                    )
                    .expect("allocate_slots arguments are valid");

                if new_blocks.is_none() {
                    // The request cannot be scheduled.

                    // NOTE: we need to untouch the request from the encode cache
                    // manager
                    let request = &self.requests[h];
                    if request.has_encoder_inputs() {
                        self.encoder_cache_manager.free(request);
                    }
                    break;
                }

                // KVTransfer: the connector uses this info to determine
                // if a load is needed. Note that
                // This information is used to determine if a load is
                // needed for this request.
                if let Some(connector) = self.connector.as_mut() {
                    let defer_load = load_kv_async
                        && self.defer_async_loads
                        && self.sched_step_seq > self.processed_step_seq;
                    if defer_load {
                        self.requests[h].last_sched_seq = self.sched_step_seq;
                        self.deferred_loads.insert(
                            request_id.clone(),
                            (
                                self.sched_step_seq,
                                num_external_computed_tokens,
                                connector_prefix_cache_queries,
                                connector_prefix_cache_hits,
                            ),
                        );
                    } else {
                        let blocks = self.kv_cache_manager.get_blocks(&request_id);
                        connector.update_state_after_alloc(
                            &self.requests[h],
                            &blocks,
                            &mut self.kv_cache_manager,
                            num_external_computed_tokens,
                        );
                    }
                    if let Some(stats) = &mut self.connector_prefix_cache_stats
                        && connector_prefix_cache_queries != 0
                        && !defer_load
                    {
                        let request = &self.requests[h];
                        let preempted = request.num_preemptions > 0;
                        // The connector knows the tier only after the load
                        // plan above exists.
                        let hits_by_source = if connector_prefix_cache_hits > 0 && !preempted {
                            connector.get_external_cache_hit_sources(
                                request,
                                connector_prefix_cache_hits,
                            )
                        } else {
                            None
                        };
                        stats.record(
                            connector_prefix_cache_queries,
                            connector_prefix_cache_hits,
                            preempted,
                            hits_by_source,
                        );
                    }
                }

                // Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup {
                    self.kv_cache_manager.record_prefix_cache_stats(
                        &self.requests[h],
                        num_new_local_computed_tokens,
                    );
                }

                if load_kv_async {
                    // If loading async, allocate memory and put request
                    // into the WAITING_FOR_REMOTE_KV state.
                    let request = &mut self.requests[h];
                    request.status = RequestStatus::WaitingForRemoteKvs;
                    num_async_kv_loads += 1;
                    // Set num_computed_tokens even though KVs are not yet loaded.
                    // request.num_computed_tokens will not be used anywhere until
                    // the request finished the KV transfer.
                    //
                    // If a transfer error is reported by the connector,
                    // request.num_computed_tokens will be re-set accordingly in
                    // _update_requests_with_invalid_blocks.
                    //
                    // When the transfer is finished, either successfully or not,
                    // request.num_computed_tokens will correctly reflect the number
                    // of computed tokens.
                    // _update_waiting_for_remote_kv will then cache
                    // only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens;
                    if let Some(aux) = self.aux_output_connector.as_mut() {
                        aux.request_loading(&self.requests[h], num_computed_tokens);
                    }
                    self.reserve_parked(h);
                    self.set_kv_fetch_stage(h, Some(KvFetchStage::InProgress));
                    self.skip_request(request_queue, &mut skipped);
                    continue;
                }

                let (queue, store) = self.queue_mut(request_queue);
                let popped = queue.pop_request(store);
                debug_assert_eq!(popped, Some(h));
                self.deferred_waiting.remove(&h);
                self.running.push(h);
                if num_external_computed_tokens > 0 {
                    // load_kv_async is False here
                    has_sync_kv_loads = true;
                }
                let request = &mut self.requests[h];
                if self.log_stats {
                    request.record_event(EngineCoreEventType::Scheduled, scheduled_timestamp);
                }
                // A resumed request is re-sent in full, like a new one: its
                // blocks were freed and its worker state dropped at preemption.
                assert!(
                    matches!(
                        request.status,
                        RequestStatus::Waiting | RequestStatus::Preempted
                    ),
                    "Invalid request status: {}",
                    request.status
                );
                scheduled_new_reqs.push(h);

                if self.max_loras.is_some()
                    && let Some(lora_int_id) = request.lora_int_id
                {
                    scheduled_loras.insert(lora_int_id);
                }
                req_to_new_blocks.insert(h, self.kv_cache_manager.get_blocks(&request_id));
                num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);
                token_budget -= num_new_tokens as i64;
                input_budget -= num_new_tokens as i64 + draft_slots;
                request.status = RequestStatus::Running;
                request.num_computed_tokens = num_computed_tokens;
                self.set_kv_fetch_stage(h, None);
                let request = &self.requests[h];
                drafts_needed = drafts_needed
                    || !Self::drafts_unused(request, num_computed_tokens, num_new_tokens);
                if pad_spec_decode {
                    assert_eq!(num_new_tokens, 1 + self.num_spec_tokens);
                    scheduled_spec_decode_tokens
                        .insert(request_id.clone(), vec![-1; self.num_spec_tokens]);
                }
                // Track requests still prefilling after this chunk. A promoted load
                // also stays tracked until _update_after_schedule drops it.
                if num_computed_tokens + num_new_tokens < request.num_tokens()
                    || self.parked_reservations.contains_key(&h)
                {
                    self.inflight_prefills.insert(h);
                }
                self.unreserve_parked(h);
                // Encoder-related.
                if let Some(schedule) = encoder_schedule {
                    if !schedule.encoder_inputs_to_schedule.is_empty() {
                        scheduled_encoder_inputs
                            .insert(request_id, schedule.encoder_inputs_to_schedule.clone());
                        if iteration_details.is_some() {
                            num_scheduled_encoder_inputs +=
                                schedule.encoder_inputs_to_schedule.len();
                        }
                        encoder_compute_budget = schedule.encoder_compute_budget;
                    }
                    self.allocate_scheduled_encoder_inputs(h, &schedule);
                }
            }

            // re-queue requests skipped in this pass ahead of older skipped items.
            if !skipped.kv_holding.is_empty() {
                self.kv_holding_waiting.prepend_requests(skipped.kv_holding, &self.requests);
            }
            for h in skipped.started_kv_loads {
                self.kv_holding_waiting.add_request(h, &self.requests);
            }
            if !skipped.waiting.is_empty() {
                self.waiting.prepend_requests(skipped.waiting, &self.requests);
            }
            if !skipped.async_kv.is_empty() {
                self.async_kv_waiting.prepend_requests(skipped.async_kv, &self.requests);
            }

            // DP prefill balancing: on a step that admitted prefills (release),
            // record whether it was capacity-bound.
            if !defer_prefills {
                let (_, num_waiting) = self.get_request_counts();
                self.prefill_capacity_bound = num_waiting > self.deferred_waiting.len();
            }
        }

        // Check if the scheduling constraints are satisfied.
        let total_num_scheduled_tokens: usize = num_scheduled_tokens.values().sum();
        assert!(total_num_scheduled_tokens <= self.max_num_scheduled_tokens);

        assert!(token_budget >= 0);
        assert!(input_budget >= 0);
        assert!(self.running.len() <= self.max_num_running_reqs);
        // Since some requests in the RUNNING queue may not be scheduled in
        // this step, the total number of scheduled requests can be smaller than
        // len(self.running).
        assert!(scheduled_new_reqs.len() + scheduled_running_reqs.len() <= self.running.len());

        // Construct the scheduler output.
        let new_reqs_data: Vec<NewRequestData> = scheduled_new_reqs
            .iter()
            .map(|h| {
                let request = &self.requests[*h];
                NewRequestData::from_request(
                    request,
                    req_to_new_blocks[h].get_block_ids(false, None).unwrap(),
                    request.prefill_token_ids().to_vec(),
                )
            })
            .collect();

        let cached_reqs_data = self.make_cached_request_data(
            &scheduled_running_reqs,
            &num_scheduled_tokens,
            &scheduled_spec_decode_tokens,
            &req_to_new_blocks,
            iteration_details.as_mut(),
        );

        if let Some(details) = &mut iteration_details {
            // Decode requests are counted in _make_cached_request_data()
            // Other scheduled requests - including new requests - are prefill
            details.num_prefill_requests = num_scheduled_tokens.len() - details.num_decode_requests;
            details.num_prefill_tokens = total_num_scheduled_tokens - details.num_decode_tokens;
            details.num_encoder_inputs = num_scheduled_encoder_inputs;
            details.num_encoder_output_tokens =
                self.max_num_encoder_input_tokens - encoder_compute_budget;
        }

        // Mamba "align" boundary states must be handed off with exact block ids;
        // they cannot be reconstructed from a connector's append-only block
        // table. Drained every step so stale offers cannot accumulate.
        let boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads();

        let (kv_cache_block_copies, cow_retained_blocks) =
            self.kv_cache_manager.take_kv_cache_block_copies();
        if !kv_cache_block_copies.is_empty() {
            // The copies run with this step's execution; the first non-empty
            // step at or after it gets seq `sched_step_seq + 1` (0-token steps
            // do not advance the seq), and its completion implies the copies
            // have run.
            self.free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1);
        }

        // Dynamic speculative decoding: compute optimal K
        let mut num_spec_tokens_to_schedule = self.num_spec_tokens;
        if let Some(lookup) = &self.dynamic_sd_lookup
            && !num_scheduled_tokens.is_empty()
        {
            num_spec_tokens_to_schedule = lookup[num_scheduled_tokens.len()];
        }

        let free_encoder_mm_hashes = self.encoder_cache_manager.get_freed_mm_hashes();
        let encoder_input_updates = self.get_encoder_input_updates(
            &num_scheduled_tokens,
            &scheduled_new_reqs,
            &scheduled_encoder_inputs,
        );
        let new_block_ids_to_zero = self.get_new_block_ids_to_zero();
        let mut scheduler_output = SchedulerOutput {
            scheduled_new_reqs: new_reqs_data,
            scheduled_cached_reqs: cached_reqs_data,
            num_scheduled_tokens,
            total_num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            scheduled_encoder_inputs,
            iteration_details,
            preempted_req_ids: Some(std::mem::take(&mut self.reset_preempted_req_ids)),
            // finished_req_ids is an existing state in the scheduler,
            // instead of being newly scheduled in this step.
            // It contains the request IDs that are finished in between
            // the previous and the current steps.
            finished_req_ids: std::mem::take(&mut self.finished_req_ids),
            free_encoder_mm_hashes,
            encoder_input_updates,
            new_block_ids_to_zero,
            has_sync_kv_loads,
            kv_cache_block_copies: (!kv_cache_block_copies.is_empty())
                .then_some(kv_cache_block_copies),
            num_spec_tokens_to_schedule,
            skip_drafts: !drafts_needed,
            ..Default::default()
        };

        // NOTE(Kuntai): this function is designed for multiple purposes:
        // 1. Plan the KV cache store
        // 2. Wrap up all the KV cache load / save ops into an opaque object
        // 3. Clear the internal states of the connector
        if let Some(connector) = self.connector.as_mut() {
            // Any request scheduled this step can become a connector job now,
            // not only the ones that were allocated blocks: a store save lands
            // on the step that fills a block, which allocated none.
            let mut req_ids: IndexSet<String> =
                scheduler_output.num_scheduled_tokens.keys().cloned().collect();
            req_ids.extend(
                boundary_state_offloads
                    .keys()
                    .filter(|req_id| self.requests.handle(req_id).is_some())
                    .cloned(),
            );
            let mut block_state = KvConnectorBlockState {
                req_ids,
                kv_cache_manager: &mut self.kv_cache_manager,
                boundary_state_offloads,
                requests: Requests(&self.requests),
            };
            let meta = connector.build_connector_meta(&scheduler_output, &mut block_state);
            scheduler_output.kv_connector_metadata = Some(meta);
        }

        if let Some(aux) = self.aux_output_connector.as_mut() {
            scheduler_output.aux_output_connector_metadata = Some(aux.build_connector_meta(
                &scheduler_output,
                Requests(&self.requests),
                &mut self.kv_cache_manager,
            ));
        }

        // Build the connector meta for ECConnector
        if let Some(ec_connector) = self.ec_connector.as_mut() {
            scheduler_output.ec_connector_metadata = Some(
                ec_connector.build_connector_meta(&scheduler_output, Requests(&self.requests)),
            );
        }

        // Advance the fence only for non-empty steps (those that actually
        // write KV and have their output processed later in update_from_output).
        if self.defer_block_free && total_num_scheduled_tokens > 0 {
            self.sched_step_seq += 1;
        }

        self.update_after_schedule(&mut scheduler_output);
        scheduler_output
    }

    /// The running request with the largest `(priority, arrival_time)`, the
    /// first one on ties (Python's `max`).
    fn lowest_priority_running(&self) -> usize {
        let mut best = self.running[0];
        for &h in &self.running[1..] {
            let (r, b) = (&self.requests[h], &self.requests[best]);
            if (r.priority, r.arrival_time) > (b.priority, b.arrival_time) {
                best = h;
            }
        }
        best
    }

    /// `skip_request` of the waiting scan: pop the head of `from_queue` and
    /// set it aside to re-queue after the scan.
    fn skip_request(&mut self, from_queue: WaitingQueue, skipped: &mut SkippedRequests) {
        let (queue, store) = self.queue_mut(from_queue);
        let request_to_skip = queue.pop_request(store).expect("queue is not empty");
        self.deferred_waiting.insert(request_to_skip);
        let holds_blocks = Self::holds_kv_blocks(&self.requests[request_to_skip]);
        if holds_blocks && from_queue != WaitingQueue::KvHolding {
            skipped.started_kv_loads.push(request_to_skip);
        } else if holds_blocks {
            skipped.kv_holding.push_front(request_to_skip);
        } else if from_queue == WaitingQueue::AsyncKv {
            skipped.async_kv.push_front(request_to_skip);
        } else {
            skipped.waiting.push_front(request_to_skip);
        }
    }

    fn get_new_block_ids_to_zero(&mut self) -> Option<Vec<u32>> {
        // Drain new attention block ids every step so the manager-side list
        // does not grow unbounded; only kv-cache zeroing consumes them.
        let new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids();
        if !self.needs_kv_cache_zeroing || new_block_ids_to_zero.is_empty() {
            return None;
        }
        Some(new_block_ids_to_zero)
    }

    /// Preempt a request and put it back to the waiting queue.
    ///
    /// NOTE: The request should be popped from the running queue outside of this
    /// method.
    ///
    /// drop_stale_output: drop (rather than deliver) any in-flight output; used
    /// by reset_prefix_cache, whose same-step resume would otherwise deliver
    /// tokens out of order, and for connectors with a pending KV hand-off,
    /// which the preemption's block free would leave without valid KV.
    pub(super) fn preempt_request(&mut self, h: usize, timestamp: f64, drop_stale_output: bool) {
        assert_eq!(
            self.requests[h].status,
            RequestStatus::Running,
            "Only running requests can be preempted"
        );
        if let Some(aux) = self.aux_output_connector.as_mut() {
            aux.request_finished(&self.requests[h], &mut self.kv_cache_manager);
        }
        self.free_request_blocks(h);
        let request = &mut self.requests[h];
        self.encoder_cache_manager.free(request);
        self.pending_encoder_input_ids.remove(&request.request_id);
        self.inflight_prefills.remove(&h);
        request.status = RequestStatus::Preempted;
        request.num_computed_tokens = 0;
        if !request.spec_token_ids.is_empty() {
            request.spec_token_ids = Vec::new();
        }
        // Async scheduling: mark all in-flight output as stale. Its tokens are
        // still delivered on return (dropping them would perturb spec-decode
        // acceptance) but must not mutate the reset counters; each step drains
        // its share in update_from_output. num_in_flight_tokens already
        // includes any undrained stale share, so assign rather than accumulate.
        // An undrained drop-mode share stays dropped: its positions have
        // already been resampled.
        request.drop_stale_output =
            drop_stale_output || (request.drop_stale_output && request.num_stale_output_tokens > 0);
        request.num_stale_output_tokens = request.num_in_flight_tokens;
        request.num_output_placeholders = 0;
        request.num_preemptions += 1;
        if self.log_stats {
            request.record_event(EngineCoreEventType::Preempted, timestamp);
        }

        // Put the request back to the waiting queue.
        let request_id = request.request_id.clone();
        self.waiting.prepend_request(h, &self.requests);
        self.reset_preempted_req_ids.insert(request_id);
    }

    pub(super) fn update_after_schedule(&mut self, scheduler_output: &mut SchedulerOutput) {
        // Advance the number of computed tokens for the request AFTER
        // the request is scheduled.
        // 1. The scheduler_output of the current step has to include the
        //    original number of scheduled tokens to determine input IDs.
        // 2. Advance the number of computed tokens here allowing us to
        //    schedule the prefill request again immediately in the next
        //    scheduling step.
        // 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        //    computed tokens will be adjusted in update_from_output.
        for (req_id, &num_scheduled_token) in &scheduler_output.num_scheduled_tokens {
            let h = self.requests.handle(req_id).expect("scheduled request exists");
            let request = &mut self.requests[h];
            request.num_computed_tokens += num_scheduled_token;
            request.num_in_flight_tokens += num_scheduled_token;
            if self.defer_block_free {
                // Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq;
            }
            request.is_prefill_chunk = request.num_computed_tokens
                < request.num_tokens() + request.num_output_placeholders;
            scheduler_output.has_structured_output_requests |=
                request.use_structured_output() && !request.is_prefill_chunk;
            // Drop from the in-flight-prefill set once it's no longer prefilling.
            if !request.is_prefill_chunk {
                self.inflight_prefills.remove(&h);
            }
        }

        // Clear the finished and preempted request IDs.
        // NOTE: We shouldn't just clear() here because it will also affect
        // the scheduler output.
        self.finished_req_ids = IndexSet::new();
        self.reset_preempted_req_ids = IndexSet::new();

        if let Some(decode_stagger) = self.decode_stagger {
            self.async_update_after_schedule(scheduler_output, decode_stagger);
        }
    }

    /// `AsyncScheduler._update_after_schedule`, after the base class's.
    fn async_update_after_schedule(
        &mut self,
        scheduler_output: &mut SchedulerOutput,
        decode_stagger: u64,
    ) {
        let spec_decode_tokens = &scheduler_output.scheduled_spec_decode_tokens;
        // Use the latest num of scheduled draft tokens in next step as placeholder.
        let spec_token_placeholders = vec![-1; scheduler_output.num_spec_tokens_to_schedule];
        let mut pending_structured_output_tokens =
            scheduler_output.pending_structured_output_tokens;
        let mut structured_output_spec_decode = scheduler_output.has_structured_output_spec_decode;
        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            let h = self.requests.handle(req_id).expect("scheduled request exists");
            let request = &mut self.requests[h];
            if request.is_prefill_chunk {
                continue;
            }

            // The request will generate num_sampled_tokens_per_step new tokens
            // plus num_spec_tokens in this scheduling step. Diffusion has no AR
            // bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            // (spec) tokens.
            let cur_num_spec_tokens = spec_decode_tokens.get(req_id).map_or(0, Vec::len);
            if scheduler_output.has_structured_output_requests && request.use_structured_output() {
                pending_structured_output_tokens |= request.num_output_placeholders > 0;
                structured_output_spec_decode |= cur_num_spec_tokens > 0;
            }
            request.num_output_placeholders +=
                self.num_sampled_tokens_per_step + cur_num_spec_tokens;
            // Add placeholders for the new draft/spec tokens.
            // We will update the actual spec token ids in the worker process.
            request.spec_token_ids = spec_token_placeholders.clone();

            // Set the next step index in which this request is eligible to be
            // scheduled for decode (for PP microbatching).
            request.next_decode_eligible_step = self.current_step + decode_stagger;
        }

        scheduler_output.pending_structured_output_tokens = pending_structured_output_tokens;
        scheduler_output.has_structured_output_spec_decode = structured_output_spec_decode;
    }

    fn make_cached_request_data(
        &self,
        running_reqs: &[usize],
        num_scheduled_tokens: &IndexMap<String, usize>,
        spec_decode_tokens: &IndexMap<String, Vec<i32>>,
        req_to_new_blocks: &FxHashMap<usize, KvCacheBlocks>,
        mut iteration_details: Option<&mut IterationDetails>,
    ) -> CachedRequestData {
        let mut req_ids = Vec::with_capacity(running_reqs.len());
        let mut new_token_ids = Vec::new();
        let mut new_block_ids = Vec::with_capacity(running_reqs.len());
        let mut num_computed_tokens = Vec::with_capacity(running_reqs.len());
        let mut num_output_tokens = Vec::with_capacity(running_reqs.len());

        for &h in running_reqs {
            let req = &self.requests[h];
            let req_id = &req.request_id;
            req_ids.push(req_id.clone());
            let is_prefill = req.is_prefill_chunk;
            // NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            // broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            // omit this payload.
            if self.use_pp && !self.async_scheduling {
                // When using PP, the scheduler sends the sampled tokens back,
                // because there's no direct communication between the first-
                // stage worker and the last-stage worker. Otherwise, we don't
                // need to send the sampled tokens back because the model runner
                // will cache them.
                let num_tokens = num_scheduled_tokens[req_id]
                    - spec_decode_tokens.get(req_id).map_or(0, Vec::len);
                let all = req.all_token_ids();
                let start = req.num_computed_tokens.min(all.len());
                let end = (req.num_computed_tokens + num_tokens).min(all.len());
                new_token_ids.push(all[start..end].to_vec());
            }
            new_block_ids.push(req_to_new_blocks[&h].get_block_ids(true, None));
            num_computed_tokens.push(req.num_computed_tokens);
            num_output_tokens.push(req.num_output_tokens() + req.num_output_placeholders);
            if let Some(details) = iteration_details.as_deref_mut()
                && !is_prefill
            {
                // Count decode requests here; all other scheduled reqs are prefills
                details.num_decode_requests += 1;
                details.num_decode_tokens += num_scheduled_tokens[req_id];
            }
        }

        CachedRequestData::new(
            req_ids,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
        )
    }

    /// `_allocate_encoder_inputs` for what `_try_schedule_encoder_inputs`
    /// scheduled.
    fn allocate_scheduled_encoder_inputs(&mut self, h: usize, schedule: &EncoderSchedule) {
        if schedule.encoder_inputs_to_schedule.is_empty()
            && schedule.external_load_encoder_input.is_empty()
        {
            return;
        }
        let request = &self.requests[h];
        for &input_id in schedule
            .encoder_inputs_to_schedule
            .iter()
            .chain(&schedule.external_load_encoder_input)
        {
            self.encoder_cache_manager.allocate(request, input_id);
            if let Some(ec_connector) = self.ec_connector.as_mut() {
                ec_connector.update_state_after_alloc(request, input_id);
            }
        }

        for &input_id in &schedule.duplicate_encoder_inputs {
            self.encoder_cache_manager.check_and_update_cache(request, input_id);
        }
    }

    /// Determine which encoder inputs need to be scheduled in the current step,
    /// and update `num_new_tokens` and encoder token budget accordingly.
    ///
    /// An encoder input will be scheduled if:
    /// - Its output tokens overlap with the range of tokens being computed
    ///   in this step, i.e.,
    ///   [num_computed_tokens, num_computed_tokens + num_new_tokens).
    /// - It is not already computed and stored in the encoder cache.
    /// - It is not exist on remote encoder cache (via ECConnector)
    /// - There is sufficient encoder token budget to process it.
    /// - The encoder cache has space to store it.
    ///
    /// If an encoder input cannot be scheduled due to cache or budget
    /// limitations, the method adjusts `num_new_tokens` to schedule only the
    /// decoder tokens up to just before the unschedulable encoder input.
    ///
    /// Note that num_computed_tokens includes both locally cached
    /// blocks and externally cached blocks (via KVConnector).
    fn try_schedule_encoder_inputs(
        &mut self,
        h: usize,
        num_computed_tokens: usize,
        mut num_new_tokens: i64,
        mut encoder_compute_budget: usize,
        shift_computed_tokens: usize,
    ) -> EncoderSchedule {
        let mut out = EncoderSchedule {
            encoder_inputs_to_schedule: Vec::new(),
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input: Vec::new(),
            duplicate_encoder_inputs: Vec::new(),
        };
        let request = &self.requests[h];
        if num_new_tokens == 0 || !request.has_encoder_inputs() {
            return out;
        }
        let rejected = |budget| EncoderSchedule {
            encoder_inputs_to_schedule: Vec::new(),
            num_new_tokens: 0,
            encoder_compute_budget: budget,
            external_load_encoder_input: Vec::new(),
            duplicate_encoder_inputs: Vec::new(),
        };

        // NOTE: since scheduler operates on the request level (possibly with
        // multiple encoder inputs per request), we need to create temporary
        // trackers for accounting at the encoder input level.
        let mut mm_hashes_to_schedule: FxHashMap<&str, usize> = FxHashMap::default();
        let mut num_embeds_to_schedule = 0;

        let computed = num_computed_tokens as i64;
        let encoder_window_end = computed + num_new_tokens + shift_computed_tokens as i64;
        let mm_features = &request.mm_features;
        let (mut lo, hi) = get_mm_features_in_window(
            mm_features,
            num_computed_tokens,
            encoder_window_end.max(0) as usize,
        );
        // For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder {
            lo = 0;
        }

        let mut mismatch = None;
        for (i, mm_feature) in mm_features.iter().enumerate().take(hi).skip(lo) {
            let start_pos = mm_feature.mm_position.offset as i64;
            let num_encoder_tokens = mm_feature.mm_position.length as i64;
            let num_encoder_embeds = mm_feature.mm_position.get_num_embeds();
            let item_identifier = mm_feature.identifier.as_str();

            if self.is_encoder_decoder && num_computed_tokens > 0 {
                assert_eq!(
                    start_pos, 0,
                    "Encoder input should be processed at the beginning of \
                     the sequence when encoder-decoder models are used."
                );
                // Encoder input has already been computed
                // The calculation here is a bit different. We don't turn encoder
                // output into tokens that get processed by the decoder and
                // reflected in num_computed_tokens. Instead, start_pos reflects
                // the position where we need to ensure we calculate encoder
                // inputs. This should always be 0 to ensure we calculate encoder
                // inputs before running the decoder.  Once we've calculated some
                // decoder tokens (num_computed_tokens > 0), then we know we
                // already calculated encoder inputs and can skip here.
                continue;
            }

            // If no encoder input chunking is allowed, we do not want to
            // partially schedule a multimodal item. If the scheduled range would
            // only cover part of the mm input, roll back to before the mm item.
            if self.disable_chunked_mm_input
                && !self.is_encoder_decoder
                && computed + num_new_tokens < start_pos + num_encoder_tokens
            {
                if computed < start_pos {
                    // Account for EAGLE shift when rolling back to avoid
                    // encoder cache miss. This ensures the scheduled range
                    // stops before start_pos even with the shift.
                    num_new_tokens = (start_pos - (computed + shift_computed_tokens as i64)).max(0);
                } else {
                    // The chunk starts on or inside the item (a prefix cache hit
                    // can end inside it), so there is nothing to roll back to.
                    // Schedule nothing until the rest of the item fits.
                    num_new_tokens = 0;
                }
                break;
            }

            if !self.is_encoder_decoder {
                // We are not using the encoder cache for encoder-decoder models,
                // yet.
                if let Some(&scheduled_num_encoder_embeds) =
                    mm_hashes_to_schedule.get(item_identifier)
                {
                    if scheduled_num_encoder_embeds != num_encoder_embeds {
                        mismatch = Some(encoder_compute_budget);
                        break;
                    }
                    // The same encoder input has already been scheduled in the
                    // current step.
                    out.duplicate_encoder_inputs.push(i);
                    continue;
                }

                if let Some(cached_num_encoder_embeds) =
                    self.encoder_cache_manager.get_cached_num_encoder_embeds(request, i)
                    && cached_num_encoder_embeds != num_encoder_embeds
                {
                    mismatch = Some(encoder_compute_budget);
                    break;
                }

                if self.encoder_cache_manager.check_and_update_cache(request, i) {
                    // The encoder input is already computed and cached from a
                    // previous step.
                    continue;
                }
            }

            if !self.encoder_cache_manager.can_allocate(
                request,
                i,
                encoder_compute_budget,
                num_embeds_to_schedule,
            ) {
                // The encoder cache is full or the encoder budget is exhausted.
                // NOTE(woosuk): We assume that the encoder input tokens should
                // be processed altogether, as the encoder usually uses
                // bidirectional attention.
                if computed < start_pos {
                    // We only schedule the decoder tokens just before the
                    // encoder input. No drafter shift here: its read-ahead into
                    // this unencoded input falls back to token embeddings.
                    num_new_tokens = num_new_tokens.min(start_pos - computed);
                } else {
                    // Because of prefix caching, num_computed_tokens is greater
                    // than start_pos even though its encoder input is not
                    // available. In this case, we can't schedule any token for
                    // the request in this step.
                    num_new_tokens = 0;
                }
                break;
            }

            // Calculate the number of embeddings to schedule in the current range
            // of scheduled encoder placeholder tokens.
            let start_idx_rel = (computed - start_pos).max(0) as usize;
            let end_idx_rel =
                num_encoder_tokens.min(encoder_window_end - start_pos).max(0) as usize;
            let (curr_embeds_start, curr_embeds_end) =
                mm_feature.mm_position.get_embeds_indices_in_range(start_idx_rel, end_idx_rel);
            // There's no embeddings in the current range of encoder placeholder tokens
            // so we can skip the encoder input.
            if curr_embeds_end == curr_embeds_start {
                continue;
            }

            if let Some(ec_connector) = self.ec_connector.as_mut()
                && ec_connector.has_cache_item(item_identifier)
            {
                mm_hashes_to_schedule.insert(item_identifier, num_encoder_embeds);
                out.external_load_encoder_input.push(i);
                num_embeds_to_schedule += num_encoder_embeds;
                continue;
            }

            num_embeds_to_schedule += num_encoder_embeds;
            encoder_compute_budget -= num_encoder_embeds;
            mm_hashes_to_schedule.insert(item_identifier, num_encoder_embeds);
            out.encoder_inputs_to_schedule.push(i);
        }

        if let Some(budget) = mismatch {
            // `_reject_on_encoder_cache_embed_mismatch`: the encoder cache entry
            // for the multimodal identifier holds a different number of
            // embeddings than this input expects; a client-provided multimodal
            // UUID may have been reused for different content.
            // TODO: Python logs a warning.
            let request_id = self.requests[h].request_id.clone();
            self.encoder_cache_mismatch_reqs.insert(request_id);
            return rejected(budget);
        }

        out.num_new_tokens = num_new_tokens;
        out.encoder_compute_budget = encoder_compute_budget;
        out
    }

    /// Move waiting requests with a cached prefix to the front of the window.
    ///
    /// Reordering is confined to the slots held by plain waiting requests, so a
    /// request in a blocked status or one already partially computed keeps its
    /// exact position and nothing is moved past it. That keeps this a
    /// reordering among peers rather than a change in priority, and leaves the
    /// queue beyond the window untouched.
    fn reorder_waiting_by_cached_prefix(&mut self) {
        if self.waiting.len() < 2 {
            return;
        }
        if self.kv_cache_manager.usage() < self.cache_aware_threshold {
            return;
        }

        let head: Vec<usize> =
            self.waiting.iter(&self.requests).take(self.cache_aware_window).collect();
        let slots: Vec<usize> = head
            .iter()
            .enumerate()
            .filter(|&(_, &h)| {
                let request = &self.requests[h];
                request.status == RequestStatus::Waiting && request.num_computed_tokens == 0
            })
            .map(|(i, _)| i)
            .collect();
        if slots.len() < 2 {
            return;
        }

        // Deepest cached prefix first, arrival order breaking ties.
        let cached: FxHashMap<usize, usize> = slots
            .iter()
            .map(|&i| {
                (
                    i,
                    self.kv_cache_manager.get_num_cached_tokens(&self.requests[head[i]]),
                )
            })
            .collect();
        let mut order = slots.clone();
        order.sort_by_key(|&i| (std::cmp::Reverse(cached[&i]), i));
        if order == slots {
            return;
        }

        let mut reordered = head.clone();
        for (&slot, &source) in slots.iter().zip(&order) {
            reordered[slot] = head[source];
        }
        // One pass to drop the whole window, then re-insert in the new order;
        // removing each request individually would be quadratic.
        self.waiting.remove_requests(head.iter().copied(), &self.requests);
        for &h in reordered.iter().rev() {
            self.waiting.prepend_request(h, &self.requests);
        }
    }

    /// self.waiting or self.async_kv_waiting, whichever head goes first.
    fn next_waiting_queue(&self, take_async_kv: bool) -> WaitingQueue {
        if !take_async_kv || self.async_kv_waiting.is_empty() {
            return WaitingQueue::Waiting;
        }
        let (Some(async_head), Some(head)) = (
            self.async_kv_waiting.peek_request(),
            self.waiting.peek_request(),
        ) else {
            return WaitingQueue::AsyncKv;
        };
        let (async_head, head) = (&self.requests[async_head], &self.requests[head]);
        let async_first = if self.policy == SchedulingPolicy::Priority {
            async_head.lt(head)
        } else {
            async_head.arrival_time < head.arrival_time
        };
        if async_first {
            WaitingQueue::AsyncKv
        } else {
            WaitingQueue::Waiting
        }
    }

    fn get_encoder_input_updates(
        &mut self,
        scheduled: &IndexMap<String, usize>,
        new_requests: &[usize],
        scheduled_encoder_inputs: &IndexMap<String, Vec<usize>>,
    ) -> IndexMap<String, IndexMap<usize, EncoderInputUpdate>> {
        // Installations (including resume/streaming) replace worker features.
        // SHM descriptors follow the processor cache's separate ACK protocol;
        // EngineCore cannot replay their resolved tensors.
        for &h in new_requests {
            let request = &self.requests[h];
            if request.mm_features.is_empty() {
                continue;
            }
            // The installed features are `strip_covered_mm_data(...)` of the
            // request's: covered payloads are dropped, unless M-RoPE keeps
            // their CPU-side metadata.
            let num_computed_tokens = request.num_computed_tokens;
            // TODO(rust-engine-core): a spec-defined order (see `py_int_set`).
            let mut pending = PyIntSet::default();
            for (i, feature) in request.mm_features.iter().enumerate() {
                let covered = num_computed_tokens > 0
                    && feature.mm_position.offset + feature.mm_position.length
                        <= num_computed_tokens;
                if feature.data == MultiModalData::Payload && (self.uses_mrope || !covered) {
                    pending.add(i);
                }
            }
            self.pending_encoder_input_ids.insert(request.request_id.clone(), pending);
        }

        let mut updates: IndexMap<String, IndexMap<usize, EncoderInputUpdate>> = IndexMap::new();
        for req_id in scheduled.keys() {
            let Some(pending) = self.pending_encoder_input_ids.get_mut(req_id) else {
                continue;
            };
            let h = self.requests.handle(req_id).expect("scheduled request exists");
            let request = &self.requests[h];
            let encoding: &[usize] =
                scheduled_encoder_inputs.get(req_id).map_or(&[], Vec::as_slice);
            if pending.is_empty() && encoding.is_empty() {
                continue;
            }
            for &input_id in encoding {
                let data = request.mm_features[input_id].data;
                if !pending.contains(input_id) && data == MultiModalData::Payload {
                    // A running request may rewind without being reinstalled.
                    updates
                        .entry(req_id.clone())
                        .or_default()
                        .insert(input_id, EncoderInputUpdate::Restore);
                    pending.add(input_id);
                }
            }

            // schedule() advances progress before execution. Exclude in-flight
            // prefill and speculative placeholders to obtain confirmed progress.
            let computed = request.num_computed_tokens as i64
                - request.num_output_placeholders.max(request.num_in_flight_tokens) as i64;
            let pending_ids: Vec<usize> = pending.iter().collect();
            for input_id in pending_ids {
                if encoding.contains(&input_id) {
                    continue;
                }
                let position = &request.mm_features[input_id].mm_position;
                if (position.offset + position.length) as i64 <= computed
                    || (self.is_encoder_decoder && computed > 0)
                {
                    updates
                        .entry(req_id.clone())
                        .or_default()
                        .insert(input_id, EncoderInputUpdate::Release);
                    pending.discard(input_id);
                }
            }
        }
        updates
    }
}
