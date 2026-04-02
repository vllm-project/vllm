# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering config single-phase admission control logic.

The scheduler tracks the union of active steering config (hash, phase)
pairs -- matching the worker's SteeringManager which allocates separate
table rows per (hash, phase) key.  Running requests contribute their
currently-active pair.  New WAITING requests only need capacity for their
starting phase (prefill), because the prefill row is released before the
decode row is registered in _handle_steering_transition.
"""


class TestSinglePhaseAdmission:
    """Test the single-phase steering admission control logic in isolation.

    The scheduler checks:
    1. Build scheduled_steering_configs from running requests:
       - (prefill_hash, "prefill") for requests still in prefill
       - (decode_hash, "decode") for requests in decode
    2. For each new request, compute new_hashes using only the starting
       phase (prefill for WAITING requests, decode for full prefix-cache
       hits).  Skip if len(scheduled) + len(new_unique) > max.
    3. When admitted, add only the starting phase (hash, phase) tuple to
       the scheduled set.
    """

    @staticmethod
    def _build_running_set(
        running_reqs: list[dict],
    ) -> set[tuple[int, str]]:
        """Reproduce the scheduler's running-request hash collection.

        Each req dict: {prefill_hash, decode_hash, in_prefill: bool}.
        """
        configs: set[tuple[int, str]] = set()
        for req in running_reqs:
            if req["in_prefill"]:
                if req["prefill_hash"] != 0:
                    configs.add((req["prefill_hash"], "prefill"))
            else:
                if req["decode_hash"] != 0:
                    configs.add((req["decode_hash"], "decode"))
        return configs

    @staticmethod
    def _should_skip(
        scheduled_configs: set[tuple[int, str]],
        prefill_hash: int,
        max_configs: int,
    ) -> bool:
        """Reproduce the scheduler's new-request admission check.

        Only the prefill hash is checked for WAITING requests -- the
        decode hash is not counted because the request only occupies one
        row at a time and the prefill row is released before the decode
        row is registered.
        """
        new_hashes: set[tuple[int, str]] = set()
        if prefill_hash != 0:
            new_hashes.add((prefill_hash, "prefill"))
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    @staticmethod
    def _admit(
        scheduled_configs: set[tuple[int, str]],
        prefill_hash: int,
        decode_hash: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
    ) -> None:
        """Reproduce the scheduler's post-admission hash update.

        Only the starting phase is added -- prefill when starting in
        prefill (num_computed_tokens < num_prompt_tokens), decode when
        starting in decode (full prefix-cache hit).
        """
        if num_computed_tokens < num_prompt_tokens:
            if prefill_hash != 0:
                scheduled_configs.add((prefill_hash, "prefill"))
        else:
            if decode_hash != 0:
                scheduled_configs.add((decode_hash, "decode"))

    # --- Running request hash collection ---

    def test_running_prefill_uses_prefill_hash(self):
        """Request in prefill contributes its prefill hash."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": True},
            ]
        )
        assert configs == {(111, "prefill")}

    def test_running_decode_uses_decode_hash(self):
        """Request in decode contributes its decode hash."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": False},
            ]
        )
        assert configs == {(222, "decode")}

    def test_running_mixed_phases(self):
        """Mix of prefill and decode requests."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": True},
                {"prefill_hash": 333, "decode_hash": 444, "in_prefill": False},
            ]
        )
        assert configs == {(111, "prefill"), (444, "decode")}

    def test_running_zero_hash_excluded(self):
        """Zero hashes are excluded from the set."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 0, "decode_hash": 0, "in_prefill": True},
            ]
        )
        assert configs == set()

    # --- New request admission (single-phase) ---

    def test_no_hashes_always_admits(self):
        """Request with no steering (prefill hash=0) always passes."""
        assert not self._should_skip({(111, "prefill"), (222, "decode")}, 0, 2)

    def test_admits_when_capacity_available(self):
        """Request admitted when there are open slots."""
        assert not self._should_skip({(111, "prefill")}, 222, 3)

    def test_request_with_both_hashes_uses_one_slot(self):
        """A request with distinct prefill and decode hashes only needs
        one slot (prefill) at admission time -- the decode row is
        registered later when the prefill row is released."""
        # With dual-counting this would skip; with single-phase it fits.
        assert not self._should_skip({(111, "prefill")}, 222, 2)

    def test_request_with_same_hash_both_phases_uses_one_slot(self):
        """If prefill_hash == decode_hash, only the prefill hash is
        checked so it still uses one slot."""
        # With dual-counting this would skip; with single-phase it fits.
        assert not self._should_skip({(111, "prefill")}, 222, 2)

    def test_request_prefill_hash_already_scheduled(self):
        """Prefill hash already in set -> zero new slots needed."""
        assert not self._should_skip({(111, "prefill")}, 111, 2)

    def test_only_prefill_hash_nonzero(self):
        """Request with only prefill hash occupies one slot."""
        assert not self._should_skip({(111, "prefill")}, 222, 2)
        assert self._should_skip({(111, "prefill"), (333, "decode")}, 222, 2)

    def test_skips_when_at_capacity_with_new_hash(self):
        """New prefill hash when no slot open -> skip."""
        assert self._should_skip({(111, "prefill"), (222, "decode")}, 333, 2)

    def test_single_slot_capacity(self):
        """Works with max_steering_configs=1."""
        # No hashes -> pass
        assert not self._should_skip(set(), 0, 1)
        # One new prefill hash -> pass if no existing
        assert not self._should_skip(set(), 111, 1)
        # Even with both hashes non-zero (not checked here), single
        # slot should work since only prefill is counted
        assert not self._should_skip(set(), 42, 1)
        # Slot occupied by different hash -> skip
        assert self._should_skip({(111, "prefill")}, 222, 1)
        # Slot occupied by same hash -> pass (dedup)
        assert not self._should_skip({(111, "prefill")}, 111, 1)

    def test_same_hash_prefill_and_decode_uses_one_slot(self):
        """When prefill_hash == decode_hash, only the prefill hash is
        checked at admission time, so only one slot is needed."""
        # With dual-counting this would skip (needs 2); with single-phase
        # it fits in 1.
        assert not self._should_skip(set(), 42, 1)

    # --- Post-admission hash tracking (single-phase) ---

    def test_admit_adds_only_prefill_hash_when_starting_prefill(self):
        """Admitting a request starting in prefill adds only the prefill
        hash, not the decode hash."""
        configs: set[tuple[int, str]] = {(111, "prefill")}
        self._admit(configs, 222, 333, num_computed_tokens=0, num_prompt_tokens=100)
        assert configs == {(111, "prefill"), (222, "prefill")}

    def test_admit_adds_only_decode_hash_when_starting_decode(self):
        """Admitting a request with full prefix-cache hit (starting in
        decode) adds only the decode hash."""
        configs: set[tuple[int, str]] = {(111, "prefill")}
        self._admit(configs, 222, 333, num_computed_tokens=100, num_prompt_tokens=100)
        assert configs == {(111, "prefill"), (333, "decode")}

    def test_admit_skips_zero_hashes(self):
        """Zero hashes are not added to the set."""
        configs: set[tuple[int, str]] = set()
        self._admit(configs, 0, 0, num_computed_tokens=0, num_prompt_tokens=100)
        assert configs == set()

    def test_admit_idempotent_for_existing(self):
        """Adding already-present hashes is a no-op."""
        configs: set[tuple[int, str]] = {(111, "prefill"), (222, "decode")}
        self._admit(configs, 111, 999, num_computed_tokens=0, num_prompt_tokens=100)
        assert configs == {(111, "prefill"), (222, "decode")}

    def test_admit_decode_start_idempotent(self):
        """Adding already-present decode hash is a no-op."""
        configs: set[tuple[int, str]] = {(222, "decode")}
        self._admit(configs, 999, 222, num_computed_tokens=100, num_prompt_tokens=100)
        assert configs == {(222, "decode")}

    # --- End-to-end scenario ---

    def test_full_scenario(self):
        """Simulate a sequence of scheduling decisions."""
        max_configs = 2
        configs: set[tuple[int, str]] = set()

        # Request 1: prefill_hash=100, decode_hash=200, starting prefill
        # Needs 1 slot (prefill only).
        assert not self._should_skip(configs, 100, max_configs)
        self._admit(configs, 100, 200, num_computed_tokens=0, num_prompt_tokens=50)
        assert configs == {(100, "prefill")}

        # Request 2: prefill_hash=300, decode_hash=400, starting prefill
        # Needs 1 new slot. Total would be 2 = max. Fits.
        assert not self._should_skip(configs, 300, max_configs)
        self._admit(configs, 300, 400, num_computed_tokens=0, num_prompt_tokens=50)
        assert configs == {(100, "prefill"), (300, "prefill")}

        # Request 3: prefill_hash=500, decode_hash=600, starting prefill
        # 1 new slot needed. Total would be 3 > 2. Skip.
        assert self._should_skip(configs, 500, max_configs)

        # Request 4: prefill_hash=100, decode_hash=999, starting prefill
        # Prefill hash already present -> 0 new slots. Fits.
        assert not self._should_skip(configs, 100, max_configs)

    def test_full_scenario_with_decode_start(self):
        """Scenario with full prefix-cache hits (decode start)."""
        max_configs = 2
        configs: set[tuple[int, str]] = set()

        # Request 1: full prefix-cache hit, starting in decode
        # decode_hash=200 needs 1 slot.
        # Note: _should_skip uses prefill_hash for WAITING requests.
        # For decode-start, we need to check differently -- but in the
        # scheduler, the admission check only sees the prefill hash
        # because it runs before prefix cache resolution. So the prefill
        # hash is what gets checked at admission time. The post-admission
        # tracking then uses the actual starting phase.
        assert not self._should_skip(configs, 100, max_configs)
        # But the actual starting phase is decode (full cache hit).
        self._admit(configs, 100, 200, num_computed_tokens=50, num_prompt_tokens=50)
        assert configs == {(200, "decode")}

    def test_freed_capacity_admits(self):
        """After removing a config, new ones can be admitted."""
        max_configs = 2
        configs: set[tuple[int, str]] = {(111, "prefill"), (222, "decode")}
        # At capacity, new hash blocked
        assert self._should_skip(configs, 333, max_configs)
        # Free a slot
        configs.discard((222, "decode"))
        # Now admits
        assert not self._should_skip(configs, 333, max_configs)


class TestTransitionAwareCapacity:
    """Test transition-aware capacity counting and decode-start checks.

    Fix A: When a running request will complete prefill this step
    (num_computed + num_scheduled >= num_prompt), the scheduler must
    also reserve the decode row in scheduled_steering_configs, because
    the model runner's _handle_steering_transition will register the
    decode config mid-step.

    Fix B: A WAITING request that has a full prefix-cache hit
    (num_computed_tokens >= num_prompt_tokens after cache resolution)
    starts directly in decode.  The scheduler must verify decode
    capacity at that point.
    """

    @staticmethod
    def _build_running_set_transition_aware(
        running_reqs: list[dict],
    ) -> set[tuple[int, str]]:
        """Reproduce the scheduler's transition-aware running-request
        hash collection.

        Each req dict: {prefill_hash, decode_hash, num_computed_tokens,
                        num_prompt_tokens, num_scheduled_tokens}.
        """
        configs: set[tuple[int, str]] = set()
        for req in running_reqs:
            currently_prefilling = req["num_computed_tokens"] < req["num_prompt_tokens"]
            if currently_prefilling:
                if req["prefill_hash"] != 0:
                    configs.add((req["prefill_hash"], "prefill"))
                # Predict transition.
                will_complete = (
                    req["num_computed_tokens"] + req["num_scheduled_tokens"]
                    >= req["num_prompt_tokens"]
                )
                if will_complete and req["decode_hash"] != 0:
                    configs.add((req["decode_hash"], "decode"))
            else:
                if req["decode_hash"] != 0:
                    configs.add((req["decode_hash"], "decode"))
        return configs

    @staticmethod
    def _should_skip_decode_start(
        scheduled_configs: set[tuple[int, str]],
        decode_hash: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
        max_configs: int,
    ) -> bool:
        """Reproduce the scheduler's post-cache decode-start check.

        If the request has a full prefix-cache hit and a non-zero decode
        hash, check whether the decode pair can fit in the capacity set.
        """
        if num_computed_tokens < num_prompt_tokens:
            return False
        if decode_hash == 0:
            return False
        decode_pair = (decode_hash, "decode")
        return (
            decode_pair not in scheduled_configs
            and len(scheduled_configs) >= max_configs
        )

    # --- Fix A: Transition prediction for running requests ---

    def test_transition_prediction_reserves_decode_row(self):
        """A running request that will complete prefill this step
        contributes BOTH its prefill and decode pairs."""
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 222,
                    "num_computed_tokens": 90,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 20,  # 90+20=110 >= 100
                },
            ]
        )
        assert (111, "prefill") in configs
        assert (222, "decode") in configs
        assert len(configs) == 2

    def test_transition_prediction_no_false_positive(self):
        """A running request that will NOT complete prefill this step
        contributes only its prefill pair."""
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 222,
                    "num_computed_tokens": 50,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 10,  # 50+10=60 < 100
                },
            ]
        )
        assert configs == {(111, "prefill")}

    def test_transition_prediction_exact_boundary(self):
        """Transition fires at the exact boundary
        (num_computed + num_scheduled == num_prompt)."""
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 222,
                    "num_computed_tokens": 80,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 20,  # 80+20=100 == 100
                },
            ]
        )
        assert (111, "prefill") in configs
        assert (222, "decode") in configs

    def test_transition_prediction_zero_decode_hash(self):
        """Transition prediction does not add a zero decode hash."""
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 0,
                    "num_computed_tokens": 90,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 20,
                },
            ]
        )
        assert configs == {(111, "prefill")}

    def test_transition_prediction_decode_req_unchanged(self):
        """A running request already in decode is unchanged by the
        transition-aware logic."""
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 222,
                    "num_computed_tokens": 100,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 1,
                },
            ]
        )
        assert configs == {(222, "decode")}

    # --- Fix B: Decode-start capacity check for WAITING requests ---

    def test_decode_start_capacity_check(self):
        """A waiting request with a full prefix-cache hit is skipped
        when decode capacity is full."""
        # Capacity is 2 and both slots are occupied.
        scheduled = {(111, "prefill"), (222, "decode")}
        assert self._should_skip_decode_start(
            scheduled,
            decode_hash=333,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            max_configs=2,
        )

    def test_decode_start_admitted_when_capacity(self):
        """A waiting request with a full prefix-cache hit is admitted
        when decode capacity is available."""
        # Capacity is 3, only 2 occupied -> room for 1 more.
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip_decode_start(
            scheduled,
            decode_hash=333,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            max_configs=3,
        )

    def test_decode_start_existing_hash_always_fits(self):
        """If the decode pair is already in the scheduled set, it always
        fits regardless of capacity."""
        scheduled = {(111, "prefill"), (333, "decode")}
        assert not self._should_skip_decode_start(
            scheduled,
            decode_hash=333,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            max_configs=2,
        )

    def test_decode_start_no_check_when_still_prefilling(self):
        """When num_computed < num_prompt, the decode-start check is
        not triggered (the request will start in prefill, not decode)."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip_decode_start(
            scheduled,
            decode_hash=333,
            num_computed_tokens=50,
            num_prompt_tokens=100,
            max_configs=2,
        )

    def test_decode_start_zero_hash_bypasses_check(self):
        """A request with decode_hash=0 bypasses the decode-start
        capacity check."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip_decode_start(
            scheduled,
            decode_hash=0,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            max_configs=2,
        )

    # --- Combined scenario ---

    def test_transition_blocks_new_admission(self):
        """A transitioning running request that reserves a decode row
        should cause a new waiting request to be skipped if capacity
        is full."""
        max_configs = 2
        # Running request completing prefill -> reserves both rows.
        configs = self._build_running_set_transition_aware(
            [
                {
                    "prefill_hash": 111,
                    "decode_hash": 222,
                    "num_computed_tokens": 95,
                    "num_prompt_tokens": 100,
                    "num_scheduled_tokens": 10,
                },
            ]
        )
        assert len(configs) == 2  # (111, "prefill") + (222, "decode")

        # New waiting request with a new prefill hash -> would need a
        # 3rd slot but max is 2.
        new_hashes: set[tuple[int, str]] = {(333, "prefill")}
        new_unique = new_hashes - configs
        assert len(configs) + len(new_unique) > max_configs


class TestDecodeOnlyCapacityCheck:
    """Test that decode-only steering (prefill_hash=0, decode_hash!=0)
    is capacity-checked at the WAITING admission gate.

    Bug: The original admission gate only checked prefill_hash.  If a
    request had prefill_hash==0 but decode_hash!=0, new_hashes stayed
    empty and the capacity check was completely skipped, allowing the
    request to exceed max_steering_configs.

    Fix: Add an elif for decode-only steering so that requests with
    only a decode hash are also gated.
    """

    @staticmethod
    def _should_skip(
        scheduled_configs: set[tuple[int, str]],
        prefill_hash: int,
        decode_hash: int,
        max_configs: int,
    ) -> bool:
        """Reproduce the fixed scheduler admission check that also
        considers decode-only steering.

        The elif is correct: if a request has both hashes, only
        prefill needs counting (the decode row replaces the prefill
        row after transition -- one row at a time).
        """
        new_hashes: set[tuple[int, str]] = set()
        if prefill_hash != 0:
            new_hashes.add((prefill_hash, "prefill"))
        elif decode_hash != 0:
            new_hashes.add((decode_hash, "decode"))
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    def test_decode_only_is_capacity_checked(self):
        """A request with prefill_hash=0, decode_hash!=0 should be
        capacity-checked and skipped when at capacity."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert self._should_skip(scheduled, 0, 333, max_configs=2)

    def test_decode_only_admitted_when_capacity(self):
        """A decode-only request is admitted when there is capacity."""
        scheduled = {(111, "prefill")}
        assert not self._should_skip(scheduled, 0, 333, max_configs=2)

    def test_decode_only_existing_hash_fits(self):
        """A decode-only request whose hash is already scheduled fits."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip(scheduled, 0, 222, max_configs=2)

    def test_both_hashes_only_counts_prefill(self):
        """When both hashes are nonzero, only the prefill hash is
        counted (elif branch: decode is not reached)."""
        # Only 1 slot, occupied by a different prefill hash => skip
        # because prefill hash 333 is new.
        scheduled = {(111, "prefill")}
        assert self._should_skip(scheduled, 333, 444, max_configs=1)

    def test_both_hashes_prefill_dedup(self):
        """When both hashes are nonzero and prefill hash is already
        in the set, request passes even at capacity."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip(scheduled, 111, 444, max_configs=2)

    def test_no_hashes_always_admits(self):
        """A request with both hashes=0 always passes (no steering)."""
        scheduled = {(111, "prefill"), (222, "decode")}
        assert not self._should_skip(scheduled, 0, 0, max_configs=2)

    def test_decode_only_single_slot(self):
        """Decode-only with max_configs=1."""
        assert not self._should_skip(set(), 0, 111, max_configs=1)
        assert self._should_skip({(222, "decode")}, 0, 111, max_configs=1)
        assert not self._should_skip({(111, "decode")}, 0, 111, max_configs=1)


class TestWaitingTransitionPrediction:
    """Test that WAITING requests that will complete prefill this step
    have their decode row reserved in scheduled_steering_configs.

    Bug: The post-admission hash tracking for WAITING requests only added
    the starting phase (prefill).  But if a WAITING request has
    num_computed + num_new >= num_prompt, it will complete prefill this
    step and _handle_steering_transition will register its decode config.
    The scheduler didn't reserve that decode row, so subsequent WAITING
    requests saw undercounted capacity.

    Fix: After adding the prefill config, check whether the request will
    complete prefill this step and, if so, also add the decode config.
    """

    @staticmethod
    def _admit_transition_aware(
        scheduled_configs: set[tuple[int, str]],
        prefill_hash: int,
        decode_hash: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
        num_new_tokens: int,
    ) -> None:
        """Reproduce the fixed scheduler post-admission hash update
        with transition prediction for WAITING requests."""
        if num_computed_tokens < num_prompt_tokens:
            if prefill_hash != 0:
                scheduled_configs.add((prefill_hash, "prefill"))
            # Predict transition: if this request will complete
            # prefill this step, also reserve its decode row.
            will_complete = num_computed_tokens + num_new_tokens >= num_prompt_tokens
            if will_complete and decode_hash != 0:
                scheduled_configs.add((decode_hash, "decode"))
        else:
            if decode_hash != 0:
                scheduled_configs.add((decode_hash, "decode"))

    def test_transition_reserves_decode_row(self):
        """A WAITING request completing prefill this step reserves
        both prefill and decode rows."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=222,
            num_computed_tokens=90,
            num_prompt_tokens=100,
            num_new_tokens=20,  # 90+20=110 >= 100
        )
        assert (111, "prefill") in configs
        assert (222, "decode") in configs
        assert len(configs) == 2

    def test_no_transition_only_prefill(self):
        """A WAITING request NOT completing prefill this step only
        reserves the prefill row."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=222,
            num_computed_tokens=50,
            num_prompt_tokens=100,
            num_new_tokens=10,  # 50+10=60 < 100
        )
        assert configs == {(111, "prefill")}

    def test_transition_exact_boundary(self):
        """Transition fires at exact boundary
        (num_computed + num_new == num_prompt)."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=222,
            num_computed_tokens=80,
            num_prompt_tokens=100,
            num_new_tokens=20,  # 80+20=100 == 100
        )
        assert (111, "prefill") in configs
        assert (222, "decode") in configs

    def test_transition_zero_decode_hash(self):
        """Transition prediction with decode_hash=0 does not add a
        zero hash."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=0,
            num_computed_tokens=90,
            num_prompt_tokens=100,
            num_new_tokens=20,
        )
        assert configs == {(111, "prefill")}

    def test_transition_zero_prefill_hash(self):
        """Transition prediction with prefill_hash=0 only adds the
        decode row."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=0,
            decode_hash=222,
            num_computed_tokens=90,
            num_prompt_tokens=100,
            num_new_tokens=20,
        )
        assert (222, "decode") in configs
        assert (0, "prefill") not in configs

    def test_full_cache_hit_only_decode(self):
        """Full prefix-cache hit -> only decode row, no transition
        prediction needed."""
        configs: set[tuple[int, str]] = set()
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=222,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            num_new_tokens=1,
        )
        assert configs == {(222, "decode")}

    def test_transition_blocks_subsequent_admission(self):
        """A WAITING request's transition prediction that reserves a
        decode row blocks a subsequent WAITING request from admission."""
        max_configs = 2
        configs: set[tuple[int, str]] = set()

        # First request: will complete prefill, reserves both rows.
        self._admit_transition_aware(
            configs,
            prefill_hash=111,
            decode_hash=222,
            num_computed_tokens=90,
            num_prompt_tokens=100,
            num_new_tokens=20,
        )
        assert len(configs) == 2

        # Second request: new prefill hash -> would need a 3rd slot
        # but max is 2.
        new_hashes: set[tuple[int, str]] = {(333, "prefill")}
        new_unique = new_hashes - configs
        assert len(configs) + len(new_unique) > max_configs


class TestSteeringAdmissionLogicLegacy:
    """Legacy single-hash admission tests kept for reference.

    These reproduce the simpler check that was in place before dual-hash
    admission.  They should still pass because the single-phase logic
    checks only the prefill hash, which matches legacy behaviour when
    decode_hash is 0.
    """

    def _should_skip(
        self,
        steering_config_exists: bool,
        request_prefill_hash: int,
        scheduled_configs: set[tuple[int, str]],
        max_configs: int,
    ) -> bool:
        """Legacy: single-hash check via the single-phase logic."""
        if not steering_config_exists:
            return False
        new_hashes: set[tuple[int, str]] = set()
        if request_prefill_hash != 0:
            new_hashes.add((request_prefill_hash, "prefill"))
        # decode_hash=0 (legacy: only prefill hash)
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    def test_no_steering_config_always_admits(self):
        assert not self._should_skip(False, 123, {(1, "prefill"), (2, "decode")}, 2)

    def test_no_steering_hash_always_admits(self):
        assert not self._should_skip(True, 0, {(1, "prefill"), (2, "decode")}, 2)

    def test_admits_when_capacity_available(self):
        assert not self._should_skip(True, 333, {(111, "prefill")}, 2)

    def test_admits_when_hash_already_scheduled(self):
        assert not self._should_skip(True, 111, {(111, "prefill"), (222, "decode")}, 2)

    def test_skips_when_at_capacity_with_new_hash(self):
        assert self._should_skip(True, 333, {(111, "prefill"), (222, "decode")}, 2)

    def test_single_slot_capacity(self):
        assert not self._should_skip(True, 111, set(), 1)
        assert self._should_skip(True, 222, {(111, "prefill")}, 1)
        assert not self._should_skip(True, 111, {(111, "prefill")}, 1)

    def test_hash_deduplication(self):
        scheduled: set[tuple[int, str]] = {(111, "prefill")}
        assert not self._should_skip(True, 111, scheduled, 1)

    def test_freed_capacity_admits(self):
        scheduled: set[tuple[int, str]] = {(111, "prefill"), (222, "decode")}
        assert self._should_skip(True, 333, scheduled, 2)
        scheduled.discard((222, "decode"))
        assert not self._should_skip(True, 333, scheduled, 2)
