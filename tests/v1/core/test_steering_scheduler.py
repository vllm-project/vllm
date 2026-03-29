# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering config phase-keyed admission control logic.

The scheduler tracks the union of active steering config (hash, phase)
pairs — matching the worker's SteeringManager which allocates separate
table rows per (hash, phase) key.  Running requests contribute their
currently-active pair.  New requests being admitted must have capacity
for BOTH their prefill and decode (hash, phase) pairs (since the
request will use both over its lifetime).
"""


class TestDualHashAdmission:
    """Test the phase-keyed steering admission control logic in isolation.

    The scheduler now checks:
    1. Build scheduled_steering_configs from running requests:
       - (prefill_hash, "prefill") for requests still in prefill
       - (decode_hash, "decode") for requests in decode
    2. For each new request, compute new_hashes as (hash, phase) tuples
       (excluding 0), then new_unique = new_hashes - scheduled.
       Skip if len(scheduled) + len(new_unique) > max.
    3. When admitted, add BOTH (hash, phase) tuples to the scheduled set.
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
        decode_hash: int,
        max_configs: int,
    ) -> bool:
        """Reproduce the scheduler's new-request admission check."""
        new_hashes: set[tuple[int, str]] = set()
        if prefill_hash != 0:
            new_hashes.add((prefill_hash, "prefill"))
        if decode_hash != 0:
            new_hashes.add((decode_hash, "decode"))
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    @staticmethod
    def _admit(
        scheduled_configs: set[tuple[int, str]],
        prefill_hash: int,
        decode_hash: int,
    ) -> None:
        """Reproduce the scheduler's post-admission hash update."""
        if prefill_hash != 0:
            scheduled_configs.add((prefill_hash, "prefill"))
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

    # --- New request admission ---

    def test_no_hashes_always_admits(self):
        """Request with no steering (both hashes=0) always passes."""
        assert not self._should_skip(
            {(111, "prefill"), (222, "decode")}, 0, 0, 2
        )

    def test_admits_when_capacity_available(self):
        """Request admitted when there are open slots."""
        assert not self._should_skip(
            {(111, "prefill")}, 222, 333, 3
        )

    def test_request_with_both_hashes_uses_two_slots(self):
        """A request with distinct prefill and decode hashes needs two
        new slots if neither is already scheduled."""
        assert self._should_skip(
            {(111, "prefill")}, 222, 333, 2
        )

    def test_request_with_shared_hash_still_uses_two_slots(self):
        """If prefill_hash == decode_hash, they still occupy two slots
        because the worker allocates separate (hash, phase) rows."""
        assert self._should_skip(
            {(111, "prefill")}, 222, 222, 2
        )

    def test_request_prefill_hash_already_scheduled(self):
        """Prefill hash already in set reduces new slots needed."""
        assert not self._should_skip(
            {(111, "prefill")}, 111, 222, 2
        )

    def test_request_decode_hash_already_scheduled(self):
        """Decode hash already in set reduces new slots needed."""
        assert not self._should_skip(
            {(222, "decode")}, 111, 222, 2
        )

    def test_request_both_hashes_already_scheduled(self):
        """Both hashes already in set -> zero new slots."""
        assert not self._should_skip(
            {(111, "prefill"), (222, "decode")}, 111, 222, 2
        )

    def test_only_prefill_hash_nonzero(self):
        """Request with only prefill hash occupies one slot."""
        assert not self._should_skip(
            {(111, "prefill")}, 222, 0, 2
        )
        assert self._should_skip(
            {(111, "prefill"), (333, "decode")}, 222, 0, 2
        )

    def test_only_decode_hash_nonzero(self):
        """Request with only decode hash occupies one slot."""
        assert not self._should_skip(
            {(111, "prefill")}, 0, 222, 2
        )
        assert self._should_skip(
            {(111, "prefill"), (333, "decode")}, 0, 222, 2
        )

    def test_skips_when_at_capacity_with_two_new_hashes(self):
        """Two new hashes when only one slot open -> skip."""
        assert self._should_skip(
            {(111, "prefill")}, 222, 333, 2
        )

    def test_single_slot_capacity(self):
        """Works with max_steering_configs=1."""
        # No hashes -> pass
        assert not self._should_skip(set(), 0, 0, 1)
        # One new hash (prefill only) -> pass if no existing
        assert not self._should_skip(set(), 111, 0, 1)
        # Two distinct new hashes -> skip (needs 2, only 1 available)
        assert self._should_skip(set(), 111, 222, 1)
        # One hash already present + matching -> pass
        assert not self._should_skip({(111, "prefill")}, 111, 0, 1)

    def test_same_hash_prefill_and_decode_counts_two_slots(self):
        """When prefill_hash == decode_hash, the scheduler must count
        two slots because the worker allocates separate (hash, phase)
        rows."""
        # Same hash for both phases: needs 2 rows, only 1 available
        assert self._should_skip(set(), 42, 42, 1)
        # Same hash for both phases: fits in 2
        assert not self._should_skip(set(), 42, 42, 2)

    # --- Post-admission hash tracking ---

    def test_admit_adds_both_hashes(self):
        """Admitting a request adds both prefill and decode hashes."""
        configs: set[tuple[int, str]] = {(111, "prefill")}
        self._admit(configs, 222, 333)
        assert configs == {(111, "prefill"), (222, "prefill"), (333, "decode")}

    def test_admit_skips_zero_hashes(self):
        """Zero hashes are not added to the set."""
        configs: set[tuple[int, str]] = set()
        self._admit(configs, 111, 0)
        assert configs == {(111, "prefill")}

    def test_admit_idempotent_for_existing(self):
        """Adding already-present hashes is a no-op."""
        configs: set[tuple[int, str]] = {(111, "prefill"), (222, "decode")}
        self._admit(configs, 111, 222)
        assert configs == {(111, "prefill"), (222, "decode")}

    # --- End-to-end scenario ---

    def test_full_scenario(self):
        """Simulate a sequence of scheduling decisions."""
        max_configs = 4
        configs: set[tuple[int, str]] = set()

        # Request 1: prefill_hash=100, decode_hash=200 -> both new (2 slots)
        assert not self._should_skip(configs, 100, 200, max_configs)
        self._admit(configs, 100, 200)
        assert configs == {(100, "prefill"), (200, "decode")}

        # Request 2: prefill_hash=100, decode_hash=300
        #   -> (100, "prefill") already in set, only (300, "decode") is new
        assert not self._should_skip(configs, 100, 300, max_configs)
        self._admit(configs, 100, 300)
        assert configs == {(100, "prefill"), (200, "decode"), (300, "decode")}

        # Request 3: prefill_hash=400, decode_hash=500 -> 2 new, at cap (5>4)
        assert self._should_skip(configs, 400, 500, max_configs)

        # Request 4: prefill_hash=100, decode_hash=200 -> 0 new
        assert not self._should_skip(configs, 100, 200, max_configs)

    def test_freed_capacity_admits(self):
        """After removing a config, new ones can be admitted."""
        max_configs = 2
        configs: set[tuple[int, str]] = {(111, "prefill"), (222, "decode")}
        # At capacity, new hash blocked
        assert self._should_skip(configs, 333, 0, max_configs)
        # Free a slot
        configs.discard((222, "decode"))
        # Now admits
        assert not self._should_skip(configs, 333, 0, max_configs)


class TestSteeringAdmissionLogicLegacy:
    """Legacy single-hash admission tests kept for reference.

    These reproduce the simpler check that was in place before dual-hash
    admission.  They should still pass because the phase-keyed logic
    degenerates to single-hash behaviour when decode_hash is 0
    (only (prefill_hash, "prefill") is added).
    """

    def _should_skip(
        self,
        steering_config_exists: bool,
        request_prefill_hash: int,
        scheduled_configs: set[tuple[int, str]],
        max_configs: int,
    ) -> bool:
        """Legacy: single-hash check via the new phase-keyed logic."""
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
        assert not self._should_skip(
            False, 123, {(1, "prefill"), (2, "decode")}, 2
        )

    def test_no_steering_hash_always_admits(self):
        assert not self._should_skip(
            True, 0, {(1, "prefill"), (2, "decode")}, 2
        )

    def test_admits_when_capacity_available(self):
        assert not self._should_skip(True, 333, {(111, "prefill")}, 2)

    def test_admits_when_hash_already_scheduled(self):
        assert not self._should_skip(
            True, 111, {(111, "prefill"), (222, "decode")}, 2
        )

    def test_skips_when_at_capacity_with_new_hash(self):
        assert self._should_skip(
            True, 333, {(111, "prefill"), (222, "decode")}, 2
        )

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
