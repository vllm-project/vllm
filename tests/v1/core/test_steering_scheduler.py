# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering config dual-hash admission control logic.

The scheduler tracks the union of active steering config hashes.
Running requests contribute their currently-active hash (prefill or
decode depending on phase).  New requests being admitted must have
capacity for BOTH their prefill and decode hashes (since the request
will use both over its lifetime).
"""


class TestDualHashAdmission:
    """Test the dual-hash steering admission control logic in isolation.

    The scheduler now checks:
    1. Build scheduled_steering_configs from running requests:
       - prefill hash for requests still in prefill
       - decode hash for requests in decode
    2. For each new request, compute new_hashes = {prefill_hash, decode_hash}
       (excluding 0), then new_unique = new_hashes - scheduled.
       Skip if len(scheduled) + len(new_unique) > max.
    3. When admitted, add BOTH hashes to the scheduled set.
    """

    @staticmethod
    def _build_running_set(
        running_reqs: list[dict],
    ) -> set[int]:
        """Reproduce the scheduler's running-request hash collection.

        Each req dict: {prefill_hash, decode_hash, in_prefill: bool}.
        """
        configs: set[int] = set()
        for req in running_reqs:
            if req["in_prefill"]:
                if req["prefill_hash"] != 0:
                    configs.add(req["prefill_hash"])
            else:
                if req["decode_hash"] != 0:
                    configs.add(req["decode_hash"])
        return configs

    @staticmethod
    def _should_skip(
        scheduled_configs: set[int],
        prefill_hash: int,
        decode_hash: int,
        max_configs: int,
    ) -> bool:
        """Reproduce the scheduler's new-request admission check."""
        new_hashes: set[int] = set()
        if prefill_hash != 0:
            new_hashes.add(prefill_hash)
        if decode_hash != 0:
            new_hashes.add(decode_hash)
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    @staticmethod
    def _admit(
        scheduled_configs: set[int],
        prefill_hash: int,
        decode_hash: int,
    ) -> None:
        """Reproduce the scheduler's post-admission hash update."""
        if prefill_hash != 0:
            scheduled_configs.add(prefill_hash)
        if decode_hash != 0:
            scheduled_configs.add(decode_hash)

    # --- Running request hash collection ---

    def test_running_prefill_uses_prefill_hash(self):
        """Request in prefill contributes its prefill hash."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": True},
            ]
        )
        assert configs == {111}

    def test_running_decode_uses_decode_hash(self):
        """Request in decode contributes its decode hash."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": False},
            ]
        )
        assert configs == {222}

    def test_running_mixed_phases(self):
        """Mix of prefill and decode requests."""
        configs = self._build_running_set(
            [
                {"prefill_hash": 111, "decode_hash": 222, "in_prefill": True},
                {"prefill_hash": 333, "decode_hash": 444, "in_prefill": False},
            ]
        )
        assert configs == {111, 444}

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
        assert not self._should_skip({111, 222}, 0, 0, 2)

    def test_admits_when_capacity_available(self):
        """Request admitted when there are open slots."""
        assert not self._should_skip({111}, 222, 333, 3)

    def test_request_with_both_hashes_uses_two_slots(self):
        """A request with distinct prefill and decode hashes needs two
        new slots if neither is already scheduled."""
        assert self._should_skip({111}, 222, 333, 2)

    def test_request_with_shared_hash_uses_one_slot(self):
        """If prefill_hash == decode_hash, only one new slot is needed."""
        assert not self._should_skip({111}, 222, 222, 2)

    def test_request_prefill_hash_already_scheduled(self):
        """Prefill hash already in set reduces new slots needed."""
        assert not self._should_skip({111}, 111, 222, 2)

    def test_request_decode_hash_already_scheduled(self):
        """Decode hash already in set reduces new slots needed."""
        assert not self._should_skip({222}, 111, 222, 2)

    def test_request_both_hashes_already_scheduled(self):
        """Both hashes already in set -> zero new slots."""
        assert not self._should_skip({111, 222}, 111, 222, 2)

    def test_only_prefill_hash_nonzero(self):
        """Request with only prefill hash occupies one slot."""
        assert not self._should_skip({111}, 222, 0, 2)
        assert self._should_skip({111, 333}, 222, 0, 2)

    def test_only_decode_hash_nonzero(self):
        """Request with only decode hash occupies one slot."""
        assert not self._should_skip({111}, 0, 222, 2)
        assert self._should_skip({111, 333}, 0, 222, 2)

    def test_skips_when_at_capacity_with_two_new_hashes(self):
        """Two new hashes when only one slot open -> skip."""
        assert self._should_skip({111}, 222, 333, 2)

    def test_single_slot_capacity(self):
        """Works with max_steering_configs=1."""
        # No hashes -> pass
        assert not self._should_skip(set(), 0, 0, 1)
        # One new hash -> pass if no existing
        assert not self._should_skip(set(), 111, 0, 1)
        # Two distinct new hashes -> skip (needs 2, only 1 available)
        assert self._should_skip(set(), 111, 222, 1)
        # One hash already present + matching -> pass
        assert not self._should_skip({111}, 111, 0, 1)

    # --- Post-admission hash tracking ---

    def test_admit_adds_both_hashes(self):
        """Admitting a request adds both prefill and decode hashes."""
        configs: set[int] = {111}
        self._admit(configs, 222, 333)
        assert configs == {111, 222, 333}

    def test_admit_skips_zero_hashes(self):
        """Zero hashes are not added to the set."""
        configs: set[int] = set()
        self._admit(configs, 111, 0)
        assert configs == {111}

    def test_admit_idempotent_for_existing(self):
        """Adding already-present hashes is a no-op."""
        configs: set[int] = {111, 222}
        self._admit(configs, 111, 222)
        assert configs == {111, 222}

    # --- End-to-end scenario ---

    def test_full_scenario(self):
        """Simulate a sequence of scheduling decisions."""
        max_configs = 3
        configs: set[int] = set()

        # Request 1: prefill_hash=100, decode_hash=200 -> both new
        assert not self._should_skip(configs, 100, 200, max_configs)
        self._admit(configs, 100, 200)
        assert configs == {100, 200}

        # Request 2: prefill_hash=100, decode_hash=300 -> 1 new (300)
        assert not self._should_skip(configs, 100, 300, max_configs)
        self._admit(configs, 100, 300)
        assert configs == {100, 200, 300}

        # Request 3: prefill_hash=400, decode_hash=500 -> 2 new, at cap
        assert self._should_skip(configs, 400, 500, max_configs)

        # Request 4: prefill_hash=100, decode_hash=200 -> 0 new
        assert not self._should_skip(configs, 100, 200, max_configs)

    def test_freed_capacity_admits(self):
        """After removing a config, new ones can be admitted."""
        max_configs = 2
        configs: set[int] = {111, 222}
        # At capacity, new hash blocked
        assert self._should_skip(configs, 333, 0, max_configs)
        # Free a slot
        configs.discard(222)
        # Now admits
        assert not self._should_skip(configs, 333, 0, max_configs)


class TestSteeringAdmissionLogicLegacy:
    """Legacy single-hash admission tests kept for reference.

    These reproduce the simpler check that was in place before dual-hash
    admission.  They should still pass because the dual-hash logic
    degenerates to single-hash behaviour when decode_hash is 0.
    """

    def _should_skip(
        self,
        steering_config_exists: bool,
        request_prefill_hash: int,
        scheduled_configs: set[int],
        max_configs: int,
    ) -> bool:
        """Legacy: single-hash check via the new dual-hash logic."""
        if not steering_config_exists:
            return False
        new_hashes: set[int] = set()
        if request_prefill_hash != 0:
            new_hashes.add(request_prefill_hash)
        # decode_hash=0 (legacy: only prefill hash)
        if not new_hashes:
            return False
        new_unique = new_hashes - scheduled_configs
        return len(scheduled_configs) + len(new_unique) > max_configs

    def test_no_steering_config_always_admits(self):
        assert not self._should_skip(False, 123, {1, 2}, 2)

    def test_no_steering_hash_always_admits(self):
        assert not self._should_skip(True, 0, {1, 2}, 2)

    def test_admits_when_capacity_available(self):
        assert not self._should_skip(True, 333, {111}, 2)

    def test_admits_when_hash_already_scheduled(self):
        assert not self._should_skip(True, 111, {111, 222}, 2)

    def test_skips_when_at_capacity_with_new_hash(self):
        assert self._should_skip(True, 333, {111, 222}, 2)

    def test_single_slot_capacity(self):
        assert not self._should_skip(True, 111, set(), 1)
        assert self._should_skip(True, 222, {111}, 1)
        assert not self._should_skip(True, 111, {111}, 1)

    def test_hash_deduplication(self):
        scheduled = {111}
        assert not self._should_skip(True, 111, scheduled, 1)

    def test_freed_capacity_admits(self):
        scheduled = {111, 222}
        assert self._should_skip(True, 333, scheduled, 2)
        scheduled.discard(222)
        assert not self._should_skip(True, 333, scheduled, 2)
