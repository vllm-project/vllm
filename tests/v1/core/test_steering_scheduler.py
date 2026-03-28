# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering config admission control logic.

These tests verify the steering capacity check that mirrors the LoRA
admission control pattern in the scheduler.
"""


class TestSteeringAdmissionLogic:
    """Test the steering admission control logic in isolation.

    The scheduler checks:
        if (steering_config
            and request.steering_config_hash != 0
            and (len(scheduled_steering_configs) == max_steering_configs
                 and request.steering_config_hash
                     not in scheduled_steering_configs)):
            skip request
    """

    def _should_skip(
        self,
        steering_config_exists: bool,
        request_hash: int,
        scheduled_configs: set[int],
        max_configs: int,
    ) -> bool:
        """Reproduce the scheduler's steering admission check."""
        return (
            steering_config_exists
            and request_hash != 0
            and (
                len(scheduled_configs) == max_configs
                and request_hash not in scheduled_configs
            )
        )

    def test_no_steering_config_always_admits(self):
        """Without steering config, all requests pass."""
        assert not self._should_skip(False, 123, {1, 2}, 2)

    def test_no_steering_hash_always_admits(self):
        """Requests without steering (hash=0) always pass."""
        assert not self._should_skip(True, 0, {1, 2}, 2)

    def test_admits_when_capacity_available(self):
        """Request admitted when slots available."""
        assert not self._should_skip(True, 333, {111}, 2)

    def test_admits_when_hash_already_scheduled(self):
        """Request with already-scheduled hash doesn't consume new slot."""
        assert not self._should_skip(True, 111, {111, 222}, 2)

    def test_skips_when_at_capacity_with_new_hash(self):
        """New hash skipped when all slots occupied."""
        assert self._should_skip(True, 333, {111, 222}, 2)

    def test_single_slot_capacity(self):
        """Works with max_steering_configs=1."""
        assert not self._should_skip(True, 111, set(), 1)
        assert self._should_skip(True, 222, {111}, 1)
        assert not self._should_skip(True, 111, {111}, 1)

    def test_hash_deduplication(self):
        """Multiple requests with same hash occupy one slot."""
        scheduled = {111}  # one config in use
        # Same hash passes even at capacity
        assert not self._should_skip(True, 111, scheduled, 1)

    def test_freed_capacity_admits(self):
        """After removing a config, new ones can be admitted."""
        scheduled = {111, 222}
        assert self._should_skip(True, 333, scheduled, 2)
        # Simulate freeing a slot
        scheduled.discard(222)
        assert not self._should_skip(True, 333, scheduled, 2)
