# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.tiering.admission.always import AlwaysAdmitPolicy
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy
from vllm.v1.kv_offload.tiering.admission.factory import AdmissionPolicyFactory


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore AdmissionPolicyFactory._registry between tests."""
    original = dict(AdmissionPolicyFactory._registry)
    yield
    AdmissionPolicyFactory._registry = original


def test_pre_registered_policies_can_be_imported():
    """CI sentinel: registered module paths must import and yield
    TieringAdmissionPolicy subclasses."""
    for policy_type in AdmissionPolicyFactory._registry:
        cls = AdmissionPolicyFactory._registry[policy_type]()
        assert issubclass(cls, TieringAdmissionPolicy)


def test_always_admit_registered():
    cls = AdmissionPolicyFactory.get_policy_class({"type": "always_admit"})
    assert cls is AlwaysAdmitPolicy


def test_create_policy_from_registry():
    policy = AdmissionPolicyFactory.create_policy({"type": "always_admit"})
    assert isinstance(policy, AlwaysAdmitPolicy)


def test_missing_type_raises():
    with pytest.raises(ValueError, match="must include 'type'"):
        AdmissionPolicyFactory.get_policy_class({})


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown admission policy type"):
        AdmissionPolicyFactory.get_policy_class({"type": "nonexistent"})


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="is already registered"):
        AdmissionPolicyFactory.register_policy("always_admit", "some.module", "Cls")


def test_register_new_policy_type():
    """External projects add a custom policy this way, without forking."""
    AdmissionPolicyFactory.register_policy(
        "custom_admit",
        "vllm.v1.kv_offload.tiering.admission.always",
        "AlwaysAdmitPolicy",
    )

    policy = AdmissionPolicyFactory.create_policy({"type": "custom_admit"})

    assert isinstance(policy, AlwaysAdmitPolicy)


def test_create_policy_passes_extra_config_as_kwargs():
    """create_policy() strips 'type' and forwards the rest as constructor
    kwargs - this is how a concrete policy (e.g. a future cap-based one)
    gets configured from extra_config."""

    class _CountingPolicy(TieringAdmissionPolicy):
        def __init__(self, limit: int):
            self.limit = limit

        def should_admit(self, job):
            return True

        def on_admitted(self, job):
            return

        def on_completed(self, job, result):
            return

        def reset(self):
            return

    AdmissionPolicyFactory._registry["counting"] = lambda: _CountingPolicy

    policy = AdmissionPolicyFactory.create_policy({"type": "counting", "limit": 5})

    assert isinstance(policy, _CountingPolicy)
    assert policy.limit == 5
