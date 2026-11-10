# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.tiering.admission import (
    AdmissionPolicyFactory,
    AlwaysAdmitPolicy,
)


def test_always_admit_policy():
    policy = AlwaysAdmitPolicy()

    assert policy.should_lookup(1, 0, object())
    assert policy.should_admit([1], 0, is_promotion=False)
    assert policy.should_admit([1], 0, is_promotion=True)


def test_admission_policy_factory():
    policy = AdmissionPolicyFactory.create_policy(
        {
            "type": "AlwaysAdmitPolicy",
            "module_path": ("vllm.v1.kv_offload.tiering.admission.always"),
        }
    )

    assert isinstance(policy, AlwaysAdmitPolicy)
