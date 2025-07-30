# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import pytest
from jsonschema.exceptions import ValidationError

from vllm.config.security_policy import SecurityPolicy


@pytest.mark.parametrize(
    "parameters",
    [
        ('{}', ValidationError),
        ('{"policy":{}}', None),
        ('{"policy":{"signatures":[]}}', ValidationError),  # must be dict
        ('{"policy":{"signatures":{"signers":{}}}}', None),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer": {'  # missing verification_method
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer": {'
            '            "verification_method": "xz"'  # bad verification method
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer": {'
         '            "verification_method": "skip"'
         '        }'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer": {'
            '            "verification_method": "key"'  # missing public_key
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer": {'
         '            "verification_method": "key",'
         '            "public_key": "/foo/bar.pem"'
         '        }'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer": {'  # missing certificate_chain
            '            "verification_method": "certificate"'
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer": {'
         '            "verification_method": "certificate",'
         '            "certificate_chain": ["/foo/bar.pem"]'
         '        }'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer": {'  # missing identity_provider
            '            "verification_method": "sigstore",'
            '            "identity": "foo@bar.pem"'
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer": {'
         '            "verification_method": "sigstore",'
         '            "identity": "foo@bar.pem",'
         '            "identity_provider": "https://foo.bar.com/"'
         '        }'
         '    }'
         ' }'
         '}}', None),
        ('{"policy":{'
         '"signatures":{'
         '    "models":{'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "models":{'
            '        "/foo/bar": {'  # missing 'signer'
            '        }'
            '    }'
            ' }'
            '}}',
            ValidationError),
        (
            '{"policy":{'
            '"signatures":{'
            '    "models":{'
            '        "/foo/bar": {'
            '            "signer": "signer1"'  # signer1 not available
            '        }'
            '    }'
            ' }'
            '}}',
            ValueError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer1": {'
         '            "verification_method": "skip"'
         '        }'
         '    },'
         '    "models":{'
         '        "/foo/bar": {'
         '            "signer": "signer1"'
         '        }'
         '    }'
         ' }'
         '}}', None),
        (
            '{"policy":{'
            '"signatures":{'
            '    "signers":{'
            '        "signer1": {'
            '            "verification_method": "skip"'
            '        }'
            '    },'
            '    "models":{'
            '        "regex:/foo/(bar": {'  # bad regex
            '            "signer": "signer1"'
            '        }'
            '    }'
            ' }'
            '}}',
            ValueError),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer1": {'
         '            "verification_method": "skip"'
         '        }'
         '    },'
         '    "models":{'
         '        "regex:/foo/bar(/)?": {'
         '            "signer": "signer1"'
         '        }'
         '    }'
         ' }'
         '}}', None),
        ('{"policy":{'
         '"signatures":{'
         '    "signers":{'
         '        "signer1": {'
         '            "verification_method": "skip"'
         '        }'
         '    },'
         '    "models":{'
         '        "regex:/foo/ba([rz]{1})(/)?": {'
         '            "signer": "signer1",'
         '            "signature": "foo.sig",'
         '            "ignore_paths": ["/foo/bar","/bar/baz"],'
         '            "ignore_git_paths": false,'
         '            "log_fingerprints": true,'
         '            "use_staging": false'
         '        }'
         '    }'
         ' }'
         '}}', None),
    ])
def test_validate_security_policy(parameters):
    policy_str, exc = parameters

    with tempfile.NamedTemporaryFile(mode="w") as fil:
        fil.write(policy_str)
        fil.flush()

        if exc:
            with pytest.raises(exc):
                SecurityPolicy.from_file(fil.name)
        else:
            SecurityPolicy.from_file(fil.name)
