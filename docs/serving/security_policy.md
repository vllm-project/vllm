# Security Policy

vLLM supports a security policy for signature verification on AI models and
LoRA adapters to ensure their integrity and provenance. The security policy is a
JSON document that is passed to vLLM using the `--security-policy` command
line option.

## Security Policy for Signature Verification

The signature verification part of the security policy allows to verify
signatures created by the
[model signing library](https://github.com/sigstore/model-transparency)
All of the signature verification methods of this library are support:

| Verification Method | Parameters  |
|---------------------|-------------|
| sigstore    | identity, identity_provider |
| certificate | certificate_chain |
| key         | public_key |

The parameters column in the above table shows the parameters needed
for each one of the verification methods. Those are important when writing
a signature verification policy.

### Signature Verification on Models

A security policy for 'sigstore' signature enforcement on an AI model may
look like this:

```text
{
  "policy": {
    "signatures": {
      "signers": {
        "signer1": {
          "verification_method": "sigstore",
          "identity": "foo@bar.com",
          "identity_provider": "http://baz.com/oauth2"
        }
      },
      "models": {
        "/tmp/foo/" {
          "signer": "signer1"
        }
      }
    }
  }
}
```

The above policy shows a single AI model with the local file path `/tmp/foo`
under its `models` object. It references the signer `signer1`, which is
mentioned above under the `signers` object. It requires the sigstore
verification method and the identity that is expected to have signed the
model must be `foo@bar.com` using the identity_provider `http://baz.com/oauth2`.
Since no other models are enumerated in the `models` object, it will not be
possible to load any other models with this policy. The vLLM log will show
whether signature verification succeeded or failed.

Note that if the `models` object is missing in a policy that all models
can be loaded and signature verification will not be required.

Extending the above shown policy to one that also covers the other signing
methods leads to a more complex policy:

```text
{
  "policy": {
    "signatures": {
      "signers": {
        "signer1": {
          "verification_method": "sigstore",
          "identity": "foo@bar.com",
          "identity_provider": "http://baz.com/oauth2"
        },
        "signer2": {
          "verification_method": "certificate",
          "certificate_chain": ["/tmp/baz/cert1.pem", "/tmp/baz/cert2.pem"]
        },
        "signer3": {
          "verification_method": "key",
          "public_key": "/tmp/baz/pubkey.pem"]
        },
        "no-signer": {
           "verification_method": "skip"
        }
      },
      "models": {
        "/tmp/foo/" {
          "signer": "signer1"
        },
        "regex:/tmp/(baz1|baz2)(/)?": {
          "signer": "signer2",
          "log_fingerprints": true,
          "ignore_paths": ["foo", "bar"],
        },
        "/tmp/test/": {
          "signer": "no-signer"
        },
        "regex:.*": {
          "signer": "signer3",
        }
      }
    }
  }
}
```

The above policy introduces 3 more signers with the methods 'certificate',
'key' and 'skip' along with their required parameter. The 'skip' method
allows access to unsigned AI models or to simply skip the signature
verification on some of them. The vLLM log will show that signature
verification on a particular model was skipped. The 'certificate' and 'key'
methods require each a different set of parameters through which they
reference PEM-formatted certificates or a public key respectively.

The 'models' object holds 3 more paths, of which two are regular
expressions. The first regular expression '/tmp/(baz1|baz2)(/)?' covers
a signature verification rule for the following paths:

- /tmp/baz1
- /tmp/baz1/
- /tmp/baz2
- /tmp/baz2/

The additional log_fingerprints parameter indicates whether to log
the fingerprints of the certificates when verifying the signature.

The following table shows what key value pairs for signer objects are
available:

| Key  | Value Type | Purpose |
|------|------------|---------|
| signature        | filename      | The name of the signature file; default is model.sig |
| log_fingerprints | true or false | Log fingerprints of certificates used for signature verification; only useful if 'certificate' method is used |
| ignore_git_paths | true or false | Ignore git related files such as `.git`, `.gitattributes` in the model path |
| use_staging      | true or false | The staging servers were used for signing with the 'sigstore' method |
| ignore_path      | list of file paths | Files to ignore when verifying the signature |

All files mentioned in the table above are relative to the model path unless they
are given as absolute paths (starting with '/').

To select the signature verification parameters for a particular model,
vLLM will first try to perform an exact path match of the model path from
the command line with the policy. Note that in this case the path `/foo/bar`
will match the path `/foo/bar/' in the policy. All paths that are not
regular expression should therefore end with a '/' in the policy. If no
matching path could be found in the policy, then vLLM will try to matching
the regular expressions against the model path and the first matching regular
expression will be used to select the signature verification parameters.

The model path `/tmp/test/` references the `no-signer` signer, and therefore
will skip signature verification on the model found there.

The last model path is again a regular expression '.*' that covers all
(remaining) paths and since it references `signer3`, it will require that
all these models will have to pass signature verification with a public key.

### Signature Verification on LoRA Adapters

The rules for selecting the signature verification parameter of a LoRA
adapter are the same as those for AI models. The difference is that LoRA
adapters are enumerated in their own 'loras' object as shown in the
following policy:

```text
{
  "policy": {
    "signatures": {
      "signers": {
        "signer1": {
          "verification_method": "sigstore",
          "identity": "foo@bar.com",
          "identity_provider": "http://baz.com/oauth2"
        }
      },
      "loras": {
        "/tmp/bar/" {
          "signer": "signer1"
        }
      }
    }
  }
}
```

This policy enforces 'sigstore' signature verification on the LoRA adapter
in the `/tmp/bar/` directory. Since no other paths are given, it will not
be possible to load any other LoRA adapters. Further, since no "models"
object is provided, no signature verification will be done on any AI models.
