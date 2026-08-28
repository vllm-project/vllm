# Contributing

Thanks for your interest in contributing! This document contains `nats-io/nats.go` specific contributing details. If you
are a first-time contributor, please refer to the general [NATS Contributor Guide](https://nats.io/contributing/) to get
a comprehensive overview of contributing to the NATS project.

## Getting started

There are three general ways you can contribute to this repo:

- Proposing an enhancement or new feature
- Reporting a bug or regression
- Contributing changes to the source code

For the first two, refer to the [GitHub Issues](https://github.com/nats-io/nats.go/issues/new/choose) which guides you
through the available options along with the needed information to collect.

## Contributing changes

_Prior to opening a pull request, it is recommended to open an issue first to ensure the maintainers can review intended
changes. Exceptions to this rule include fixing non-functional source such as code comments, documentation or other
supporting files._

Proposing source code changes is done through GitHub's standard pull request workflow.

If your branch is a work-in-progress then please start by creating your pull requests as draft, by clicking the
down-arrow next to the `Create pull request` button and instead selecting `Create draft pull request`.

This will defer the automatic process of requesting a review from the NATS team and significantly reduces noise until
you are ready. Once you are happy, you can click the `Ready for review` button.

### Guidelines

A good pull request includes:

- A high-level description of the changes, including links to any issues that are related by adding comments
  like `Resolves #NNN` to your description.
  See [Linking a Pull Request to an Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
  for more information.
- An up-to-date parent commit. Please make sure you are pulling in the latest `main` branch and rebasing your work on
  top of it, i.e. `git rebase main`.
- Unit tests where appropriate. Bug fixes will benefit from the addition of regression tests. New features will not be
  accepted without suitable test coverage!
- No more commits than necessary. Sometimes having multiple commits is useful for telling a story or isolating changes
  from one another, but please squash down any unnecessary commits that may just be for clean-up, comments or small
  changes.
- No additional external dependencies that aren't absolutely essential. Please do everything you can to avoid pulling in
  additional libraries/dependencies into `go.mod` as we will be very critical of these.

### Sign-off

In order to accept a contribution, you will first need to certify that the contribution is your original work and that
you license the work to the project under
the [Apache-2.0 license](https://github.com/nats-io/nats.go/blob/main/LICENSE).

This is done by using `Signed-off-by` statements, which should appear in **both** your commit messages and your PR
description. Please note that we can only accept sign-offs under a legal name. Nicknames and aliases are not permitted.

To perform a sign-off with `git`, use `git commit -s` (or `--signoff`).

## Get help

If you have questions about the contribution process, please start
a [GitHub discussion](https://github.com/nats-io/nats.go/discussions), join the [NATS Slack](https://slack.nats.io/), or
send your question to the [NATS Google Group](https://groups.google.com/forum/#!forum/natsio).

## Testing

The integration tests run against real NATS servers spawned by the
[synadia/ntf-server](https://hub.docker.com/r/synadia/ntf-server) docker service. See [TESTING.md](TESTING.md) for the
full workflow; the short version:

```shell
make tester-up-host                  # once: start the test-server manager (requires docker)
make test T=TestName PKG=./test/...  # iterate on a single test
make test                            # full race-enabled suite
make tester-down                     # tear everything down
```

Without a running tester (`TESTER_NATS_URL` unset), `go test ./...` runs only the unit tests — the integration suites
skip silently, so a green run on a fresh checkout does not mean the integration tests passed.

You should use `go_test.mod` to manage your testing dependencies. Please use the following command to update your
dependencies and avoid changing the main `go.mod` in a PR:

```shell
go mod tidy -modfile=go_test.mod
```
