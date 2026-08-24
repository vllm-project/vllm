# Contributing

## Introduction

We appreciate your interest in considering contributing to go-redis.
Community contributions mean a lot to us.

## Contributions we need

You may already know how you'd like to contribute, whether it's a fix for a bug you
encountered, or a new feature your team wants to use.

If you don't know where to start, consider improving
documentation, bug triaging, and writing tutorials are all examples of
helpful contributions that mean less work for you.

## Your First Contribution

Unsure where to begin contributing? You can start by looking through
[help-wanted
issues](https://github.com/redis/go-redis/issues?q=is%3Aopen+is%3Aissue+label%3ahelp-wanted).

Never contributed to open source before? Here are a couple of friendly
tutorials:

-   <http://makeapullrequest.com/>
-   <http://www.firsttimersonly.com/>

## Getting Started

Here's how to get started with your code contribution:

1.  Create your own fork of go-redis
2.  Do the changes in your fork
3.  If you need a development environment, run `make docker.start`.
 
> Note: this clones and builds the docker containers specified in `docker-compose.yml`, to understand more about
> the infrastructure that will be started you can check the `docker-compose.yml`. You also have the possiblity
> to specify the redis image that will be pulled with the env variable `CLIENT_LIBS_TEST_IMAGE`.
> By default the docker image that will be pulled and started is `redislabs/client-libs-test:8.10.0`.
> If you want to test with newer Redis version, using a newer version of `redislabs/client-libs-test` should work out of the box.

4.  While developing, make sure the tests pass by running `make test` (if you have the docker containers running, `make test.ci` may be sufficient).
> Note: `make test` will try to start all containers, run the tests with `make test.ci` and then stop all containers.
5.  If you like the change and think the project could use it, send a
    pull request

To see what else is part of the automation, run `invoke -l`


## Testing

### Setting up Docker
To run the tests, you need to have Docker installed and running. If you are using a host OS that does not support
docker host networks out of the box (e.g. Windows, OSX), you need to set up a docker desktop and enable docker host networks.

### Running tests
Call `make test` to run all tests.

Continuous Integration uses these same wrappers to run all of these
tests against multiple versions of redis. Feel free to test your
changes against all the go versions supported, as declared by the
[build.yml](./.github/workflows/build.yml) file.

### Troubleshooting

If you get any errors when running `make test`, make sure
that you are using supported versions of Docker and go.

## How to Report a Bug

### Security Vulnerabilities

**NOTE**: If you find a security vulnerability, do NOT open an issue.
Email [Redis Open Source (<oss@redis.com>)](mailto:oss@redis.com) instead.

In order to determine whether you are dealing with a security issue, ask
yourself these two questions:

-   Can I access something that's not mine, or something I shouldn't
    have access to?
-   Can I disable something for other people?

If the answer to either of those two questions are *yes*, then you're
probably dealing with a security issue. Note that even if you answer
*no*  to both questions, you may still be dealing with a security
issue, so if you're unsure, just email [us](mailto:oss@redis.com).

### Everything Else

When filing an issue, make sure to answer these five questions:

1.  What version of go-redis are you using?
2.  What version of redis are you using?
3.  What did you do?
4.  What did you expect to see?
5.  What did you see instead?

## Suggest a feature or enhancement

If you'd like to contribute a new feature, make sure you check our
issue list to see if someone has already proposed it. Work may already
be underway on the feature you want or we may have rejected a
feature like it already.

If you don't see anything, open a new issue that describes the feature
you would like and how it should work.

## Code review process

The core team regularly looks at pull requests. We will provide
feedback as soon as possible. After receiving our feedback, please respond
within two weeks. After that time, we may close your PR if it isn't
showing any activity.

## Support

Maintainers can provide limited support to contributors on discord: https://discord.gg/W4txy5AeKM
