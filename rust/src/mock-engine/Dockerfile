# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

ARG RUST_VERSION=1.95

FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /workspace

COPY rust rust

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    cargo build \
        --manifest-path rust/Cargo.toml \
        --package vllm-mock-engine \
        --release \
        --locked

FROM debian:bookworm-slim

COPY --from=builder \
    /workspace/rust/target/release/vllm-mock-engine \
    /usr/local/bin/vllm-mock-engine

USER 65532:65532

ENTRYPOINT ["vllm-mock-engine"]
