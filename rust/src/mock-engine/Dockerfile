# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

FROM rust:bookworm AS builder

WORKDIR /workspace

COPY rust-toolchain.toml rust-toolchain.toml
RUN rustc --version

COPY rust rust

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    cargo build \
        --manifest-path rust/Cargo.toml \
        --package vllm-mock-engine \
        --release \
        --locked

FROM debian:bookworm-slim

RUN useradd --uid 2000 --gid 0 --create-home vllm

COPY --from=builder \
    /workspace/rust/target/release/vllm-mock-engine \
    /usr/local/bin/vllm-mock-engine

ENV HOME=/home/vllm
WORKDIR /home/vllm
USER 2000

ENTRYPOINT ["vllm-mock-engine"]
