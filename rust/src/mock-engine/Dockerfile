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
        --package vllm-cmd \
        --package vllm-mock-engine \
        --release \
        --locked

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 2000 --gid 0 --create-home vllm \
    && install --directory --owner 2000 --group 0 /home/vllm/.cache/huggingface

COPY --from=builder \
    /workspace/rust/target/release/vllm-mock-engine \
    /usr/local/bin/vllm-mock-engine
COPY --from=builder \
    /workspace/rust/target/release/vllm-rs \
    /usr/local/bin/vllm-rs

ENV HOME=/home/vllm
WORKDIR /home/vllm
USER 2000

ENTRYPOINT ["vllm-mock-engine"]
