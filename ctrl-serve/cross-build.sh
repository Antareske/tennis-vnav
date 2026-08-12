#!/usr/bin/env bash
# Build ctrl-serve for SG2002 (RISC-V musl).
#
# Prerequisites: nightly rustup toolchain + musl cross toolchain at
# /opt/riscv64-linux-musl-cross

set -euo pipefail
cd "$(dirname "$0")"

chmod +x scripts/linker.sh

cargo build --release \
  --target riscv64gc-unknown-linux-musl \
  -Zbuild-std=std,panic_abort

echo "Done: target/riscv64gc-unknown-linux-musl/release/ctrl-serve"
