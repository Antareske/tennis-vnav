#!/usr/bin/env bash
# ctrl-serve RISC-V musl linker wrapper.

set -euo pipefail

MUSL_GCC="/opt/riscv64-linux-musl-cross/bin/riscv64-linux-musl-gcc"
TOOLCHAIN="/opt/riscv64-linux-musl-cross"
SYSROOT="$TOOLCHAIN/riscv64-linux-musl"
GCC_LIB="$TOOLCHAIN/lib/gcc/riscv64-linux-musl/11.2.1"

if [[ ! -x "$MUSL_GCC" ]]; then
  echo "error: linker not found: $MUSL_GCC" >&2
  exit 1
fi

exec "$MUSL_GCC" \
  -B"$SYSROOT/lib" \
  -B"$GCC_LIB" \
  --sysroot="$SYSROOT" \
  "$@"
