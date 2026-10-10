#!/usr/bin/env bash
# Format, lint, and test both feature configurations. --coverage uses cargo-llvm-cov.
set -euo pipefail
cd "$(dirname "$0")/.."

coverage=0
if [[ "${1:-}" == "--coverage" ]]; then
  coverage=1
  if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
    echo "etc/check.sh: cargo-llvm-cov is not installed (cargo install cargo-llvm-cov)" >&2
    exit 1
  fi
fi

cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings

# Transport tests require the MCP fixture binary, including coverage runs.
cargo build --features mcp --example mcp_fixture

if [[ "$coverage" == 1 ]]; then
  cargo llvm-cov --all-features
else
  cargo test --all-features
fi
cargo test --no-default-features
