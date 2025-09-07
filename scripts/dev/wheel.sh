#!/bin/bash
set -e
set -o pipefail

# This script builds wheels for Setu
# Can be used both locally and in CI environments

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_root_dir=$(dirname "$(dirname "$_script_dir")")

# Load utilities and initialize CI environment if needed
source "${_script_dir}/utils.sh"
init_ci_environment

# Verify required environment variables in CI context
if [[ "${SETU_IS_CI_CONTEXT:-0}" == "true" ]]; then
    if [[ -z "${PYPI_RELEASE_CUDA_VERSION:-}" ]]; then
        echo "Error: PYPI_RELEASE_CUDA_VERSION not set"
        exit 1
    fi
    if [[ -z "${PYPI_RELEASE_TORCH_VERSION:-}" ]]; then
        echo "Error: PYPI_RELEASE_TORCH_VERSION not set"
        exit 1
    fi
    if [[ -z "${IS_NIGHTLY_BUILD:-}" ]]; then
        echo "Error: IS_NIGHTLY_BUILD not set"
        exit 1
    fi
fi

cd "$_root_dir"

# Use the build.sh wheel function which handles all dependencies correctly
"${_script_dir}/build.sh" wheel
