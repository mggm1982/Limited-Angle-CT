#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_local_cuda.py "$@"
