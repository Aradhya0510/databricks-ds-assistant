#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist build *.egg-info
python -m pip install --upgrade build
python -m build
ls -lah dist
