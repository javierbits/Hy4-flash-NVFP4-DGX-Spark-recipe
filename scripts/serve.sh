#!/usr/bin/env bash
# Resolves javierbits/Hy4-flash-NVFP4 from the Hugging Face hub cache and checks it.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
checker="${root}/scripts/check_checkpoint.py"

path="$(python3 "${checker}" --find)"
python3 "${checker}" "${path}"
echo "checkpoint ok: ${path}"
exit 2
