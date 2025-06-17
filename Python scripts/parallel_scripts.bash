#!/usr/bin/env bash
#
# Usage: ./run_parallel_on_cores.sh path/to/processor.py /path/to/parent_videos_dir
#
# Loops over each subdirectory under the parent directory,
# launching one instance of processor.py per subfolder,
# pinned to its own CPU core (round-robin), and
# cleans up child processes on exit.

set -euo pipefail
IFS=$'\n\t'

# --- args & validation ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <python_script.py> <parent_videos_dir>"
  exit 1
fi

PYTHON_SCRIPT="$1"
PARENT_DIR="$2"

PYTHON_SCRIPT="$(realpath "$PYTHON_SCRIPT")"
PARENT_DIR="$(realpath "$PARENT_DIR")"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$PARENT_DIR" ]]; then
  echo "Error: Directory not found: $PARENT_DIR" >&2
  exit 1
fi

# --- cleanup on interrupt ---
cleanup() {
  echo "Interrupted. Killing all child jobs…" >&2
  pkill -P $$ || true
  exit 1
}
trap cleanup SIGINT SIGTERM

# --- determine core count ---
if command -v nproc &>/dev/null; then
  TOTAL_CORES="$(nproc)"
else
  TOTAL_CORES="$(getconf _NPROCESSORS_ONLN)"
fi
echo "Launching jobs on $TOTAL_CORES cores (round-robin)…"

# --- main loop ---
counter=0
for subdir in "$PARENT_DIR"/*/; do
  [[ -d "$subdir" ]] || continue

  core=$(( counter % TOTAL_CORES ))
  echo "→ [$subdir] → core #$core"
  # bind to core, launch in background
  taskset -c "$core" python3 "$PYTHON_SCRIPT" "$subdir" &

  (( counter++ ))
done

# Wait for all to finish
wait
echo "✅ All jobs complete."
