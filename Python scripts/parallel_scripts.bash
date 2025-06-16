#!/usr/bin/env bash
#
# Usage: ./run_parallel.sh path/to/processor.py /path/to/parent_videos_dir
#
# Loops over each subdirectory under the parent directory,
# launching one instance of processor.py per subfolder,
# up to NPROC parallel jobs. Cleans up child processes on exit.

set -euo pipefail
IFS=$'\n\t'

# --- args & validation ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <python_script.py> <parent_videos_dir>"
  exit 1
fi

PYTHON_SCRIPT="$1"
PARENT_DIR="$2"

# Resolve to absolute paths
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
  CORES="$(nproc)"
else
  CORES="$(getconf _NPROCESSORS_ONLN)"
fi
echo "Launching up to $CORES parallel jobs…"

running=0

# --- main loop ---
for subdir in "$PARENT_DIR"/*/; do
  [[ -d "$subdir" ]] || continue

  echo "→ Starting: $subdir"
  python3 "$PYTHON_SCRIPT" "$subdir" &

  (( running++ ))
  if (( running >= CORES )); then
    # wait for any one to finish before launching more
    if wait -n 2>/dev/null; then
      :
    else
      # on older Bash, wait -n may not exist; fallback to wait
      wait
    fi
    # decrement so we keep the throttle correct
    (( running-- ))
  fi
done

# Wait for remaining jobs
wait

echo "✅ All jobs complete."
