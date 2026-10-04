#!/usr/bin/env bash
# NOTE: historical per-phase runner (phases 01-22 were built this way). Current work is ad hoc; see README.md.
# Runs one FRESH `claude -p` session per phase (that is the "clear context after each phase").
# Stops at live/corpus gates, BLOCKED.md, no-progress, or after MAX_PHASES (default 1).
# Refuses to run unless you choose permissions explicitly (see specs/USER-INPUTS.md U1):
#   CLAUDE_PERMS="--permission-mode acceptEdits --allowedTools Bash Read Write Edit Glob Grep" scripts/run_phases.sh [MAX_PHASES]
#   DRY=1 prints the command instead of running it.
set -u
cd "$(dirname "$0")/.."
MAX="${1:-1}"
: "${CLAUDE_PERMS:?set CLAUDE_PERMS explicitly (see header of this script)}"
PROMPT='Fresh session. Follow specs/AUTONOMY.md exactly: do ONE phase (the one scripts/next_phase.py prints), tests first, pytest -q green, update specs, write HANDOFF.md, then exit. Do not start another phase.'
for ((i = 1; i <= MAX; i++)); do
  out=$(python3 scripts/next_phase.py); code=$?
  echo "[runner] $out"
  [ "$code" -ne 0 ] && exit "$code"
  before=$(md5sum specs/PHASES.md)
  if [ "${DRY:-0}" = 1 ]; then echo "[runner] would run: claude -p \"...\" --model sonnet $CLAUDE_PERMS"; exit 0; fi
  # shellcheck disable=SC2086
  claude -p "$PROMPT" --model sonnet $CLAUDE_PERMS || { echo "[runner] claude exited non-zero"; exit 1; }
  [ "$before" = "$(md5sum specs/PHASES.md)" ] && { echo "[runner] PHASES.md unchanged: no progress, stopping"; exit 1; }
done
