#!/usr/bin/env bash
# Idempotent tmux entry point for csai_supervisor.sh. Safe to call from
# @reboot AND from the */15 watchdog: if a supervisor is already running, or
# all the work is genuinely finished, it exits 0 without doing anything.
set -u
cd /home/asosoft/abtin/paper/csai
mkdir -p logs/supervisor .pipeline_state/claims
STAMP="$(date -Is)"
log () { echo "$STAMP $*" >> logs/supervisor/launcher.log; }

# A supervisor holds .pipeline_state/supervisor.lock for its whole life, which
# is a more reliable liveness test than the tmux session name: a crashed
# session can leave the name behind, and a session started by hand may not use
# that name at all.
if ! flock -n .pipeline_state/supervisor.lock -c true 2>/dev/null; then
  log "a supervisor holds the lock; nothing to do"
  exit 0
fi
if tmux has-session -t csai_sup 2>/dev/null; then
  log "session csai_sup already running"
  exit 0
fi

# ---- from here on, NO supervisor is running, so recovery is safe ----------

# STALE CLAIMS. `next_job` claims a job by creating a directory, and a reboot
# (or a kill) leaves that directory behind for a job that never finished. Left
# alone they make every unfinished job permanently unclaimable, the workers
# find an "empty" queue, and the run is declared complete having done nothing.
# Clearing them is only safe when no supervisor is live -- which is exactly
# the state established above.
if [ -n "$(ls -A .pipeline_state/claims 2>/dev/null)" ]; then
  log "clearing stale claims: $(ls .pipeline_state/claims | tr '\n' ' ')"
  rm -rf .pipeline_state/claims
  mkdir -p .pipeline_state/claims
fi

# PREMATURE COMPLETION. Only honour the done-marker if every job's own marker
# is really present; otherwise a mis-declared completion would silently retire
# the watchdog and the remaining work would never run.
if [ -e .pipeline_state/SUPERVISOR_ALL_DONE ]; then
  missing=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    marker="${line%%|*}"
    [ -e "$marker" ] || missing=$((missing + 1))
  done < <(grep -v '^\s*$' .pipeline_state/queue.txt 2>/dev/null || true)
  if [ "$missing" -gt 0 ]; then
    log "SUPERVISOR_ALL_DONE present but $missing job marker(s) missing -- removing it and restarting"
    rm -f .pipeline_state/SUPERVISOR_ALL_DONE
  else
    log "all work complete -- not relaunching"
    exit 0
  fi
fi

tmux new-session -d -s csai_sup "bash csai_supervisor.sh"
log "started tmux session csai_sup"
