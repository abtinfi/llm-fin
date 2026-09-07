#!/usr/bin/env bash
# Keep the Claude Code session alive across SSH drops AND the 03:01 reboot.
#
# WHAT PROTECTS AGAINST WHAT -- these are two different failures and tmux only
# fixes one of them:
#
#   SSH disconnect  tmux fixes it. The session keeps running on the server and
#                   you reattach later.
#   reboot          tmux does NOT fix it. The tmux server is a process and dies
#                   with everything else. Only the @reboot cron entry that runs
#                   this script brings it back.
#
# `claude --continue` resumes the most recent conversation for the working
# directory, so after a reboot you reattach and the history is intact.
#
# It resumes into an interactive prompt and WAITS. It does not replay or
# auto-execute anything -- the restart makes the session available again, it
# does not act on its own.
#
# Usage:
#   bash claude_session.sh          start it (idempotent)
#   tmux attach -t claude           reattach
#   Ctrl-b d                        detach, leaving it running
set -u

SESSION="${CLAUDE_TMUX_SESSION:-claude}"
WORKDIR="${CLAUDE_WORKDIR:-/home/asosoft/abtin}"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
LOG="$WORKDIR/logs_claude_session.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
log () { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -x "$CLAUDE_BIN" ]; then
  log "FATAL: $CLAUDE_BIN is not executable"
  echo "claude binary not found at $CLAUDE_BIN" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "session '$SESSION' already exists -- nothing to do"
  echo "tmux session '$SESSION' is already running.  Attach with:  tmux attach -t $SESSION"
  exit 0
fi

# A claude already running OUTSIDE tmux would end up as a second process on the
# same conversation history. Refuse rather than create the conflict, unless
# explicitly overridden -- after a reboot there is no such process, so the
# @reboot path is unaffected.
# Match claude as the COMMAND, not as a substring of someone's arguments.
# The first pattern tried also matched this script's own shell, because its
# command line mentions ~/.claude/ -- a false positive here would be harmless
# at reboot (nothing is running) but blocks the manual path every time.
OTHER=$(pgrep -u "$(id -u)" -x claude 2>/dev/null | head -1)
if [ -n "${OTHER:-}" ] && [ "${FORCE:-0}" != "1" ]; then
  log "refusing: claude already running outside tmux (pid $OTHER)"
  cat >&2 <<MSG
A Claude Code process is already running outside tmux (pid $OTHER).

Starting a second one would put two processes on the same conversation
history. Do this instead:

  1. finish or exit that session
  2. bash $0
  3. tmux attach -t $SESSION

Your conversation is resumed either way -- it lives in
~/.claude/projects/, not in the terminal.

To start anyway:  FORCE=1 bash $0
MSG
  exit 2
fi

cd "$WORKDIR" || exit 1
tmux new-session -d -s "$SESSION" -c "$WORKDIR" "$CLAUDE_BIN --continue"
log "started tmux session '$SESSION' running: $CLAUDE_BIN --continue (cwd $WORKDIR)"
echo "Started Claude Code in tmux session '$SESSION'."
echo "Attach:  tmux attach -t $SESSION      Detach:  Ctrl-b then d"
