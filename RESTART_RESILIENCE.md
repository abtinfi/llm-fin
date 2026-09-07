# Surviving the 03:01 reboot and an SSH drop

The machine reboots **every day at 03:01** and this is not disableable from
this account (`last -x reboot` shows an unbroken run). The 2026-09-02 scaled
pipeline run was lost to exactly that.

Two different failures, and they need different fixes. Conflating them is the
usual mistake:

| failure | what saves you | what does NOT |
|---|---|---|
| SSH drops / laptop sleeps | `tmux` — the process keeps running on the server | — |
| the 03:01 reboot | the `@reboot` cron entries below | **tmux does not survive a reboot** — the tmux server is a process and dies with everything else |

---

## 1. The Claude Code session

```bash
bash ~/abtin/claude_session.sh    # start it inside tmux (idempotent)
tmux attach -t claude             # reattach
# Ctrl-b then d                   # detach, leaving it running
```

Wired to `@reboot`, so after the nightly restart the session is back and
holding the conversation. `claude --continue` resumes the most recent
conversation for `/home/asosoft/abtin` and then **waits at an interactive
prompt** — it does not replay the transcript or take any action on its own.
The restart makes the session available again; it does not act unattended.

The conversation itself lives in `~/.claude/projects/`, not in the terminal,
so it is never lost by a disconnect — only the live process is. That is why
reattaching (or letting `@reboot` restart it) recovers everything.

`claude_session.sh` refuses to start a second copy while one is already
running outside tmux, because two processes on one conversation history is a
mess. Override with `FORCE=1` only if you know the other one is gone.

**Migrating a session that is already running outside tmux:** it cannot be
moved in place (that needs `reptyr`, which is not installed). Exit it, run
`bash ~/abtin/claude_session.sh`, then `tmux attach -t claude` — the
conversation resumes.

## 2. The GPU pipeline

```bash
cd ~/abtin/paper/csai && ./status.sh    # where everything stands
```

`csai_supervisor.sh` is a two-worker pool, one per GPU, pulling from a queue.
It is started by `supervisor_launch.sh` from `@reboot` **and** from a `*/15`
watchdog, so it also recovers from a crash that is not a reboot.

Why it is safe to fire that launcher every fifteen minutes:

* a live supervisor holds `.pipeline_state/supervisor.lock`, and the launcher
  exits immediately if that lock is taken;
* every job script takes its own `flock`, so even a hand-launched copy cannot
  double-run a stage;
* every stage is checkpointed to `.pipeline_state/`, so a restart **resumes**
  rather than restarts;
* jobs are claimed with `mkdir`, which is atomic, so two workers never take
  the same job.

**The recovery that matters after a reboot.** A claim directory left behind by
a job that never finished would make that job permanently unclaimable — the
workers would see an empty queue and declare the run complete having done
nothing. `supervisor_launch.sh` clears stale claims, and removes a
`SUPERVISOR_ALL_DONE` written while job markers are still missing, but **only
when no supervisor holds the lock**. Verified in a sandbox reboot simulation.

## 3. Adding a job without killing the running supervisor

Never edit a shell script that is executing. Bash reads scripts by byte
offset, so inserting lines makes the running shell resume parsing at the wrong
place and die with a misleading syntax error — this is how the OpenBioLLM run
was killed on 2026-09-07 (its lane A had checkpointed, so it resumed).

The supervisor builds its queue from a heredoc inside itself, so adding a job
means editing it. Stage the new version instead:

```bash
cp csai_supervisor.sh csai_supervisor.sh.next
# edit the .next file
```

`supervisor_launch.sh` promotes `.next` at the one moment no supervisor is
running, keeps the previous version as `.prev`, and refuses to promote
anything that does not pass `bash -n`.

## 4. Current cron

```
@reboot sleep 45   claude_session.sh          the Claude Code session
@reboot sleep 120  supervisor_launch.sh       the GPU pipeline
*/15   * * * *     supervisor_launch.sh       watchdog for non-reboot crashes
```

`scaled_launch.sh` was retired from `@reboot` on 2026-09-07. It launched both
lanes in parallel across both cards and would have raced the supervisor; it
was only harmless because `.pipeline_state/SCALED_PIPELINE_COMPLETE` happened
to exist. The commented-out line is kept in the crontab with that reason.

Crontab backups are in `paper/csai/logs/scaled/crontab_backup_*.txt`.

## 5. Checking it actually works

```bash
cd ~/abtin/paper/csai && ./status.sh
tmux ls                                   # csai_sup, and claude after a reboot
crontab -l | grep -v '^#'
tail ~/abtin/logs_claude_session.log
tail paper/csai/logs/supervisor/supervisor.log
```

A job that depends on nothing but `init` is the point. Verify with:

```bash
ps -eo pid,ppid,cmd | grep -E 'run_model|run_medcalc|csai_supervisor'
```

Every ancestry should terminate at `init(1)` — never at an `sshd` or a
`claude` process. If a job descends from your shell, it dies when you
disconnect.
