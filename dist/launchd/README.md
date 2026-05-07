# ios-graphrag-watch — launchd Setup

This directory ships a sample `launchd` per-user agent plist that auto-starts
`ios-graphrag-watch` at login. The watcher monitors your iOS repo and triggers
an incremental reindex on each meaningful change, keeping the embedding model
warm across runs (Phase 4b/4c). The plist is **opt-in** — no install script
runs automatically.

## What you get

- A long-lived process that calls `index_repository(..., full_reindex=False)`
  every time you save a `.swift`, `.h`, or `.m` file.
- Sliding 1-second debounce window so a burst of saves coalesces into one
  reindex.
- Logs to `~/Library/Logs/ios-graphrag/watcher.log` (JSON, rotated at 50 MB ×
  5) plus `watcher.stdout.log` / `watcher.stderr.log` from launchd itself.
- `KeepAlive` — launchd restarts the watcher if it crashes.

## Fill in the placeholders

Open `com.user.ios-graphrag.watcher.plist` and replace these tokens:

| Placeholder | Replace with |
|---|---|
| `/PATH/TO/.venv/bin/ios-graphrag-watch` | Absolute path to the console script in your venv (e.g. `/Users/<you>/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-watch`) |
| `/PATH/TO/YOUR/IOS/REPO` | Absolute path to the repo root you want indexed |
| `/Users/YOUR_USER/.cache/ios-graphrag/graph.sqlite` | Absolute path to the SQLite DB (must match `GRAPH_DB_PATH` your MCP server uses) |
| `/Users/YOUR_USER` | Your home directory (3 occurrences: `WorkingDirectory`, `StandardOutPath`, `StandardErrorPath`) |

> All paths in launchd plists must be absolute. No `~`, no `$HOME`, no
> environment expansion of `~/Library/Logs/...`. Spell every `/Users/<you>/...`
> out.

## Install

```bash
# 1. Make sure the log directory exists (launchd will not mkdir for you).
mkdir -p ~/Library/Logs/ios-graphrag

# 2. Copy the plist into your LaunchAgents directory.
cp dist/launchd/com.user.ios-graphrag.watcher.plist \
   ~/Library/LaunchAgents/com.user.ios-graphrag.watcher.plist

# 3. Bootstrap (load + enable) the agent.
launchctl bootstrap gui/$(id -u) \
   ~/Library/LaunchAgents/com.user.ios-graphrag.watcher.plist
```

## Start / stop / status

```bash
# Status (look for State = running and a non-zero PID).
launchctl print gui/$(id -u)/com.user.ios-graphrag.watcher

# Stop without unloading (will auto-restart due to KeepAlive=true).
launchctl kill SIGTERM gui/$(id -u)/com.user.ios-graphrag.watcher

# Disable + unload (full removal).
launchctl bootout gui/$(id -u) \
   ~/Library/LaunchAgents/com.user.ios-graphrag.watcher.plist
```

## Per-user vs system-wide

This sample is a **per-user agent** (`gui/$UID` domain, plist under
`~/Library/LaunchAgents`):

- Runs as you, only while you are logged in.
- Has access to your home directory, your `.venv`, and the repo at the path
  you pasted.
- Is the right scope for a developer tool. Each developer installs their own.

A system-wide daemon (`/Library/LaunchDaemons`, `system` domain) is **not
recommended** for this watcher: it would run as root, need explicit `UserName`
keys, and have no reason to access individual developer repos. If you want a
shared-machine setup (e.g. CI), invoke `ios-graphrag-watch` from your CI
runner's process supervisor instead.

## Verify it's working

```bash
# Touch a Swift file in the watched repo, then look for the flush log line:
touch /PATH/TO/YOUR/IOS/REPO/SomeFile.swift
tail -f ~/Library/Logs/ios-graphrag/watcher.log | grep -E 'debounce flush|incremental reindex'
```

You should see `debounce flush: reindexing after N changed file(s)` followed by
`incremental reindex complete` within ~1 second.
