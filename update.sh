#!/bin/bash
#
# SimForge background updater for macOS.
#
# The ph.nrdev.simforge.update LaunchAgent runs this at login and every hour;
# launchd runs a check missed during sleep as soon as the Mac wakes. It updates
# the app with `git pull` (done as fetch + fast-forward, so a slow network after
# waking can be retried). This file lives in the repository, so an update
# updates the updater too.

dir="$(cd "$(dirname "$0")" && pwd)"
git=/usr/bin/git

# Right after waking, Wi-Fi may not be connected yet: retry for about 2 minutes.
try=1
until "$git" -C "$dir" fetch --quiet origin main 2>/dev/null; do
  [ "$try" -ge 5 ] && exit 0               # still offline; the next run tries again
  try=$((try + 1))
  sleep 30
done

# A fast-forward refuses when a file was changed by hand (or the history was
# squashed). This folder only holds the released app, so put it back to the
# published state.
"$git" -C "$dir" merge --ff-only --quiet origin/main 2>/dev/null ||
  "$git" -C "$dir" reset --hard --quiet origin/main
