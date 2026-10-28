#!/bin/bash
#
# SimForge background updater for macOS.
#
# The ph.nrdev.simforge.update LaunchAgent runs this at login and every 6 hours
# (see README.md). It updates the app with `git pull`. This file lives in the
# repository, so a pull updates the updater too.

dir="$(cd "$(dirname "$0")" && pwd)"

# A pull refuses when a file was changed by hand (or the history was squashed).
# This folder only holds the released app, so put it back to the published state.
/usr/bin/git -C "$dir" pull --ff-only --quiet || {
  /usr/bin/git -C "$dir" fetch --depth 1 --quiet origin main &&
    /usr/bin/git -C "$dir" reset --hard --quiet origin/main
}
