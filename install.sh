#!/bin/bash
#
# SimForge installer for macOS (Apple Silicon, M1 or newer).
#
# Students paste ONE line into Terminal (no admin rights needed):
#
#   curl -fsSL https://raw.githubusercontent.com/engrniel/simforge/main/install.sh | bash
#
# It does everything, and running it again is safe (it repairs and updates):
#   1. Git + Python: Apple's Command Line Tools (asks the student to click Install once).
#   2. SimForge: `git clone` into ~/SimForge (or `git pull` if it is there).
#   3. ~/Applications/SimForge.app, a link to the app inside ~/SimForge.
#   4. A LaunchAgent that runs update.sh (`git pull`) at login and every 6 hours.
#   5. Opens SimForge.

set -u

REPO="https://github.com/engrniel/simforge.git"
DIR="$HOME/SimForge"
LABEL="ph.nrdev.simforge.update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

step() { printf '\033[36m[%s/5] %s\033[0m\n' "$1" "$2"; }
ok()   { printf '\033[32m      %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$1"; exit 1; }

echo ""
echo "SimForge installer"
echo ""

[ "$(uname -s)" = "Darwin" ] || fail "This installer is for macOS. On Windows, open PowerShell and run:
  irm https://raw.githubusercontent.com/engrniel/simforge/main/install.ps1 | iex"
[ "$(uname -m)" = "arm64" ] || fail "SimForge for Mac needs Apple Silicon (M1 or newer). This Mac is $(uname -m)."

# --- 1. Git + Python (Command Line Tools) -----------------------------------
step 1 "Checking Git and Python"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install >/dev/null 2>&1 || true
  echo ""
  printf '\033[33m      A window opened: click "Install", then "Agree".\033[0m\n'
  echo "      This downloads Apple's Command Line Tools (Git + Python). It can take 5-15 minutes."
  echo "      Leave this Terminal open. The installer continues by itself when it is done."
  waited=0
  until xcode-select -p >/dev/null 2>&1 && [ -x /usr/bin/git ]; do
    sleep 10
    waited=$((waited + 10))
    [ "$waited" -ge 3600 ] && fail "Command Line Tools did not finish within an hour. Run the command again to retry."
  done
fi
/usr/bin/git --version >/dev/null 2>&1 || fail "Git is not working. Run: xcode-select --install   then run this command again."
/usr/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null ||
  fail "Python 3.9 or newer is not working. Run: xcode-select --install   then run this command again."
ok "Git and Python are ready"

# --- 2. SimForge -------------------------------------------------------------
step 2 "Downloading SimForge"
if [ -d "$DIR/.git" ]; then
  /usr/bin/git -C "$DIR" pull --ff-only --quiet || {
    /usr/bin/git -C "$DIR" fetch --depth 1 --quiet origin main &&
      /usr/bin/git -C "$DIR" reset --hard --quiet origin/main
  } || fail "Could not update SimForge. Check your internet connection and run the command again."
else
  [ -e "$DIR" ] && mv "$DIR" "$DIR-old-$(date +%Y%m%d-%H%M%S)"
  /usr/bin/git clone --depth 1 --quiet "$REPO" "$DIR" ||
    fail "Could not download SimForge. Check your internet connection and run the command again."
fi
VERSION="$(cat "$DIR/VERSION")"
ok "SimForge $VERSION"

# --- 3. Applications link ------------------------------------------------------
step 3 "Adding SimForge to your Applications folder"
mkdir -p "$HOME/Applications"
# A real SimForge.app left there by the old .dmg updater would swallow the link.
if [ -d "$HOME/Applications/SimForge.app" ] && [ ! -L "$HOME/Applications/SimForge.app" ]; then
  mv "$HOME/Applications/SimForge.app" "$HOME/Applications/SimForge-old-$(date +%Y%m%d-%H%M%S).app"
fi
ln -sfn "$DIR/macos/SimForge.app" "$HOME/Applications/SimForge.app"
ok "~/Applications/SimForge.app"

# --- 4. Automatic updates ------------------------------------------------------
step 4 "Turning on automatic updates"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$DIR/update.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>21600</integer>
</dict>
</plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
if launchctl bootstrap "gui/$(id -u)" "$PLIST" >/dev/null 2>&1; then
  ok "SimForge will update itself at login and every 6 hours"
else
  printf '\033[33m      Automatic updates could not be turned on. To update later, run this same command again.\033[0m\n'
fi

# --- 5. Open -------------------------------------------------------------------
step 5 "Opening SimForge"
open "$HOME/Applications/SimForge.app" || true

echo ""
printf '\033[32mDone! SimForge %s is installed.\033[0m\n' "$VERSION"
echo "Next time, open it from your Applications folder (Finder > Go > Home > Applications),"
echo "or right-click its Dock icon > Options > Keep in Dock."
echo ""
