# SimForge background updater for Windows.
#
# The "SimForge Update" scheduled task runs this at sign-in, on every unlock,
# when the laptop wakes from sleep, and daily at noon (install.ps1 sets it up).
# It updates the app with `git pull` (done as fetch + fast-forward, so a slow
# network after waking can be retried). This file lives in the repository, so
# an update updates the updater too.

$dir = $PSScriptRoot

# SimForge.exe cannot be replaced while it is open. Skip now and try next time.
if (Get-Process SimForge -ErrorAction SilentlyContinue) { exit 0 }

# Git on PATH, or the portable copy install.ps1 puts in %LOCALAPPDATA%\SimForge-Git.
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) { $git = Join-Path $env:LOCALAPPDATA "SimForge-Git\cmd\git.exe" }

# Right after waking, Wi-Fi may not be connected yet: retry for about 2 minutes.
for ($try = 1; $try -le 5; $try++) {
    & $git -C $dir fetch --quiet origin main *> $null
    if ($LASTEXITCODE -eq 0) { break }
    if ($try -eq 5) { exit 0 }            # still offline; the next trigger tries again
    Start-Sleep -Seconds 30
}

# SimForge may have been opened while we waited.
if (Get-Process SimForge -ErrorAction SilentlyContinue) { exit 0 }

& $git -C $dir merge --ff-only --quiet origin/main *> $null

# A fast-forward refuses when a file was changed by hand (or the history was
# squashed). This folder only holds the released app, so put it back to the
# published state.
if ($LASTEXITCODE -ne 0) {
    & $git -C $dir reset --hard --quiet origin/main *> $null
}

# A file an update could not replace (SimForge was opened mid-update) shows as
# modified. Restore it now that SimForge is closed.
if (& $git -C $dir status --porcelain --untracked-files=no 2>$null) {
    & $git -C $dir reset --hard --quiet HEAD *> $null
}
