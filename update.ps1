# SimForge background updater for Windows.
#
# The "SimForge Update" scheduled task runs this every day (install.ps1 sets
# it up). It updates the app with `git pull`. This file lives in the
# repository, so a pull updates the updater too.

$dir = $PSScriptRoot

# SimForge.exe cannot be replaced while it is open. Skip now and try next time.
if (Get-Process SimForge -ErrorAction SilentlyContinue) { exit 0 }

# Git on PATH, or the portable copy install.ps1 puts in %LOCALAPPDATA%\SimForge-Git.
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) { $git = Join-Path $env:LOCALAPPDATA "SimForge-Git\cmd\git.exe" }

& $git -C $dir pull --ff-only --quiet *> $null

# A pull refuses when a file was changed by hand (or the history was squashed).
# This folder only holds the released app, so put it back to the published state.
if ($LASTEXITCODE -ne 0) {
    & $git -C $dir fetch --depth 1 --quiet origin main *> $null
    & $git -C $dir reset --hard --quiet origin/main *> $null
}
