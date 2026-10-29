# SimForge installer for Windows 10/11.
#
# Students paste ONE line into PowerShell (no admin rights needed):
#
#   irm https://raw.githubusercontent.com/engrniel/simforge/main/install.ps1 | iex
#
# It does everything, and running it again is safe (it repairs and updates):
#   1. Git: uses the one on PATH, or installs portable MinGit for this user.
#   2. Python 3.9+: uses the one on PATH, or installs Python 3.12 for this user.
#   3. SimForge: `git clone` into %LOCALAPPDATA%\SimForge (or `git pull` if it is there).
#   4. Desktop and Start Menu shortcuts.
#   5. The "SimForge Update" scheduled task, which runs update.ps1 (`git pull`) at
#      sign-in, on every unlock (so after waking from sleep) and daily at noon.
#
# Everything sits inside one script block, so `irm | iex` leaves no variables
# behind and an error never closes the student's PowerShell window.

& {
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"   # the progress bar makes downloads 10x slower in PS 5.1
    # Windows PowerShell 5.1 still defaults to TLS 1.0, which GitHub refuses.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $Repo      = "https://github.com/engrniel/simforge.git"
    $Dir       = Join-Path $env:LOCALAPPDATA "SimForge"
    $GitHome   = Join-Path $env:LOCALAPPDATA "SimForge-Git"
    $MinGitUrl = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip"
    $PythonUrl = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"

    function Step($n, $text) { Write-Host "[$n/5] $text" -ForegroundColor Cyan }
    function Ok($text)       { Write-Host "      $text" -ForegroundColor Green }
    function Fail($text)     { Write-Host ""; Write-Host $text -ForegroundColor Red }

    function Add-UserPath($p) {
        $user = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not (($user -split ";") -contains $p)) {
            $new = if ($user) { "$user;$p" } else { $p }
            [Environment]::SetEnvironmentVariable("Path", $new, "User")
        }
        if (-not (($env:Path -split ";") -contains $p)) { $env:Path = "$p;$env:Path" }
    }

    function Update-SessionPath {
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "User")
    }

    # True when `python` or `py -3` is a real Python 3.9+. The Microsoft Store
    # "python" alias is not: it prints an error instead of running the code.
    function Test-Python {
        $probe = "import sys; print(sys.version_info >= (3, 9))"
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try {
            if ((& python -c $probe 2>$null) -eq "True") { return $true }
        } catch {}
        try {
            if ((& py -3 -c $probe 2>$null) -eq "True") { return $true }
        } catch {}
        finally { $ErrorActionPreference = $old }
        return $false
    }

    # Git writes progress to stderr, which PowerShell 5.1 can turn into errors.
    function Invoke-Git {
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & git @args *> $null
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
        return $code
    }

    Write-Host ""
    Write-Host "SimForge installer" -ForegroundColor Cyan
    Write-Host ""

    # --- 1. Git ------------------------------------------------------------
    Step 1 "Checking Git"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $gitExe = Join-Path $GitHome "cmd\git.exe"
        if (-not (Test-Path $gitExe)) {
            Write-Host "      installing Git (about 45 MB)..."
            $zip = Join-Path $env:TEMP "simforge-mingit.zip"
            try {
                Invoke-WebRequest $MinGitUrl -OutFile $zip -UseBasicParsing
            } catch {
                Fail "Could not download Git. Check your internet connection and run the command again."
                return
            }
            if (Test-Path $GitHome) { Remove-Item -Recurse -Force $GitHome }
            Expand-Archive -Path $zip -DestinationPath $GitHome -Force
            Remove-Item $zip -Force
        }
        Add-UserPath (Join-Path $GitHome "cmd")
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Fail "Git did not install. Run the command again, or install Git from git-scm.com."
            return
        }
    }
    Ok "Git is ready"

    # --- 2. Python ---------------------------------------------------------
    Step 2 "Checking Python"
    if (-not (Test-Python)) {
        Write-Host "      installing Python 3.12 (about 27 MB, takes a minute)..."
        $exe = Join-Path $env:TEMP "simforge-python-setup.exe"
        try {
            Invoke-WebRequest $PythonUrl -OutFile $exe -UseBasicParsing
        } catch {
            Fail "Could not download Python. Check your internet connection and run the command again."
            return
        }
        # Per-user install: no admin rights, adds python to this user's PATH.
        $p = Start-Process $exe -Wait -PassThru -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0",
            "Include_launcher=1", "InstallLauncherAllUsers=0")
        Remove-Item $exe -Force -ErrorAction SilentlyContinue
        Update-SessionPath
        if ($p.ExitCode -ne 0 -or -not (Test-Python)) {
            Fail "Python did not install (code $($p.ExitCode)). Install it from python.org, tick 'Add python.exe to PATH', then run the command again."
            return
        }
    }
    Ok "Python is ready"

    # --- 3. SimForge -------------------------------------------------------
    Step 3 "Downloading SimForge"
    if (Get-Process SimForge -ErrorAction SilentlyContinue) {
        Write-Host "      closing SimForge so it can be updated..."
        Get-Process SimForge | Stop-Process -Force
        Start-Sleep -Seconds 1
    }
    if (Test-Path (Join-Path $Dir ".git")) {
        if ((Invoke-Git -C $Dir pull --ff-only --quiet) -ne 0) {
            Invoke-Git -C $Dir fetch --depth 1 --quiet origin main | Out-Null
            Invoke-Git -C $Dir reset --hard --quiet origin/main | Out-Null
        }
    } else {
        if (Test-Path $Dir) {
            Rename-Item $Dir ("SimForge-old-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        }
        if ((Invoke-Git clone --depth 1 --quiet $Repo $Dir) -ne 0) {
            Fail "Could not download SimForge. Check your internet connection and run the command again."
            return
        }
    }
    $version = (Get-Content (Join-Path $Dir "VERSION") -Raw).Trim()
    Ok "SimForge $version"

    # --- 4. Shortcuts ------------------------------------------------------
    Step 4 "Adding Desktop and Start Menu shortcuts"
    $app = Join-Path $Dir "windows\SimForge.exe"
    $ws = New-Object -ComObject WScript.Shell
    foreach ($folder in [Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs")) {
        $s = $ws.CreateShortcut((Join-Path $folder "SimForge.lnk"))
        $s.TargetPath = $app
        $s.WorkingDirectory = Split-Path $app
        $s.IconLocation = "$app,0"
        $s.Save()
    }
    Ok "Shortcuts added"

    # --- 5. Automatic updates ----------------------------------------------
    Step 5 "Turning on automatic updates"
    $updater = Join-Path $Dir "update.ps1"
    # `conhost --headless` runs the updater with no window at all. A plain
    # -WindowStyle Hidden still flashes a console (or opens Windows Terminal on
    # Windows 11) every time the task runs.
    $action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$updater`""
    # StartWhenAvailable: a run missed while the laptop was asleep or off
    # happens as soon as it is awake again.
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
    $me = "$env:USERDOMAIN\$env:USERNAME"
    $daily = New-ScheduledTaskTrigger -Daily -At 12pm
    $triggers = @($daily)
    try { $triggers += New-ScheduledTaskTrigger -AtLogOn -User $me } catch {}
    try {
        # Unlock: waking from sleep ends on the lock screen, so this runs after every wake.
        $cls = Get-CimClass -Namespace ROOT\Microsoft\Windows\TaskScheduler -ClassName MSFT_TaskSessionStateChangeTrigger
        $triggers += New-CimInstance -CimClass $cls -ClientOnly -Property @{ StateChange = 8; UserId = $me; Enabled = $true }
    } catch {}
    try {
        # Wake: Windows logs "resumed from sleep" (Power-Troubleshooter, event 1),
        # so this also covers laptops with no lock screen.
        $ecls = Get-CimClass -Namespace ROOT\Microsoft\Windows\TaskScheduler -ClassName MSFT_TaskEventTrigger
        $wake = "<QueryList><Query Id=`"0`" Path=`"System`"><Select Path=`"System`">" +
                "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]" +
                "</Select></Query></QueryList>"
        $triggers += New-CimInstance -CimClass $ecls -ClientOnly -Property @{ Subscription = $wake; Enabled = $true }
    } catch {}
    # Some locked-down laptops refuse a trigger type. Drop triggers from the end
    # (wake, then unlock, then sign-in) until Windows accepts the task. The daily
    # noon trigger always stays, and it still catches up after sleep.
    $used = 0
    $why = ""
    for ($n = $triggers.Count; $n -ge 1; $n--) {
        try {
            Register-ScheduledTask -TaskName "SimForge Update" -Action $action -Trigger $triggers[0..($n - 1)] `
                -Settings $settings -Force -ErrorAction Stop | Out-Null
            $used = $n
            break
        } catch { $why = $_.Exception.Message }
    }
    if ($used -eq 4) {
        Ok "SimForge will update itself at sign-in, after waking from sleep, and daily at noon"
    } elseif ($used -ge 2) {
        Ok "SimForge will update itself at sign-in and daily at noon (a missed update runs as soon as the laptop is awake)"
    } elseif ($used -eq 1) {
        Ok "SimForge will update itself daily at noon (a missed update runs as soon as the laptop is awake)"
    } else {
        Write-Host "      Automatic updates could not be turned on ($why)." -ForegroundColor Yellow
        Write-Host "      To update later, run this same command again."
    }
    if ($used -ge 1) {
        # Run the updater once now, so the required check below shows a real
        # result (LastTaskResult 0) instead of "never run". Wait up to 30 s.
        try {
            Start-ScheduledTask -TaskName "SimForge Update"
            for ($i = 0; $i -lt 30; $i++) {
                Start-Sleep -Seconds 1
                if ((Get-ScheduledTask -TaskName "SimForge Update").State -ne "Running") { break }
            }
        } catch {}
    }

    Write-Host ""
    Write-Host "Done! SimForge $version is installed. Opening it now..." -ForegroundColor Green
    Write-Host "Next time, open it from the SimForge icon on your Desktop."
    Write-Host "(If Windows says 'Windows protected your PC', click More info, then Run anyway.)"
    Write-Host ""
    Write-Host "Last step (required): check that automatic updates are on. Run:" -ForegroundColor Yellow
    Write-Host '    Get-ScheduledTaskInfo -TaskName "SimForge Update"'
    Write-Host "You should see 'LastTaskResult : 0' and a date next to 'NextRunTime'."
    Write-Host ""
    Start-Process $app -WorkingDirectory (Split-Path $app)
}
