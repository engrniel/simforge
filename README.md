# SimForge

SimForge is a MATLAB (MATrix LABoratory) IDE (Integrated Development Environment) for the classroom. Your code runs on SimForge's built-in engine, so you don't need MATLAB.

## Install

Copy **one** command, paste it, and press **Enter**. It installs everything and opens SimForge. No admin password needed.

### Windows 10/11

1. Click **Start**, type `PowerShell`, and press **Enter**.
2. Paste this and press **Enter**:

```powershell
irm https://raw.githubusercontent.com/engrniel/simforge/main/install.ps1 | iex
```

3. Wait for **Done!**. SimForge opens by itself. Next time, use the **SimForge** icon on your Desktop.

If Windows says **"Windows protected your PC"**, click **More info**, then **Run anyway**.

### Mac (M1 or newer)

1. Press **⌘ Space**, type `Terminal`, and press **Enter**.
2. Paste this and press **Enter**:

```bash
curl -fsSL https://raw.githubusercontent.com/engrniel/simforge/main/install.sh | bash
```

3. If a window asks to install **Command Line Tools**, click **Install**, then **Agree**. Leave Terminal open. It continues by itself.
4. Wait for **Done!**. SimForge opens by itself. Next time, open it from your **Applications** folder.

## Updates

SimForge updates itself in the background, including after your laptop wakes from sleep. You don't need to do anything.

To update right now, run the same install command again.

## Something went wrong?

Run the same install command again. It is safe to run any number of times: it fixes the install and updates SimForge.

---

## For teachers and IT

**What the installer does.** `install.ps1` (Windows) and `install.sh` (Mac):

1. Install Git and Python 3.9+ if they are missing, for the current user only.
2. `git clone` this repository into `%LOCALAPPDATA%\SimForge` (Windows) or `~/SimForge` (Mac).
3. Add Desktop and Start Menu shortcuts (Windows) or `~/Applications/SimForge.app` (Mac).
4. Turn on automatic updates. Windows: a scheduled task, **SimForge Update**, at sign-in, on every unlock, when the laptop wakes from sleep, and daily at noon; a missed noon run happens as soon as the laptop is awake. Mac: a LaunchAgent, `ph.nrdev.simforge.update`, at login and every hour; an hourly run missed during sleep happens on wake. Both run `update.ps1` / `update.sh`, which run `git pull`. Right after waking, the updaters wait up to about 2 minutes for Wi-Fi. Windows skips a run while SimForge is open and catches up on the next one.

**Repository layout**

- `windows/`: `SimForge.exe` and the engine's `.py` files
- `macos/SimForge.app`: the Mac app
- `install.ps1`, `install.sh`: one-command installers
- `update.ps1`, `update.sh`: background updaters
- `VERSION`: the version in this commit

**Manual commands**

| Task | Windows (PowerShell) | Mac (Terminal) |
|---|---|---|
| Update now | `git -C "$env:LOCALAPPDATA\SimForge" pull` | `git -C ~/SimForge pull` |
| Installed version | `Get-Content "$env:LOCALAPPDATA\SimForge\VERSION"` | `cat ~/SimForge/VERSION` |

On Windows, close SimForge before updating by hand. Git cannot replace `SimForge.exe` while it is open.

**Uninstall**

Windows:

```powershell
Unregister-ScheduledTask -TaskName "SimForge Update" -Confirm:$false
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\SimForge", "$env:LOCALAPPDATA\SimForge-Git" -ErrorAction SilentlyContinue
Remove-Item "$([Environment]::GetFolderPath('Desktop'))\SimForge.lnk", "$([Environment]::GetFolderPath('Programs'))\SimForge.lnk"
```

Mac:

```bash
launchctl bootout gui/$(id -u)/ph.nrdev.simforge.update
rm -f ~/Library/LaunchAgents/ph.nrdev.simforge.update.plist ~/Applications/SimForge.app
rm -rf ~/SimForge
```

**Zero-touch install.** On school-managed laptops, IT can run the same installer through MDM (Mobile Device Management). In Microsoft Intune, add `install.ps1` as a Windows platform script with **Run this script using the logged on credentials = Yes**. In Jamf Pro, run `install.sh` as the logged-in user with a Login trigger, once per user per computer. Install the Command Line Tools first on Macs.
