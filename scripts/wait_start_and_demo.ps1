# scripts/wait_start_and_demo.ps1
#
# Keeps trying to start the A100 VM until Azure has capacity, then runs the
# v6 demo on it, copies the demo log back to this laptop, and (by default)
# deallocates the VM so it stops billing.
#
# One-time setup on the laptop:
#   winget install -e --id Microsoft.AzureCLI     (then open a new PowerShell)
#   az login
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts\wait_start_and_demo.ps1

$VM_NAME = "NCAD"
$RESOURCE_GROUP = "NCAD_group"
$VM_USER = "azureuser"
$SSH_KEY = Join-Path $HOME ".ssh\id_ed25519"
$RETRY_SECONDS = 180
$DEALLOCATE_WHEN_DONE = $true
$LOCAL_LOG = Join-Path $PSScriptRoot "..\v6_demo.log"

function Log($msg) { Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" }

$azPath = (Get-Command az -ErrorAction SilentlyContinue).Source
if (-Not $azPath) { $azPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }
if (-Not (Test-Path $azPath)) {
    Log "Azure CLI not found. Install it with: winget install -e --id Microsoft.AzureCLI"
    Log "Then open a new PowerShell window, run 'az login', and start this script again."
    exit 1
}
& $azPath account show -o none 2>$null
if ($LASTEXITCODE -ne 0) { Log "Not logged in to Azure. Run 'az login' first."; exit 1 }

# Passwordless SSH: create a key on this laptop if there isn't one. It gets
# installed on the VM through Azure (step 2), so no password prompt is needed.
if (-Not (Test-Path $SSH_KEY)) {
    New-Item -ItemType Directory -Force (Split-Path $SSH_KEY) | Out-Null
    ssh-keygen -t ed25519 -q -N '""' -f $SSH_KEY
}
$PUBKEY = (Get-Content "$SSH_KEY.pub" -Raw).Trim()

$sshArgs = @("-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
             "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes", "-i", $SSH_KEY)

# --- 1. wait for capacity -----------------------------------------------------
Log "Capacity hunter started for $VM_NAME (retry every $RETRY_SECONDS s)"
$attempt = 0
while ($true) {
    $attempt++
    Log "Start attempt $attempt..."
    $output = & $azPath vm start --name $VM_NAME --resource-group $RESOURCE_GROUP 2>&1
    if ($LASTEXITCODE -eq 0) { Log "VM started."; break }
    $text = ($output | Out-String).Trim()
    Log ("Not started: " + $text.Split("`n")[0])
    if ($text -match "az login|expired|AADSTS") {
        Log "Azure CLI login has expired. Run 'az login' and start this script again."
        exit 1
    }
    Start-Sleep -Seconds $RETRY_SECONDS
}

# The public IP can change after a deallocate/start cycle, so look it up.
$VM_IP = (& $azPath vm show -d --name $VM_NAME --resource-group $RESOURCE_GROUP --query publicIps -o tsv).Trim()
Log "Public IP: $VM_IP"
$target = "$VM_USER@$VM_IP"

# --- 2. install this laptop's key on the VM, then wait for SSH -----------------
Log "Installing SSH key on the VM via Azure run-command..."
$keyScript = "u=/home/$VM_USER; mkdir -p `$u/.ssh; touch `$u/.ssh/authorized_keys; " +
             "grep -qF '$PUBKEY' `$u/.ssh/authorized_keys || echo '$PUBKEY' >> `$u/.ssh/authorized_keys; " +
             "chown -R ${VM_USER}:${VM_USER} `$u/.ssh; chmod 700 `$u/.ssh; chmod 600 `$u/.ssh/authorized_keys"
& $azPath vm run-command invoke --name $VM_NAME --resource-group $RESOURCE_GROUP `
    --command-id RunShellScript --scripts $keyScript -o none
if ($LASTEXITCODE -ne 0) { Log "Key install failed; SSH below will not connect without a password." }

while ($true) {
    ssh @sshArgs $target "echo ok" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 10
}
Log "SSH is up."

# --- 3. launch the demo detached from SSH, so a dropped connection can't kill it
$remote = @'
cd ~/Formal-Logic-8B
source thrm_env/bin/activate
git pull --ff-only
nvidia-smi -L
rm -f v6_demo.log
setsid nohup bash -c 'python scripts/thrm_demo.py --adapter ./thrm_v6 --attempts 5; echo "DEMO_EXIT=$?"' > v6_demo.log 2>&1 < /dev/null &
echo launched
'@
$remote = $remote -replace "`r", ""
$remote | ssh @sshArgs $target "bash -s"
Log "Demo launched. Waiting for it to finish..."

# --- 4. wait for the demo to finish, then copy the log back ---------------------
while ($true) {
    Start-Sleep -Seconds 30
    $done = ssh @sshArgs $target "grep -c DEMO_EXIT= ~/Formal-Logic-8B/v6_demo.log 2>/dev/null" 2>$null
    if ("$done".Trim() -match "^[1-9]") { break }
}
scp @sshArgs "${target}:~/Formal-Logic-8B/v6_demo.log" $LOCAL_LOG | Out-Null
Log "Demo finished. Log copied to $LOCAL_LOG"
Get-Content $LOCAL_LOG -Tail 25

# --- 5. stop billing ------------------------------------------------------------
if ($DEALLOCATE_WHEN_DONE) {
    Log "Deallocating $VM_NAME..."
    & $azPath vm deallocate --name $VM_NAME --resource-group $RESOURCE_GROUP -o none
    Log "VM deallocated."
}

[System.Console]::Beep(1000, 800)
$wshell = New-Object -ComObject Wscript.Shell
$wshell.Popup("v6 demo finished. Log saved to v6_demo.log", 0, "Azure VM", 0x40) | Out-Null
