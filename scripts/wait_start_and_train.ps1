$VM_NAME = "NCAD"
$RESOURCE_GROUP = "NCAD_group"
$VM_IP = "20.97.226.119"
$VM_USER = "azureuser"
$SSH_KEY = "C:\Users\taban\Downloads\NCAD_key.pem"

$azPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
if (-Not (Test-Path $azPath)) { $azPath = "C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }

Write-Host "[$(Get-Date)] Starting Azure capacity hunter for $VM_NAME..."

while ($true) {
    Write-Host "[$(Get-Date)] Attempting to start VM..."
    $output = & $azPath vm start --name $VM_NAME --resource-group $RESOURCE_GROUP 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date)] VM start command accepted. Waiting for boot sequence..."
        break
    }
    
    Write-Host "[$(Get-Date)] Capacity error or start failed. Retrying in 3 minutes..."
    Start-Sleep -Seconds 180
}

Write-Host "[$(Get-Date)] Waiting for SSH daemon to become reachable at $VM_IP..."

while ($true) {
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i $SSH_KEY "$VM_USER@$VM_IP" "echo ok" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date)] SSH connected. Syncing git and starting pipeline..."
        
        $remoteScript = @"
cd ~/Formal-Logic-8B
source thrm_env/bin/activate

git stash
git pull

nvidia-smi

if [ ! -f thrm_training_v2.jsonl ]; then
    python reformat_dataset.py
fi

python -c "
import json
with open('thrm_training_v2.jsonl') as f:
    row = json.loads(f.readline())
text = row['text']
assert '<|begin_of_text|>' in text, 'WRONG TEMPLATE - missing begin_of_text'
assert '<|eot_id|>' in text, 'WRONG TEMPLATE - missing eot_id'
assert '<|system|>' not in text, 'WRONG TEMPLATE - old Llama2 tokens present'
print('Dataset format OK')
print('Preview:', text[:300])
"

tmux new-session -d -s training "python scripts/train_v4.py 2>&1 | tee logs/train_v4.log; echo TRAINING_DONE >> logs/train_v4.log"

echo "Training started in tmux session."
"@

        $remoteScript | ssh -i $SSH_KEY "$VM_USER@$VM_IP" "bash"
        
        [System.Console]::Beep(1000, 1000)
        $wshell = New-Object -ComObject Wscript.Shell
        $wshell.Popup("GPU secured and training has started", 0, "Azure VM", 0x40) | Out-Null
        
        Write-Host "[$(Get-Date)] Complete. Pipeline running."
        break
    }
    
    Start-Sleep -Seconds 10
}