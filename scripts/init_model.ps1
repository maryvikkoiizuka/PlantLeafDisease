<#
PowerShell helper to initialize the ML model via the Django API.
Usage:
  - From project root: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
  - Run: .\scripts\init_model.ps1
  - Or pass custom paths: .\scripts\init_model.ps1 -ModelPath 'C:\path\model.keras' -ClassIndices 'C:\path\class_indices.json'
#>

param(
    [string]$ModelPath = 'C:\PlantLeafDiseasePrediction\models\plant_disease_model_efficientnetb0.keras',
    [string]$ClassIndices = 'C:\PlantLeafDiseasePrediction\models\class_indices_efficientnetb0.json',
    [string]$ServerUrl = 'http://127.0.0.1:8000/api/initialize-model/'
)

# Optionally activate virtualenv if present
$venvActivate = Join-Path -Path $PSScriptRoot -ChildPath '..\.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment: $venvActivate"
    try {
        . $venvActivate
    } catch {
        Write-Warning "Failed to activate venv: $_"
    }
}

# Verify files exist
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found: $ModelPath"
    exit 1
}
if (-not (Test-Path $ClassIndices)) {
    Write-Error "Class indices file not found: $ClassIndices"
    exit 1
}

# Build JSON body and POST
$body = @{ model_path = $ModelPath; class_indices_path = $ClassIndices } | ConvertTo-Json

Write-Host "Posting to $ServerUrl ..."
try {
    $resp = Invoke-RestMethod -Uri $ServerUrl -Method Post -ContentType 'application/json' -Body $body -ErrorAction Stop
    Write-Host "Response:`n" ($resp | ConvertTo-Json -Depth 4)
} catch {
    Write-Error "Request failed: $_"
    exit 2
}

Write-Host "Done."
