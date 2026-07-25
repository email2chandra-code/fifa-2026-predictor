# Run 2026 production forecast in the deeplearning conda env.
# Usage: .\run_production.ps1 --iteration 5

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
conda run -n deeplearning --no-capture-output python run_production.py @args
