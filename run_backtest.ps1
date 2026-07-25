# Run walk-forward backtest in the deeplearning conda env.
# Usage: .\run_backtest.ps1 --year 2018

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
conda run -n deeplearning --no-capture-output python run_backtest.py @args
