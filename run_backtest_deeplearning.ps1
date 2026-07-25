# Run backtest in deeplearning
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
conda activate deeplearning
python run_backtest.py @args
