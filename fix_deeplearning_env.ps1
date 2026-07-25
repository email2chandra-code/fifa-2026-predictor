# Fix PyTorch 2.2.2 + NumPy compatibility in deeplearning env
# Run in PowerShell: .\fix_deeplearning_env.ps1

$python = "C:\Users\sas_c\anaconda3\envs\deeplearning\python.exe"
$pip = "C:\Users\sas_c\anaconda3\envs\deeplearning\Scripts\pip.exe"

Write-Host "=== Pinning numpy 1.26.4 (required for torch 2.2.2) ==="
& $pip install "numpy==1.26.4" --force-reinstall --no-deps

Write-Host "=== Reinstalling pandas against numpy 1.26 ==="
& $pip install --force-reinstall --no-cache-dir "pandas>=2.2,<2.4"

Write-Host "=== Verification ==="
& $python -c @"
import numpy as np
import pandas as pd
import torch
print('numpy ', np.__version__)
print('pandas', pd.__version__)
print('torch ', torch.__version__)
t = torch.tensor([1.0, 2.0])
print('tensor->numpy OK:', t.numpy())
import pathlib
p = pathlib.Path(r'outputs/backtest_summary_2018.csv')
if p.exists():
    df = pd.read_csv(p)
    print('read_csv OK:', df.shape)
else:
    print('read_csv skip (run from project folder or path missing)')
"@

Write-Host "=== Done. Restart Jupyter kernel before re-running presentation.ipynb ==="
