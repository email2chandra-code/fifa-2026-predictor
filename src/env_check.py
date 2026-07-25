import sys

def require_torch():
    try:
        import torch
        _=torch.tensor([1.0])
        return True
    except (ImportError, RuntimeError) as exc:
        print("PyTorch is required for training/backtest.")
        print("  Python:", sys.executable)
        print("  Error:", exc)
        print()
        print("Fix: conda activate deeplearning")
        print("     python run_backtest.py")
        print("NumPy issue? Run fix_deeplearning_env.ps1")
        sys.exit(1)
