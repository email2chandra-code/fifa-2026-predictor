@echo off
call conda activate deeplearning
if errorlevel 1 exit /b 1
cd /d "C:\Users\sas_c\OneDrive\Documents\2026\AI Engineer\Course\MS AI Machine Learning\fifa_2026_predictor"
python run_backtest.py %*
