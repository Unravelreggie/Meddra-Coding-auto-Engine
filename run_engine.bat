@echo off
cd /d "E:\PV_Pf\projects\LLT_engine"
echo Activiting llt_engine environment...
call conda activate llt_engine
if errorlevel 1 (
    echo [ERROR] Conda environment 'llt_engine' not found or conda not in PATH.
    echo Please create it: conda env update -f environment.yaml
    pause
    exit /b
)

echo Starting LLT Engine Server on Port 9005...
python -m llt_engine.server
pause
