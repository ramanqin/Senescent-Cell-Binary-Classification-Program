@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import matplotlib, pandas" >nul 2>&1
    if not errorlevel 1 (
        start "" ".venv\Scripts\pythonw.exe" "spectrum_batch_viewer.py"
        goto :done
    )
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    py -3 "spectrum_batch_viewer.py"
) else (
    python "spectrum_batch_viewer.py"
)

:done

endlocal
