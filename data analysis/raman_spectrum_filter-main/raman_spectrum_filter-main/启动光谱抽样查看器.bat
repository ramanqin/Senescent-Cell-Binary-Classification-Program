@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\pythonw.exe" "spectrum_batch_viewer.py"
) else (
    python "spectrum_batch_viewer.py"
)

endlocal
