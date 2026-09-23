@echo off
setlocal
cd /d "%~dp0"

where py.exe >nul 2>&1
if errorlevel 1 goto use_python

py -3 "%~dp0gui.py"
goto check_result

:use_python
python "%~dp0gui.py"

:check_result
if errorlevel 1 (
    echo.
    echo PCA-SVM GUI failed to start. See the error above.
    pause
)
endlocal
