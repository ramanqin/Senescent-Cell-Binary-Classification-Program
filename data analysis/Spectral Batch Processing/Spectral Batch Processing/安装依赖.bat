@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m pip install -r requirements.txt
echo.
echo 依赖安装完成，可以双击“启动软件.bat”。
pause
