@echo off
chcp 65001 >nul
cd /d "%~dp0"
python app.py
if errorlevel 1 (
  echo.
  echo 软件启动失败。请先双击“安装依赖.bat”，或检查Python是否已加入PATH。
  pause
)
