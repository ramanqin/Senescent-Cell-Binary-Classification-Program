@echo off
cd /d "%~dp0"
python __main__.py --group0-size 50 --group1-size 50 --unknown-size 0 --strategy balanced --seed 20260813
if errorlevel 1 pause
