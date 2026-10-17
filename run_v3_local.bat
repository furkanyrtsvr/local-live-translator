@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.13 app\translator_v3_local.py %*
if errorlevel 1 pause
