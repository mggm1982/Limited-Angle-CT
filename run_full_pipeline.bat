@echo off
cd /d "%~dp0"
python run_local_cuda.py %*
exit /b %errorlevel%
