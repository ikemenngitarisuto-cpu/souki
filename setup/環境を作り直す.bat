@echo off
setlocal
rem ===================================================================
rem  Souki v2 - rebuild the private Python runtime from scratch.
rem  Deletes runtime\uv, runtime\python, runtime\venv, runtime\cache
rem  and runs setup again. Your ledger (data\), character names and
rem  downloaded models (models\) are NOT touched.
rem ===================================================================
cd /d "%~dp0.."
set "RT=%CD%\runtime"
echo.
echo  This deletes and rebuilds:
echo     %RT%
echo  (the ledger, character names and models are kept)
echo.
echo  Close the Souki window first if it is running.
echo.
set "ANS="
set /p ANS=  Type Y and press Enter to continue: 
if /i not "%ANS%"=="Y" goto :cancel

if exist "%RT%\venv" rmdir /s /q "%RT%\venv"
if exist "%RT%\venv" goto :locked
if exist "%RT%\python" rmdir /s /q "%RT%\python"
if exist "%RT%\cache" rmdir /s /q "%RT%\cache"
if exist "%RT%\uv" rmdir /s /q "%RT%\uv"
if exist "%RT%\setup_ok.txt" del /q "%RT%\setup_ok.txt"
if exist "%RT%\env.json" del /q "%RT%\env.json"
if exist "%RT%\metric.json" del /q "%RT%\metric.json"
call "%~dp0setup.bat"
exit /b

:locked
echo.
echo  [ERROR] runtime\venv is in use. Close the Souki window and try again.
echo.
pause
exit /b 1

:cancel
echo  Cancelled.
pause
exit /b 0
