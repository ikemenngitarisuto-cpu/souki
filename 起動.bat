@echo off
setlocal
cd /d "%~dp0"
title Souki
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0runtime\venv\Scripts\python.exe"
set "MARK=souki-runtime-2"

rem --- The private runtime (made by setup\setup.bat) -----------------
if not exist "%PY%" goto :setup
findstr /x /c:"%MARK%" "%~dp0runtime\setup_ok.txt" >nul 2>&1
if errorlevel 1 goto :setup
goto :run

:setup
echo.
echo  First start of this version: preparing the private Python runtime.
echo  (needs internet, about 3-10 minutes, only once)
call "%~dp0setup\setup.bat" nopause
if not exist "%PY%" goto :fallback
findstr /x /c:"%MARK%" "%~dp0runtime\setup_ok.txt" >nul 2>&1
if errorlevel 1 goto :fallback

:run
chcp 65001 >nul
"%PY%" souki.py serve
goto :end

rem --- Setup failed: start the ledger with any Python that has Pillow ---
:fallback
echo.
echo  [WARN] Setup did not finish. Starting the ledger with the system Python.
echo         The character screen will not work until setup succeeds:
echo         run setup\setup.bat again later.
echo.
set "PYTHON="
python -c "import PIL" >nul 2>&1
if not errorlevel 1 set "PYTHON=python"
if defined PYTHON goto :run_sys
py -3 -c "import PIL" >nul 2>&1
if not errorlevel 1 set "PYTHON=py -3"
if defined PYTHON goto :run_sys
echo  [ERROR] No usable Python was found either.
goto :end

:run_sys
chcp 65001 >nul
%PYTHON% souki.py serve

:end
echo.
pause
