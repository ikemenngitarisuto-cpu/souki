@echo off
setlocal
rem ===================================================================
rem  Souki v2 - setup of the private Python runtime (first time only)
rem
rem  Everything goes into the "runtime" folder next to souki.py:
rem    runtime\uv       uv.exe (Python manager, from GitHub or PyPI)
rem    runtime\python   a private Python 3.12
rem    runtime\venv     numpy, pillow, onnxruntime-directml (pinned)
rem  Nothing is installed into ComfyUI or into a system Python.
rem  Safe to run again at any time.
rem ===================================================================
cd /d "%~dp0.."
set "APP=%CD%"
set "RT=%APP%\runtime"
set "UVDIR=%RT%\uv"
set "UV=%RT%\uv\uv.exe"
set "PY=%RT%\venv\Scripts\python.exe"
set "UV_PYTHON_INSTALL_DIR=%RT%\python"
set "UV_CACHE_DIR=%RT%\cache"
set "UV_NO_CONFIG=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "MARK=souki-runtime-2"
set "UV_VER=0.12.18"
set "UV_ZIP_URL=https://github.com/astral-sh/uv/releases/download/0.12.18/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_SHA=cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff"
set "UV_WHL_URL=https://files.pythonhosted.org/packages/86/b3/5f33b9bad8cb875ed547ef8aa403d423a2610f23f619f687308761dd02e1/uv-0.12.18-py3-none-win_amd64.whl"
set "UV_WHL_SHA=0a3c4e415515bd4c24c7516523d9042a2957ce716423e193f8b2be9fb8773340"
if not exist "%RT%" mkdir "%RT%"
if exist "%RT%\setup_ok.txt" del /q "%RT%\setup_ok.txt"

echo.
echo  ============================================================
echo    Souki setup  (first time only, about 3-10 minutes)
echo    Folder: %RT%
echo  ============================================================
echo.

rem ---------------------------------------------------------------
rem  1. uv
rem ---------------------------------------------------------------
if exist "%UV%" goto :have_uv
echo  [1/4] Downloading uv %UV_VER% (Python manager, 18 MB)
if not exist "%UVDIR%" mkdir "%UVDIR%"
call :get_zip
if exist "%UV%" goto :have_uv
call :get_whl
if exist "%UV%" goto :have_uv
echo.
echo  [ERROR] Could not download uv from GitHub or PyPI.
echo          Check the internet connection (or a firewall / security
echo          software blocking curl.exe) and run this file again.
goto :fail

:have_uv
"%UV%" --version
if errorlevel 1 goto :bad_uv

rem ---------------------------------------------------------------
rem  2. Python 3.12 + venv
rem ---------------------------------------------------------------
echo.
echo  [2/4] Preparing a private Python 3.12
if exist "%PY%" goto :have_venv
"%UV%" venv "%RT%\venv" --python 3.12 --managed-python --clear
if exist "%PY%" goto :have_venv
echo  Download of Python failed. Trying a Python 3.12 already on this PC...
"%UV%" venv "%RT%\venv" --python 3.12 --no-managed-python --clear
if exist "%PY%" goto :have_venv
echo.
echo  [ERROR] Could not prepare Python 3.12.
echo          Check the internet connection and run this file again.
goto :fail

:have_venv
"%PY%" -c "import sys; print('  Python', sys.version.split()[0])"
if errorlevel 1 goto :bad_venv

rem ---------------------------------------------------------------
rem  3. packages (exact versions from setup\requirements.txt)
rem ---------------------------------------------------------------
echo.
echo  [3/4] Installing packages (numpy, pillow, onnxruntime-directml)
"%UV%" pip sync --python "%PY%" "%APP%\setup\requirements.txt"
if errorlevel 1 goto :bad_pip
"%PY%" -c "import numpy, PIL, onnxruntime as o; print('  numpy', numpy.__version__, '/ onnxruntime', o.__version__, o.get_available_providers())"
if errorlevel 1 goto :bad_pip
> "%RT%\setup_ok.txt" echo %MARK%

rem ---------------------------------------------------------------
rem  4. models + GPU self-test (not fatal: can be done from the UI)
rem ---------------------------------------------------------------
echo.
echo  [4/4] Downloading the character models (about 200 MB) and testing the GPU
chcp 65001 >nul
"%PY%" "%APP%\chara.py" setup
echo.
echo  ============================================================
echo    Setup finished.
echo  ============================================================
goto :end

rem ===============================================================
:get_zip
echo    from GitHub ...
call :download "%UV_ZIP_URL%" "%RT%\uv_dl.zip"
if not exist "%RT%\uv_dl.zip" exit /b 1
call :sha256 "%RT%\uv_dl.zip"
if /i not "%HASH%"=="%UV_ZIP_SHA%" goto :zip_bad
tar -xf "%RT%\uv_dl.zip" -C "%UVDIR%"
if not exist "%UV%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%RT%\uv_dl.zip' -DestinationPath '%UVDIR%' -Force"
del /q "%RT%\uv_dl.zip" >nul 2>&1
exit /b 0
:zip_bad
echo    (file check failed: %HASH%)
del /q "%RT%\uv_dl.zip" >nul 2>&1
exit /b 1

:get_whl
echo    from PyPI ...
call :download "%UV_WHL_URL%" "%RT%\uv_dl.zip"
if not exist "%RT%\uv_dl.zip" exit /b 1
call :sha256 "%RT%\uv_dl.zip"
if /i not "%HASH%"=="%UV_WHL_SHA%" goto :zip_bad
if exist "%RT%\uv_whl" rmdir /s /q "%RT%\uv_whl"
mkdir "%RT%\uv_whl"
tar -xf "%RT%\uv_dl.zip" -C "%RT%\uv_whl"
if not exist "%RT%\uv_whl\uv-%UV_VER%.data\scripts\uv.exe" powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%RT%\uv_dl.zip' -DestinationPath '%RT%\uv_whl' -Force"
copy /y "%RT%\uv_whl\uv-%UV_VER%.data\scripts\uv.exe" "%UV%" >nul
rmdir /s /q "%RT%\uv_whl" >nul 2>&1
del /q "%RT%\uv_dl.zip" >nul 2>&1
exit /b 0

:download
rem  %1 = URL, %2 = output file. curl first, PowerShell as a fallback.
if exist "%~2" del /q "%~2"
curl.exe -fL --retry 3 --connect-timeout 20 -o "%~2" "%~1"
if not errorlevel 1 exit /b 0
if exist "%~2" del /q "%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '%~1' -OutFile '%~2'"
if errorlevel 1 if exist "%~2" del /q "%~2"
exit /b 0

:sha256
rem  Sets HASH to the SHA-256 of file %1 (lower case not required).
set "HASH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile "%~1" SHA256') do if not defined HASH set "HASH=%%h"
if not defined HASH exit /b 1
set "HASH=%HASH: =%"
exit /b 0

rem ===============================================================
:bad_uv
echo.
echo  [ERROR] uv.exe does not start. Security software may have blocked it.
echo          Folder: %UVDIR%
goto :fail

:bad_venv
echo.
echo  [ERROR] The private Python does not start.
echo          Run setup\rebuild (the other .bat in this folder) to start over.
goto :fail

:bad_pip
echo.
echo  [ERROR] Installing packages failed (see the message above).
echo          Check the internet connection and run this file again.
goto :fail

:fail
echo.
echo  Setup did not finish. Nothing outside the runtime folder was changed.
if /i not "%~1"=="nopause" pause
exit /b 1

:end
if /i not "%~1"=="nopause" pause
exit /b 0
