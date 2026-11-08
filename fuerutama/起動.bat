@echo off
setlocal
rem ---------------------------------------------------------------
rem  Fuerutama: opens fuerutama.html in its own Edge window.
rem  (falls back to the default browser when Edge is not found)
rem ---------------------------------------------------------------
set "FT_HTML=%~dp0fuerutama.html"
set "FT_PROFILE=%~dp0data\edge"
if not exist "%FT_HTML%" goto missing

set "EDGE="
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined EDGE if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined EDGE if exist "%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe" set "EDGE=%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"
if not defined EDGE goto plain

rem --- The page address (file:///...) with non-ASCII letters escaped ---
set "FT_URL="
for /f "usebackq delims=" %%U in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Uri $env:FT_HTML).AbsoluteUri"`) do set "FT_URL=%%U"
if not defined FT_URL goto plain

rem --- Window size only on the very first start (Edge remembers it after that) ---
set "FT_SIZE="
if not exist "%FT_PROFILE%" set "FT_SIZE=--window-size=760,980"

start "" "%EDGE%" --app="%FT_URL%" --user-data-dir="%FT_PROFILE%" --no-first-run --no-default-browser-check --autoplay-policy=no-user-gesture-required %FT_SIZE%
exit /b 0

:plain
start "" "%FT_HTML%"
exit /b 0

:missing
echo.
echo  [ERROR] fuerutama.html was not found next to this file.
echo          Keep all files of the zip in the same folder.
echo.
pause
exit /b 1
