@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_windows.ps1"
if errorlevel 1 (
  echo.
  echo Installation did not finish. See the message above.
  pause
)
endlocal
