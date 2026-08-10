@echo off
REM Double-click this to open the course with videos working.
REM Leave the window open while you read, and close it when you are done.
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (py -3 start.py) || (python start.py)
if errorlevel 1 (
  echo.
  echo Could not start Python. Install it from python.org, then try again.
  echo.
  pause
)
