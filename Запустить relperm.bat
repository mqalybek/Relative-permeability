@echo off
rem Double-click to open the relperm editor (relative permeability and Pc).
cd /d "%~dp0"
python -m relperm app
if errorlevel 1 (
  echo.
  echo Could not start relperm. Install it once from this folder:  pip install -e ".[scal]"
  pause
)
