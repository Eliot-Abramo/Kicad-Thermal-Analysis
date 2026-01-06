@echo off
setlocal enabledelayedexpansion

REM install_scipy_for_kicad.bat
REM Installs SciPy into KiCad's Python (Windows). Run as admin if possible.

set "PY="

REM Try common KiCad locations
for %%P in (
  "%ProgramFiles%\KiCad\9.0\bin\python.exe"
  "%ProgramFiles%\KiCad\8.0\bin\python.exe"
  "%ProgramFiles%\KiCad\bin\python.exe"
  "%ProgramFiles(x86)%\KiCad\9.0\bin\python.exe"
  "%ProgramFiles(x86)%\KiCad\8.0\bin\python.exe"
  "%ProgramFiles(x86)%\KiCad\bin\python.exe"
) do (
  if exist "%%~P" (
    set "PY=%%~P"
    goto :found
  )
)

:found
if "%PY%"=="" (
  echo [ERROR] Could not find KiCad's python.exe. Edit this script or install KiCad in the default folder.
  exit /b 1
)

echo [INFO] Using KiCad Python: "%PY%"
"%PY%" -c "import sys,struct; print('exe=', sys.executable); print('ver=', sys.version); print('bits=', struct.calcsize('P')*8)"

REM Ensure pip (ignore failures)
"%PY%" -m ensurepip --upgrade >nul 2>nul

echo [INFO] Upgrading pip tooling...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip tooling upgrade failed.
  exit /b 1
)

echo [INFO] Installing SciPy...
"%PY%" -m pip install --upgrade scipy
if errorlevel 1 (
  echo [WARN] SciPy install to site-packages failed (permissions?). Retrying with --user...
  "%PY%" -m pip install --upgrade --user scipy
  if errorlevel 1 (
    echo [ERROR] SciPy install failed.
    exit /b 1
  )
)

echo [INFO] Verifying import...
"%PY%" -c "import scipy; import scipy.sparse; print('SciPy OK:', scipy.__version__)"
if errorlevel 1 (
  echo [ERROR] SciPy still not importable in KiCad's Python.
  exit /b 1
)

echo [INFO] Installing reportlab...
"%PY%" -m pip install --upgrade reportlab
if errorlevel 1 (
  echo [WARN] reportlab install to site-packages failed (permissions?). Retrying with --user...
  "%PY%" -m pip install --upgrade --user reportlab
  if errorlevel 1 (
    echo [ERROR] reportlab install failed.
    exit /b 1
  )
)

echo [INFO] Verifying import...
"%PY%" -c "from reportlab.lib import colors; import reportlab.lib; print('reportlab OK:', reportlab.__version__)"
if errorlevel 1 (
  echo [ERROR] reportlab still not importable in KiCad's Python.
  exit /b 1
)

echo [INFO] Done. Restart KiCad.
exit /b 0
