@echo off
REM Build script for TVAC Thermal Analyzer native engine (Windows)
REM 
REM Requires MinGW-w64 with GCC or Visual Studio with cl.exe
REM For OpenMP support, use GCC from MinGW-w64

echo ===================================
echo TVAC Thermal Engine Build Script
echo ===================================
echo.

cd /d "%~dp0"

REM Try GCC first (MinGW)
where gcc >nul 2>&1
if %errorlevel% equ 0 (
    echo Compiler: GCC (MinGW)
    echo.
    echo Building with OpenMP support...
    gcc -O3 -fopenmp -shared -o thermal_engine.dll thermal_engine.c -lm
    
    if exist thermal_engine.dll (
        echo.
        echo Build successful!
        echo Output: %CD%\thermal_engine.dll
        dir thermal_engine.dll
    ) else (
        echo Build failed!
        exit /b 1
    )
    goto :end
)

REM Try MSVC
where cl >nul 2>&1
if %errorlevel% equ 0 (
    echo Compiler: MSVC
    echo.
    echo Building with OpenMP support...
    cl /O2 /openmp /LD thermal_engine.c /Fe:thermal_engine.dll
    
    if exist thermal_engine.dll (
        echo.
        echo Build successful!
        echo Output: %CD%\thermal_engine.dll
        dir thermal_engine.dll
    ) else (
        echo Build failed!
        exit /b 1
    )
    goto :end
)

echo Error: No C compiler found!
echo.
echo Please install one of:
echo   - MinGW-w64 (recommended): https://www.mingw-w64.org/
echo   - Visual Studio Build Tools: https://visualstudio.microsoft.com/
echo.
exit /b 1

:end
echo.
echo The native engine will be automatically loaded by the plugin.
pause
