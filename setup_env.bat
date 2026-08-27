@echo off
chcp 65001 >nul
rem voice0 one-click installer (Windows). Double-click me, or run:  python setup_env.py
rem NOTE: echo messages below are ASCII to avoid Windows console codepage issues;
rem the Chinese guidance is printed by setup_env.py itself when it runs.

setlocal
set PYCMD=

rem 1) python on PATH (Anaconda Prompt / normal python install)
where python >nul 2>nul
if not errorlevel 1 set PYCMD=python

rem 2) py launcher (python.org installs)
if not defined PYCMD (
  where py >nul 2>nul
  if not errorlevel 1 set PYCMD=py
)

rem 3) conda-only machine (no python on PATH): derive base python from conda
if not defined PYCMD (
  where conda >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%b in ('conda info --base') do set "PYCMD=%%b\python.exe"
)

if not defined PYCMD (
  echo.
  echo [ERROR] No python found.
  echo   Open "Anaconda Prompt", cd to the voice0 folder, and run:  python setup_env.py
  pause
  exit /b 1
)

echo Using python: %PYCMD%
%PYCMD% setup_env.py %*
set EC=%ERRORLEVEL%
echo.
if "%EC%"=="0" (
  echo [DONE] Installer finished. Scroll up for usage instructions.
) else (
  echo [ERROR] setup_env.py exited with code %EC%.
  echo   Open "Anaconda Prompt", cd to the voice0 folder, and run:  python setup_env.py
  echo   to see the full error output.
)
pause
