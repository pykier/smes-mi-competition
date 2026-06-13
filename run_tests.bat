@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHASHSEED=0"
call :resolve_python || exit /b 1

echo [run_tests] =========================================
echo [run_tests] mode=workspace unittest discover
echo [run_tests] cwd=%CD%
echo [run_tests] python=%PYTHON_EXE%
if /I "%PYTHON_EXE%"=="python" (
    where python
)
echo [run_tests] discovered test files:
for /r tests %%F in (test_*.py) do (
    echo [run_tests]   %%~dpnxF
)
echo [run_tests] -----------------------------------------
echo [run_tests] running: python -m unittest discover -v -s tests -p test_*.py
"%PYTHON_EXE%" -m unittest discover -v -s tests -p "test_*.py"
set "TEST_EXIT_CODE=%ERRORLEVEL%"
echo [run_tests] =========================================
if not "%TEST_EXIT_CODE%"=="0" (
    echo [run_tests] tests failed with exit code %TEST_EXIT_CODE%
) else (
    echo [run_tests] tests passed
)
pause
endlocal & exit /b %TEST_EXIT_CODE%

:resolve_python
if defined BCI_PYTHON_EXE if exist "%BCI_PYTHON_EXE%" (
    set "PYTHON_EXE=%BCI_PYTHON_EXE%"
    exit /b 0
)
set "PYTHON_EXE="
if exist "%CD%\Python310\python.exe" set "PYTHON_EXE=%CD%\Python310\python.exe"
if not defined PYTHON_EXE if exist "D:\software\anaconda\envs\world_robot_env\python.exe" set "PYTHON_EXE=D:\software\anaconda\envs\world_robot_env\python.exe"
if not defined PYTHON_EXE if exist "D:\anaconda3\envs\BCI_competation_2026\python.exe" set "PYTHON_EXE=D:\anaconda3\envs\BCI_competation_2026\python.exe"
if not defined PYTHON_EXE if exist "D:\anaconda3\envs\BCI_competition_2026\python.exe" set "PYTHON_EXE=D:\anaconda3\envs\BCI_competition_2026\python.exe"
if not defined PYTHON_EXE if exist "D:\software\anaconda\python.exe" set "PYTHON_EXE=D:\software\anaconda\python.exe"
if not defined PYTHON_EXE for %%I in (python.exe) do set "PYTHON_EXE=%%~$PATH:I"
if not defined PYTHON_EXE (
    echo [run_tests] Python not found.
    pause
    exit /b 1
)
exit /b 0
