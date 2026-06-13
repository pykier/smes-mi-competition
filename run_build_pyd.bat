@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHASHSEED=0"
call :resolve_python || exit /b 1

set "TARGET_ROOT=%~1"
if "%TARGET_ROOT%"=="" (
    set "TARGET_ROOT=%CD%"
)
echo [run_build_pyd] cwd=%CD%
echo [run_build_pyd] target_root=%TARGET_ROOT%
echo [run_build_pyd] python=%PYTHON_EXE%
echo [run_build_pyd] required release assets:
echo [run_build_pyd]   app\Collector\Collector\receiver\virtual_receiver\data\**
echo [run_build_pyd]   app\Algorithm\Algorithm\method\model_artifacts\baseline_example\baseline_EEGNet.py
echo [run_build_pyd]   app\Algorithm\Algorithm\method\model_artifacts\baseline_example\baseline_preprocessing.py
echo [run_build_pyd]   app\ProcessHub\ApplicationFramework\config\RuntimeStageCoordinatorLauncherConfig.yml
"%PYTHON_EXE%" tests\support\build_pyd_app.py --target-root "%TARGET_ROOT%"
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if not "%BUILD_EXIT_CODE%"=="0" (
    echo [run_build_pyd] build failed with exit code %BUILD_EXIT_CODE%
) else (
    echo [run_build_pyd] build passed
)
pause
endlocal & exit /b %BUILD_EXIT_CODE%

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
    echo [run_build_pyd] Python not found.
    pause
    exit /b 1
)
exit /b 0
