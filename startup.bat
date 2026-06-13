@echo off
setlocal
for %%I in ("%~dp0.") do set "originalPath=%%~fI"
set "BCI_APP_ROOT=%originalPath%\app"
set "RUNTIME_STAGE_LAUNCHER_CONFIG=%BCI_APP_ROOT%\ProcessHub\ApplicationFramework\config\RuntimeStageCoordinatorLauncherConfig.yml"
call :resolve_python || exit /b 1

if not exist "%pythonExe%" (
    echo [startup] Python not found: %pythonExe%
    pause
    exit /b 1
)

if not exist "%BCI_APP_ROOT%" (
    echo [startup] app root not found: %BCI_APP_ROOT%
    pause
    exit /b 1
)

if not exist "%RUNTIME_STAGE_LAUNCHER_CONFIG%" (
    echo [startup] runtime stage launcher config not found: %RUNTIME_STAGE_LAUNCHER_CONFIG%
    pause
    exit /b 1
)

if not exist "%originalPath%\proceed\centrol\centrol.jar" (
    echo [startup] centrol.jar not found: %originalPath%\proceed\centrol\centrol.jar
    pause
    exit /b 1
)

if not exist "%originalPath%\proceed\collector\collector.jar" (
    echo [startup] collector.jar not found: %originalPath%\proceed\collector\collector.jar
    pause
    exit /b 1
)

if not exist "%originalPath%\proceed\task\task.jar" (
    echo [startup] task.jar not found: %originalPath%\proceed\task\task.jar
    pause
    exit /b 1
)

if not exist "%BCI_APP_ROOT%\Algorithm\Algorithm\log" mkdir "%BCI_APP_ROOT%\Algorithm\Algorithm\log"
if not exist "%BCI_APP_ROOT%\CentralController\ApplicationFramework\log" mkdir "%BCI_APP_ROOT%\CentralController\ApplicationFramework\log"
if not exist "%BCI_APP_ROOT%\CentralController\CentralController\log" mkdir "%BCI_APP_ROOT%\CentralController\CentralController\log"
if not exist "%BCI_APP_ROOT%\Collector\ApplicationFramework\log" mkdir "%BCI_APP_ROOT%\Collector\ApplicationFramework\log"
if not exist "%BCI_APP_ROOT%\Collector\Collector\log" mkdir "%BCI_APP_ROOT%\Collector\Collector\log"
if not exist "%BCI_APP_ROOT%\ProcessHub\ApplicationFramework\log" mkdir "%BCI_APP_ROOT%\ProcessHub\ApplicationFramework\log"
if not exist "%BCI_APP_ROOT%\ProcessHub\ProcessHub\log" mkdir "%BCI_APP_ROOT%\ProcessHub\ProcessHub\log"

echo [startup] Launching Central Java Controller
start "[BCI] Central Java Controller" /D "%originalPath%\proceed\centrol" cmd /k "title [BCI] Central Java Controller && java -jar centrol.jar"
timeout /t 15 /nobreak

echo [startup] Launching CentralController Python
start "[BCI] CentralController Python" /D "%BCI_APP_ROOT%\CentralController" cmd /k "title [BCI] CentralController Python && ""%pythonExe%"" -m ApplicationFramework.main"

set "LAUNCHER_CONFIG_PATH=%RUNTIME_STAGE_LAUNCHER_CONFIG%"
echo [startup] Launching RuntimeStageCoordinator Python
start "[BCI] RuntimeStageCoordinator Python" /D "%BCI_APP_ROOT%\ProcessHub" cmd /k "title [BCI] RuntimeStageCoordinator Python && ""%pythonExe%"" -m ApplicationFramework.main"
set "LAUNCHER_CONFIG_PATH="

echo [startup] Launching Collector Java Bridge
start "[BCI] Collector Java Bridge" /D "%originalPath%\proceed\collector" cmd /k "title [BCI] Collector Java Bridge && java -jar collector.jar"

echo [startup] Launching Task Java Bridge
start "[BCI] Task Java Bridge" /D "%originalPath%\proceed\task" cmd /k "title [BCI] Task Java Bridge && java -jar task.jar"

echo [startup] Launching Algorithm Python
start "[BCI] Algorithm Python" /D "%BCI_APP_ROOT%\Algorithm" cmd /k "title [BCI] Algorithm Python && ""%pythonExe%"" -m Algorithm.main"
timeout /t 15 /nobreak

echo [startup] Launching Collector Python
start "[BCI] Collector Python" /D "%BCI_APP_ROOT%\Collector" cmd /k "title [BCI] Collector Python && ""%pythonExe%"" -m ApplicationFramework.main"

echo [startup] Launching ProcessHub Python
start "[BCI] ProcessHub Python" /D "%BCI_APP_ROOT%\ProcessHub" cmd /k "title [BCI] ProcessHub Python && ""%pythonExe%"" -m ApplicationFramework.main"

endlocal
exit /b 0

:resolve_python
if defined BCI_PYTHON_EXE if exist "%BCI_PYTHON_EXE%" (
    set "pythonExe=%BCI_PYTHON_EXE%"
    exit /b 0
)
set "pythonExe="
if exist "%originalPath%\Python310\python.exe" set "pythonExe=%originalPath%\Python310\python.exe"
if not defined pythonExe if exist "D:\software\anaconda\envs\world_robot_env\python.exe" set "pythonExe=D:\software\anaconda\envs\world_robot_env\python.exe"
if not defined pythonExe if exist "D:\anaconda3\envs\BCI_competation_2026\python.exe" set "pythonExe=D:\anaconda3\envs\BCI_competation_2026\python.exe"
if not defined pythonExe if exist "D:\anaconda3\envs\BCI_competition_2026\python.exe" set "pythonExe=D:\anaconda3\envs\BCI_competition_2026\python.exe"
if not defined pythonExe if exist "D:\software\anaconda\python.exe" set "pythonExe=D:\software\anaconda\python.exe"
if not defined pythonExe for %%I in (python.exe) do set "pythonExe=%%~$PATH:I"
if not defined pythonExe (
    echo [startup] Python not found.
    pause
    exit /b 1
)
exit /b 0
