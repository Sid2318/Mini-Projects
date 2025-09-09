@echo off
echo Setting up a compatible MediaPipe environment...

:: Create new virtual environment
echo Creating virtual environment...
python -m venv mp_env_compatible

:: Activate the environment
echo Activating environment...
call .\mp_env_compatible\Scripts\activate

:: Install compatible packages
echo Installing compatible packages...
pip install -r requirements_compatible.txt

echo.
echo Environment setup complete!
echo Use ".\mp_env_compatible\Scripts\activate" to activate the environment
echo.
pause
