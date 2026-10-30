@echo off
REM One-click setup and launch for Windows. Run from the project folder.
if not exist venv (
    python -m venv venv || (echo Python not found. Install Python 3.10+ and tick "Add Python to PATH". & pause & exit /b 1)
)
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Dependency install failed. & pause & exit /b 1)
python train_model.py || (echo Training failed. & pause & exit /b 1)
streamlit run app.py
