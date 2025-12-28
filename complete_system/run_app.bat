@echo off
echo ========================================
echo   Universal Data Science Platform
echo   Phase 6: Streamlit Frontend
echo ========================================
echo.

echo [1/2] Installing dependencies...
pip install -r requirements_frontend.txt --quiet

echo.
echo [2/2] Starting Streamlit app...
echo.
echo    App will open in your browser at:
echo    http://localhost:8501
echo.

streamlit run app.py
