@echo off
echo ============================================================
echo MATERIALSTREAM - ENHANCED CAMERA DEPLOYMENT
echo ============================================================
echo.
echo Starting camera with enhanced ensemble model...
echo Press 'q' to quit
echo.

"C:/Users/moham/Desktop/MaterialStream Identification System/.venv/Scripts/python.exe" src\deploy.py --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced.pkl --class-map models\class_map_enhanced.json --confidence 0.35

pause
