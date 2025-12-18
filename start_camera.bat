@echo off
REM Simple deployment with KNN model (works better for your images)

echo Starting Material Detection...
echo Using KNN model (better accuracy for your images)
echo.
echo Controls:
echo   Q - Quit
echo   S - Save screenshot
echo.

python src\deploy.py --model models\knn_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json --confidence 0.50

pause
