@echo off
echo ============================================================
echo TESTING CARDBOARD WITH ENHANCED MODEL
echo ============================================================
echo.

"C:/Users/moham/Desktop/MaterialStream Identification System/.venv/Scripts/python.exe" src\predict.py --image "my_images\CardBoard.jpg" --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced.pkl --class-map models\class_map_enhanced.json

echo.
echo ============================================================
echo Test complete!
echo ============================================================
pause
