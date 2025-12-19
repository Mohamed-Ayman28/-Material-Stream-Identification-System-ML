@echo off
REM Real-time Material Classification with Live Camera Feed

echo ========================================
echo Material Stream Identification System
echo Real-Time Camera Classification
echo ========================================
echo.
echo Model Options:
echo   1. SVM (Recommended - 96.67%% accuracy, fast)
echo   2. Ensemble (Best - 98.33%% accuracy)
echo   3. KNN (90.83%% accuracy)
echo.
set /p choice="Select model (1/2/3) [default: 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="2" goto ensemble
if "%choice%"=="3" goto knn

:svm
echo.
echo Starting with SVM model (96.67%% accuracy)...
python src\deploy.py --model models\svm_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map.json --confidence 0.45
goto end

:ensemble
echo.
echo Starting with Ensemble model (98.33%% accuracy)...
python src\deploy.py --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map.json --confidence 0.30
goto end

:knn
echo.
echo Starting with KNN model (90.83%% accuracy)...
python src\deploy.py --model models\knn_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map.json --confidence 0.50
goto end

:end
echo.
echo ========================================
echo Camera closed
echo ========================================
pause
