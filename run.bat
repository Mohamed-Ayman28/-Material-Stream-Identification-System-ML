@echo off
REM Quick Start Script for Material Detection System
REM Using KNN and SVM models (No Deep Learning)

echo ========================================
echo Material Stream Identification System
echo KNN + SVM Ensemble (No Deep Learning)
echo ========================================
echo.

:menu
echo Select an option:
echo.
echo 1. Test models (verify installation)
echo 2. Predict single image
echo 3. Start camera detection (recommended)
echo 4. Train new models from features
echo 5. Extract features from dataset
echo 6. Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto test
if "%choice%"=="2" goto predict
if "%choice%"=="3" goto camera
if "%choice%"=="4" goto train
if "%choice%"=="5" goto extract
if "%choice%"=="6" goto end

echo Invalid choice!
goto menu

:test
echo.
echo Testing models...
python test_models.py
pause
goto menu

:predict
echo.
set /p imgpath="Enter image path: "
python src\predict.py --image "%imgpath%" --model models\ensemble_model.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json
pause
goto menu

:camera
echo.
echo Starting camera detection...
echo Press Q to quit, S to save screenshot
echo.
python src\deploy.py --model models\ensemble_model.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json --confidence 0.65
pause
goto menu

:train
echo.
echo Training new models...
echo This will use features_optimized.npz
echo.
python src\train_knn_svm.py --features features_optimized.npz --output-dir models
pause
goto menu

:extract
echo.
echo Extracting features from dataset...
python src\main.py --mode extract --input-dir dataset --features-file features_optimized.npz
pause
goto menu

:end
echo.
echo Goodbye!
