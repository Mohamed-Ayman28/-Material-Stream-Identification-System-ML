@echo off
REM Complete retraining workflow to fix misclassification issues

echo ========================================
echo FIX MISCLASSIFICATION - RETRAIN SYSTEM
echo ========================================
echo.
echo This will:
echo 1. Collect 30 images per class with YOUR camera
echo 2. Extract features from YOUR images
echo 3. Retrain models on YOUR data
echo 4. Test the improved models
echo.
echo This fixes the problem where paper is detected as plastic!
echo.
pause

echo.
echo ========================================
echo STEP 1: Collect Custom Images
echo ========================================
echo.
echo You will collect images for each material:
echo   - paper (lined notebooks, printer paper, etc.)
echo   - cardboard
echo   - plastic
echo   - metal
echo   - glass
echo   - trash
echo.
echo Instructions:
echo   - Hold material in front of camera
echo   - Press SPACE to capture each image
echo   - Collect 30 images per material
echo   - Move and rotate the material for variety
echo.
pause

python src\collect_custom_data.py --images-per-class 30 --output-dir custom_dataset

if errorlevel 1 (
    echo Error collecting images!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 2: Extract Features
echo ========================================
echo.

python src\main.py --mode extract --input-dir custom_dataset --features-file custom_features.npz

if errorlevel 1 (
    echo Error extracting features!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 3: Train New Models
echo ========================================
echo.

python src\train_knn_svm.py --features custom_features.npz --output-dir models_custom

if errorlevel 1 (
    echo Error training models!
    pause
    exit /b 1
)

echo.
echo ========================================
echo STEP 4: Test New Models
echo ========================================
echo.
echo Testing on your paper image...
echo.

python src\predict.py --image "my_images\CardBoard.jpg" --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json

echo.
echo ========================================
echo RETRAINING COMPLETE!
echo ========================================
echo.
echo New models saved in: models_custom\
echo.
echo To use new models with camera:
echo   python src\deploy.py --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json
echo.
echo If accuracy is good, copy models_custom to models:
echo   xcopy /Y models_custom\*.pkl models\
echo   xcopy /Y models_custom\*.json models\
echo.
pause
