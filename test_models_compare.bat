@echo off
echo ============================================================
echo MODEL COMPARISON TEST
echo Testing KNN, SVM, and Ensemble models
echo ============================================================
echo.

if "%1"=="" (
    echo Usage: test_models.bat ^<image_path^>
    echo.
    echo Examples:
    echo   test_models.bat my_images\CardBoard.jpg
    echo   test_models.bat my_images\Fork.jpg
    echo   test_models.bat my_images\PlasticBottel.jpg
    echo.
    pause
    exit /b 1
)

"C:/Users/moham/Desktop/MaterialStream Identification System/.venv/Scripts/python.exe" src\compare_models.py --image "%1"

echo.
echo ============================================================
pause
