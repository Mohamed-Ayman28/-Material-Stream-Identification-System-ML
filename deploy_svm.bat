@echo off
REM Deploy with SVM model
python src\deploy.py --model models\svm_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json --confidence 0.50
