@echo off
REM Deploy with KNN model (recommended - better for your images)
python src\deploy.py --model models\knn_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json --confidence 0.50
