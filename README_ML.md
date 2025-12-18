# Material Stream Identification System
## KNN and SVM Classification (No Deep Learning)

This system uses traditional machine learning (KNN and SVM) for material classification without deep learning.

## Files Overview

### Training
- `src/train_knn_svm.py` - Main training script for KNN and SVM models with ensemble
- `src/feature_extraction.py` - Feature extraction pipeline
- `src/main.py` - Feature extraction from dataset

### Prediction
- `src/predict.py` - Single image prediction
- `src/deploy.py` - Real-time webcam detection

### Models (in models/ directory)
- `svm_model.pkl` - Optimized SVM classifier
- `knn_model.pkl` - Optimized KNN classifier  
- `ensemble_knn_svm.pkl` - Ensemble combining both models
- `scaler.pkl` - Feature scaler
- `class_map.json` - Class index to name mapping

## Usage

### 1. Extract Features from Dataset

```bash
python src\main.py --mode extract --input-dir dataset --features-file features.npz
```

### 2. Train Models

```bash
python src\train_knn_svm.py --features features.npz --output-dir models
```

This will:
- Grid search for optimal hyperparameters
- Train SVM and KNN models
- Create an ensemble model combining both
- Save all models and training report

### 3. Test on Single Image

```bash
python src\predict.py --image "test_image.jpg" --model models\ensemble_knn_svm.pkl --scaler models\scaler.pkl --class-map models\class_map.json
```

### 4. Real-Time Camera Detection

```bash
python src\deploy.py --model models\ensemble_knn_svm.pkl --scaler models\scaler.pkl --class-map models\class_map.json --confidence 0.65
```

#### Deployment Options:
- `--confidence` - Minimum confidence threshold (default: 0.65)
- `--cam` - Camera index (default: 0)
- `--width` / `--height` - Frame size for processing (default: 128x128)
- `--no-fps` - Disable FPS display

#### Controls:
- Press `Q` to quit
- Press `S` to save screenshot

## Model Performance

The ensemble model combines:
- **SVM**: Good at finding optimal decision boundaries
- **KNN**: Good at capturing local patterns

Expected accuracy: 70-80% on test set

## Improving Accuracy

To improve accuracy further:

1. **Collect Custom Data**: Use your actual camera and materials
   ```bash
   python src\collect_training_data.py --images-per-class 30
   ```

2. **Increase Dataset Size**: Add more diverse images

3. **Tune Confidence Threshold**: Adjust `--confidence` in deploy.py

4. **Feature Engineering**: Modify `src/feature_extraction.py`

## Material Classes

1. Glass
2. Paper
3. Cardboard
4. Plastic
5. Metal
6. Trash

## Requirements

```
opencv-python
scikit-learn
numpy
joblib
```

## Notes

- CNN models have been removed as requested
- Focus is on traditional ML (KNN + SVM)
- Ensemble approach provides better accuracy than individual models
- Real-time detection includes prediction smoothing for stability
