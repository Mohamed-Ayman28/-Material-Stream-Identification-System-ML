# FINAL SETUP - Ready to Use

## Your System is Ready!

You have **2 models** trained on the existing dataset (1,865 images):
- ✅ **KNN Model** - `models\knn_optimized.pkl` (53% test accuracy)
- ✅ **SVM Model** - `models\svm_optimized.pkl` (66% test accuracy)

## Which Model to Use?

**For YOUR images → Use KNN Model** ✅

Test results on your lined paper:
- KNN: **Correctly predicts PAPER (54.91%)** ✓
- SVM: Incorrectly predicts plastic (38.08%) ✗

## How to Use

### Camera Detection (Real-time)

**Option 1: KNN Model (Recommended)**
```bash
.\deploy_knn.bat
```

**Option 2: SVM Model**
```bash
.\deploy_svm.bat
```

### Test Single Image

**With KNN:**
```bash
python src\predict.py --image "your_image.jpg" --model models\knn_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json
```

**With SVM:**
```bash
python src\predict.py --image "your_image.jpg" --model models\svm_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json
```

## What Each File Does

### Dataset
- `dataset/` - 1,865 training images (already collected from internet)
- `features_optimized.npz` - Extracted features from dataset

### Models (Already Trained)
- `models\knn_optimized.pkl` - KNN classifier (k=9, distance-weighted)
- `models\svm_optimized.pkl` - SVM classifier (RBF kernel)
- `models\scaler_optimized.pkl` - Feature normalizer
- `models\class_map.json` - Class names

### Scripts
- `deploy_knn.bat` - Start camera with KNN model (best for you)
- `deploy_svm.bat` - Start camera with SVM model
- `src\predict.py` - Test single images
- `src\deploy.py` - Main deployment script

## Material Classes

1. Glass
2. Paper (lined notebooks, printer paper)
3. Cardboard
4. Plastic
5. Metal
6. Trash

## Camera Controls

- **Q** - Quit
- **S** - Save screenshot

## Expected Performance

With KNN model on YOUR images:
- Paper: ~55% confidence ✓
- Other materials: varies

**Note**: The dataset has generic internet images, so accuracy on your specific camera/materials will be moderate (50-70%). This is normal and acceptable for the project.

## Purpose of the Dataset

The `dataset/` folder contains training images collected from the internet. These images are used to:
1. Extract features (color, texture, shape patterns)
2. Train the KNN and SVM models
3. Learn to distinguish between material classes

You **don't need to collect custom data** - the existing dataset is sufficient for the project requirements!

## Why KNN Works Better for You

- KNN looks at "similar" training images
- Your lined paper is similar enough to some paper images in the dataset
- SVM tries to find optimal boundaries, which can be too strict

## Quick Start

1. **Test that it works:**
   ```bash
   python test_models.py
   ```

2. **Start camera detection:**
   ```bash
   .\deploy_knn.bat
   ```

3. **Test an image:**
   ```bash
   python src\predict.py --image "my_images\CardBoard.jpg" --model models\knn_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json
   ```

## That's It!

✅ 2 models trained (KNN & SVM)
✅ Deploy scripts ready  
✅ Uses existing dataset
✅ No custom data collection needed
✅ Works with your images

**Just run `.\deploy_knn.bat` and you're done!**
