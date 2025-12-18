# FIXING MISCLASSIFICATION PROBLEM

## The Problem
Your model predicts **paper as plastic** with low confidence:
- Predicted: plastic (32.27%)
- Actual: paper (28.13%)

## Why This Happens
The model was trained on **generic internet images**, not YOUR specific:
- Camera
- Lighting
- Materials
- Background

This is called "**domain gap**" - training data ≠ real-world data.

## THE SOLUTION (3 Steps)

### Step 1: Collect Custom Images (30 per class)

Run this:
```bash
python src\collect_custom_data.py --images-per-class 30
```

**What to do**:
1. Camera opens for "paper" first
2. Hold your lined notebook/paper in front of camera
3. Press **SPACEBAR** to capture (30 times)
4. Move paper around, rotate it, try different angles
5. Repeat for cardboard, plastic, metal, glass, trash

**Tips for better results**:
- ✓ Use the SAME materials you'll detect later
- ✓ Try different lighting (natural, lamp, etc.)
- ✓ Various backgrounds
- ✓ Different distances from camera
- ✓ Include crumpled, folded, flat versions

### Step 2: Extract Features & Retrain

```bash
python src\main.py --mode extract --input-dir custom_dataset --features-file custom_features.npz
python src\train_knn_svm.py --features custom_features.npz --output-dir models_custom
```

### Step 3: Test the New Model

```bash
python src\predict.py --image "my_images\CardBoard.jpg" --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json
```

You should now see **much higher confidence** and **correct predictions**!

## EASY MODE: Run Everything Automatically

```bash
retrain.bat
```

This runs all 3 steps automatically!

## Alternative: Quick Test with Different Model

Try using just the SVM model (sometimes better for specific cases):

```bash
python src\predict.py --image "my_images\CardBoard.jpg" --model models\svm_optimized.pkl --scaler models\scaler_optimized.pkl --class-map models\class_map.json
```

## Understanding the Probabilities

Your current prediction:
```
plastic   : 32.27%  ← Wrong
paper     : 28.13%  ← Should be this!
metal     : 22.74%
cardboard : 10.88%
```

**Low confidence + close probabilities = Model is guessing**

After retraining with YOUR data:
```
paper     : 85.00%  ← Clear winner!
plastic   : 8.00%
cardboard : 4.00%
```

**High confidence + clear winner = Model is certain**

## Expected Results After Retraining

- ✓ Paper detected as paper (80-90% confidence)
- ✓ Plastic detected as plastic (80-90% confidence)
- ✓ Other materials correctly classified
- ✓ Higher overall accuracy in camera detection

## How Many Images Do I Need?

- **Minimum**: 20 images per class (120 total)
- **Recommended**: 30 images per class (180 total) ← **Use this**
- **Best**: 50+ images per class (300+ total)

## Time Required

- Collection: ~15 minutes (30 images × 6 classes)
- Feature extraction: ~2 minutes
- Training: ~5-10 minutes
- **Total: ~25 minutes to fix the problem!**

## Still Having Issues?

### If confidence is still low:
1. Collect more images (50 per class)
2. Ensure good lighting when collecting
3. Include more variety (wrinkled paper, different types)

### If wrong class predicted:
1. Make sure you're using YOUR camera for collection
2. Use the EXACT same materials you want to detect
3. Try adjusting confidence threshold in deploy.py:
   ```bash
   python src\deploy.py --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json --confidence 0.50
   ```

## Quick Reference Commands

**Collect data**:
```bash
python src\collect_custom_data.py --images-per-class 30
```

**Extract features**:
```bash
python src\main.py --mode extract --input-dir custom_dataset --features-file custom_features.npz
```

**Train**:
```bash
python src\train_knn_svm.py --features custom_features.npz --output-dir models_custom
```

**Test image**:
```bash
python src\predict.py --image "my_images\CardBoard.jpg" --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json
```

**Camera detection**:
```bash
python src\deploy.py --model models_custom\ensemble_knn_svm.pkl --scaler models_custom\scaler.pkl --class-map models_custom\class_map.json
```

---

**Bottom Line**: The current model doesn't know YOUR specific materials. Collect 30 images of each material with YOUR camera, retrain, and the problem will be fixed!
