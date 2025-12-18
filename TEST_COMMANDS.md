# Quick Test Commands - WITH REJECTION MECHANISM

Use these commands to test any image with the three models that include rejection for low-confidence predictions.

---

## 🎯 ENSEMBLE MODEL (Recommended - Best Accuracy)
Combines KNN + SVM for superior performance with rejection mechanism.

**Command Template:**
```bash
python src\predict.py --image <IMAGE_PATH> --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

**Examples:**
```bash
# Test cardboard
python src\predict.py --image my_images\CardBoard.jpg --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# Test fork (should reject as Unknown)
python src\predict.py --image my_images\Fork.jpg --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# Test plastic bottle
python src\predict.py --image my_images\PlasticBottel.jpg --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

---

## 🔵 SVM MODEL
Support Vector Machine with rejection for uncertain items.

**Command Template:**
```bash
python src\predict.py --image <IMAGE_PATH> --model models\svm_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

**Examples:**
```bash
# Test paper
python src\predict.py --image my_images\paper.jpg --model models\svm_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# Test glass cup
python src\predict.py --image my_images\glassCup.jpg --model models\svm_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

---

## 🟢 KNN MODEL
K-Nearest Neighbors with distance-based rejection.

**Command Template:**
```bash
python src\predict.py --image <IMAGE_PATH> --model models\knn_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

**Examples:**
```bash
# Test written paper
python src\predict.py --image my_images\WrittenPaper.jpg --model models\knn_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# Test eraser
python src\predict.py --image my_images\eraser.jpeg --model models\knn_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

---

## 📊 Test ALL Images at Once
```bash
python test_all_images.py
```

This will test all images in the `my_images` folder and show which are accepted vs rejected as Unknown.

---

## 🎨 Material Classes
The system classifies materials into:
1. **glass** - Glass bottles, jars, cups
2. **paper** - Clean paper, newspapers
3. **cardboard** - Boxes, packaging
4. **plastic** - Bottles, containers, bags
5. **metal** - Cans, foil, containers
6. **trash** - Non-recyclable waste
7. **Unknown** - Low confidence or out-of-distribution items

---

## ⚙️ Rejection Mechanism (How it Works)

Each model uses tailored confidence thresholds:

### Ensemble Model
- Requires **≥60% confidence**
- Requires **≥15% gap** between top-2 predictions
- Rejects ambiguous or low-confidence items

### SVM Model
- Requires **≥65% probability** (if available)
- Falls back to **≥60% margin-based confidence**
- Rejects uncertain classifications

### KNN Model
- Uses **mean neighbor distance**
- Requires **distance ≤0.50**
- Rejects items far from training examples

**Result:** Items that don't meet these criteria → classified as **"Unknown"**

---

## 📝 Quick Copy-Paste Commands

Replace `YOUR_IMAGE.jpg` with your actual image path:

```bash
# Ensemble (best overall)
python src\predict.py --image YOUR_IMAGE.jpg --model models\ensemble_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# SVM
python src\predict.py --image YOUR_IMAGE.jpg --model models\svm_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json

# KNN
python src\predict.py --image YOUR_IMAGE.jpg --model models\knn_enhanced.pkl --scaler models\scaler_enhanced_87.pkl --class-map models\class_map_enhanced.json
```

---

## 🖼️ Test Images Available in `my_images` folder:
- CardBoard.jpg
- eraser.jpeg
- Fork.jpg
- glassCup.jpg
- glassCup2.jpg
- paper.jpg
- plastic.jpg
- PlasticBottel.jpg
- screenshot_1765461986.jpg
- WoodStick.jpg
- WrittenPaper.jpg
