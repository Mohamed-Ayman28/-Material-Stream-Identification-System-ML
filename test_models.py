"""
test_models.py - Quick test script to verify KNN and SVM models
"""

import joblib
import numpy as np
from pathlib import Path


def test_models():
    """Test that all models can load and make predictions"""
    
    models_dir = Path('models')
    
    print("Testing Material Stream Identification System")
    print("=" * 60)
    
    # Test loading models
    models_to_test = [
        ('svm_optimized.pkl', 'SVM Model'),
        ('knn_optimized.pkl', 'KNN Model'),
        ('ensemble_model.pkl', 'Ensemble Model'),
        ('scaler_optimized.pkl', 'Feature Scaler')
    ]
    
    loaded_models = {}
    
    for filename, name in models_to_test:
        filepath = models_dir / filename
        if filepath.exists():
            try:
                model = joblib.load(filepath)
                loaded_models[filename] = model
                print(f"✓ {name:20s} - Loaded successfully")
            except Exception as e:
                print(f"✗ {name:20s} - Error: {e}")
        else:
            print(f"✗ {name:20s} - File not found")
    
    print("\n" + "=" * 60)
    
    # Test prediction with dummy data
    if 'scaler_optimized.pkl' in loaded_models:
        print("\nTesting prediction with dummy features...")
        
        scaler = loaded_models['scaler_optimized.pkl']
        dummy_features = np.random.rand(1, 1881)  # Assuming 1881 features
        
        scaled_features = scaler.transform(dummy_features)
        
        for filename, name in [('svm_optimized.pkl', 'SVM'), 
                               ('knn_optimized.pkl', 'KNN'), 
                               ('ensemble_model.pkl', 'Ensemble')]:
            if filename in loaded_models:
                try:
                    model = loaded_models[filename]
                    prediction = model.predict(scaled_features)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(scaled_features)[0]
                        max_prob = np.max(proba)
                        print(f"  {name:10s}: Class {prediction}, Confidence: {max_prob:.2%}")
                    else:
                        print(f"  {name:10s}: Class {prediction}")
                except Exception as e:
                    print(f"  {name:10s}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("\nAll tests completed!")
    print("\nNext steps:")
    print("1. Test on real image:")
    print('   python src\\predict.py --image "test.jpg" --model models\\ensemble_model.pkl --scaler models\\scaler_optimized.pkl --class-map models\\class_map.json')
    print("\n2. Start camera detection:")
    print('   python src\\deploy.py --model models\\ensemble_model.pkl --scaler models\\scaler_optimized.pkl --class-map models\\class_map.json')
    

if __name__ == '__main__':
    test_models()
