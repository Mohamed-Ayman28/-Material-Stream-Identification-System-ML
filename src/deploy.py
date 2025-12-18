#deploy.py
##Real-time webcam deployment for the Material Stream Identification (MSI) System.
##Optimized for KNN and SVM models with high accuracy detection

import argparse
import time
import json
import os
from pathlib import Path
import cv2
import joblib
import numpy as np

# Try enhanced features first, fall back to original
try:
    from feature_extraction_enhanced import extract_features
    USING_ENHANCED = True
except ImportError:
    from feature_extraction import extract_features
    USING_ENHANCED = False

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from utils import predict_with_rejection

def load_class_map(path):
    if path is None:
        return None
    try:
        with open(path, 'r') as f:
            # support both plain json mapping or training_report.json containing class_map
            j = json.load(f)
            if isinstance(j, dict):
                # simple dict mapping
                return {int(k): v for k, v in j.items()}
            else:
                # maybe training_report.json with key 'class_map'
                if 'class_map' in j:
                    cm = j['class_map']
                    return {int(k): v for k, v in cm.items()}
    except Exception as e:
        print('Warning: cannot load class_map from', path, e)
    return None


def pretty_label(label_idx, class_map):
    if label_idx is None:
        return 'Unknown'
    if class_map is None:
        return str(label_idx)
    return class_map.get(int(label_idx), str(label_idx))


def svm_predict_with_reject(model, X_feat, threshold=0.6):
    # X_feat: (1, D)
    try:
        probs = model.predict_proba(X_feat)
        maxp = np.max(probs, axis=1)[0]
        pred = int(np.argmax(probs, axis=1)[0])
        if maxp >= threshold:
            return pred, float(maxp)
        else:
            return None, float(maxp)
    except Exception as e:
        # model without predict_proba
        try:
            dec = model.decision_function(X_feat)
            # convert margin to pseudo confidence via tanh normalization
            # this is a heuristic
            margin = np.max(dec)
            conf = 1.0 / (1.0 + np.exp(-margin))
            pred = int(model.predict(X_feat)[0])
            if conf >= threshold:
                return pred, float(conf)
            else:
                return None, float(conf)
        except Exception:
            pred = int(model.predict(X_feat)[0])
            return pred, 1.0


def knn_predict_with_reject(model, X_feat, dist_threshold=0.5):
    # Use mean neighbor distance to decide rejection. Lower distance => more confident
    try:
        neigh_dist, neigh_idx = model.kneighbors(X_feat, n_neighbors=model.n_neighbors, return_distance=True)
        mean_dist = float(np.mean(neigh_dist))
        pred = int(model.predict(X_feat)[0])
        if mean_dist <= dist_threshold:
            # confidence as inverse distance (heuristic)
            conf = 1.0 / (1.0 + mean_dist)
            return pred, conf, mean_dist
        else:
            conf = 1.0 / (1.0 + mean_dist)
            return None, conf, mean_dist
    except Exception as e:
        # fallback: predict only
        pred = int(model.predict(X_feat)[0])
        return pred, 1.0, 0.0


def run_camera(model_path, scaler_path=None, class_map_path=None,
               confidence_threshold=0.65, cam_index=0,
               frame_size=(128, 128), show_fps=True):
    """
    Run real-time webcam detection with KNN+SVM ensemble
    
    Args:
        model_path: Path to model file (pkl)
        scaler_path: Path to scaler file (pkl)
        class_map_path: Path to class map JSON
        confidence_threshold: Minimum confidence to display prediction (0-1)
        cam_index: Camera device index
        frame_size: Size to resize frames for feature extraction
        show_fps: Whether to display FPS
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    model = joblib.load(str(model_path))
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print('Loaded scaler from', scaler_path)

    class_map = load_class_map(class_map_path) if class_map_path else None
    if class_map:
        print('Loaded class_map with', len(class_map), 'entries')

    # determine model type
    is_svm = isinstance(model, SVC)
    is_knn = isinstance(model, KNeighborsClassifier)
    is_ensemble = isinstance(model, VotingClassifier)

    model_type = 'Ensemble' if is_ensemble else ('SVM' if is_svm else ('KNN' if is_knn else 'Unknown'))
    print(f'Model type: {model_type}')

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera index ' + str(cam_index))

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_time = time.time()
    fps = 0.0
    
    # For smoothing predictions
    prediction_history = []
    max_history = 5

    window_name = 'Material Detection - Press Q to quit, S to screenshot'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed; exiting')
                break

            # Preprocess frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, frame_size)

            # Extract features
            feat = extract_features(resized)
            feat = feat.reshape(1, -1)
            if scaler is not None:
                feat = scaler.transform(feat)

            label_idx = None
            conf_val = 0.0
            status_text = ''

            try:
                # Model-specific rejection first
                rej = predict_with_rejection(model, feat)
                label_idx = rej.label_idx
                conf_val = float(rej.confidence) if rej.confidence is not None else 0.0

                if label_idx is None:
                    # rejected by tailored mechanism
                    if rej.confidence is not None:
                        status_text = f'Unknown ({rej.reason}: {rej.confidence:.2%})'
                    else:
                        status_text = f'Unknown ({rej.reason})'
                else:
                    # Apply the existing deployment threshold as an additional safety gate
                    if rej.confidence is not None and rej.confidence < confidence_threshold:
                        label_idx = None
                        status_text = f'Low confidence: {rej.confidence:.2%}'
                    else:
                        status_text = f'Confidence: {rej.confidence:.2%}' if rej.confidence is not None else 'Confidence: N/A'

                        # Add to history for smoothing
                        prediction_history.append(int(label_idx))
                        if len(prediction_history) > max_history:
                            prediction_history.pop(0)

                        # Use most common prediction from history
                        if len(prediction_history) >= 3:
                            label_idx = max(set(prediction_history), key=prediction_history.count)
                    
            except Exception as e:
                print(f'Prediction error: {e}')
                label_idx = None
                status_text = 'Error'

            # Prepare display text
            if label_idx is not None:
                label_text = pretty_label(label_idx, class_map)
                color = (0, 255, 0)  # Green for confident prediction
            else:
                label_text = 'Unknown'
                color = (0, 0, 255)  # Red for uncertain
                if not prediction_history:
                    prediction_history = []

            # Draw prediction box
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (w-10, 120), color, 2)
            
            # Draw text
            cv2.putText(frame, f'Material: {label_text}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, status_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Model: {model_type}', (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw FPS
            if show_fps:
                now = time.time()
                dt = now - last_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps != 0 else (1.0 / dt)
                last_time = now
                cv2.putText(frame, f'FPS: {fps:.1f}', (w-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(window_name, frame)

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # save a screenshot for debugging
                outp = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(outp, frame)
                print('Saved', outp)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Real-time material detection with KNN+SVM')
    p.add_argument('--model', required=True, help='Path to model file (ensemble_knn_svm.pkl, svm_model.pkl, or knn_model.pkl)')
    p.add_argument('--scaler', required=False, help='Path to scaler.pkl (StandardScaler)')
    p.add_argument('--class-map', required=False, help='Path to class_map.json or training_report.json')
    p.add_argument('--confidence', type=float, default=0.65, help='Minimum confidence threshold (0-1)')
    p.add_argument('--cam', type=int, default=0, help='Camera index')
    p.add_argument('--width', type=int, default=128, help='Frame width for feature extraction')
    p.add_argument('--height', type=int, default=128, help='Frame height for feature extraction')
    p.add_argument('--no-fps', dest='show_fps', action='store_false', help='Disable FPS display')
    args = p.parse_args()

    run_camera(args.model, scaler_path=args.scaler, class_map_path=args.class_map,
               confidence_threshold=args.confidence,
               cam_index=args.cam, frame_size=(args.width, args.height),
               show_fps=args.show_fps)
