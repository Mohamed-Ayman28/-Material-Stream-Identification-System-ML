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

    print('\n' + '='*60)
    print('CAMERA INITIALIZATION')
    print('='*60)
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera index ' + str(cam_index))

    # Set camera properties for consistent capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Warm up camera (stabilize exposure and white balance)
    print('Warming up camera...')
    for i in range(30):
        ret, _ = cap.read()
        if i % 10 == 0:
            print(f'  Frame {i+1}/30')
    print('✓ Camera ready!\n')
    
    print('TIPS FOR BEST RESULTS:')
    print('  • Use good, even lighting')
    print('  • Hold object steady in detection zone')
    print('  • Fill 50-70% of green box with object')
    print('  • Wait for prediction to stabilize (2-3 seconds)')
    print('  • Avoid shadows and reflections')
    print('='*60 + '\n')

    last_time = time.time()
    fps = 0.0
    
    # Exponential Moving Average for smoother predictions
    ema_alpha = 0.3  
    num_classes = len(class_map) if class_map else 6
    class_confidences = [0.0] * num_classes
    last_stable_prediction = None
    stable_count = 0

    window_name = 'Material Detection - Press Q to quit, S to screenshot'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed; exiting')
                break

            # Use center region for prediction (more stable, avoids edges)
            h, w = frame.shape[:2]
            # Crop to center 50% of frame (more focused detection zone)
            crop_h, crop_w = int(h * 0.5), int(w * 0.5)
            start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
            cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            # Enhanced preprocessing for camera frames
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            # Normalize brightness/contrast to reduce lighting variations
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Adaptive histogram equalization for lighting consistency
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge back and convert to RGB
            lab = cv2.merge([l_channel, a, b])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Resize for feature extraction
            resized = cv2.resize(rgb, frame_size)

            # Extract and scale features
            feat = extract_features(resized)
            feat = feat.reshape(1, -1)
            if scaler is not None:
                feat = scaler.transform(feat)

            label_idx = None
            conf_val = 0.0
            status_text = ''

            try:
                rej = predict_with_rejection(
                    model, feat,
                    svm_min_prob=0.35,  
                    svm_min_prob_gap=0.10,
                    knn_max_mean_distance=55.0,  
                    ensemble_min_prob=0.30,
                    ensemble_min_prob_gap=0.05
                )
                
                raw_label = rej.label_idx
                raw_conf = float(rej.confidence) if rej.confidence is not None else 0.0

                for i in range(num_classes):
                    class_confidences[i] *= (1 - ema_alpha)
                
                # Boost current prediction if valid
                if raw_label is not None and raw_conf > 0.30:
                    class_confidences[raw_label] += ema_alpha * raw_conf
                
                # Get smoothed prediction (highest confidence)
                max_conf = max(class_confidences)
                if max_conf > 0.35:  # Minimum threshold for display
                    label_idx = class_confidences.index(max_conf)
                    conf_val = max_conf
                    
                    # Stability detection
                    if label_idx == last_stable_prediction:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_stable_prediction = label_idx
                    
                    is_stable = stable_count >= 3
                    status_text = f'{conf_val:.1%}'
                    if is_stable:
                        status_text += ' ✓'
                else:
                    label_idx = None
                    conf_val = 0.0
                    status_text = 'Not Recognized'
                    last_stable_prediction = None
                    stable_count = 0
                    
            except Exception as e:
                print(f'Prediction error: {e}')
                label_idx = None
                status_text = 'Error'

            # Prepare display with stability-aware colors
            if label_idx is not None:
                label_text = pretty_label(label_idx, class_map)
                # Color based on stability and confidence
                if stable_count >= 3 and conf_val >= 0.60:
                    color = (0, 255, 0)  
                elif stable_count >= 3:
                    color = (0, 255, 255)  
                elif conf_val >= 0.60:
                    color = (0, 200, 255)  
                else:
                    color = (0, 165, 255) 
            else:
                label_text = 'Unknown'
                color = (0, 0, 255)  
                
            # Draw detection zone rectangle
            zone_color = (0, 255, 0) if label_idx is not None else (128, 128, 128)
            cv2.rectangle(frame, (start_x, start_y), 
                         (start_x + crop_w, start_y + crop_h),
                         zone_color, 3)
            cv2.putText(frame, 'DETECTION ZONE', (start_x + 10, start_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)

            # Draw prediction box with semi-transparent background
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (10, 10), (w-10, 130), color, 3)
            
            # Draw text
            cv2.putText(frame, f'Material: {label_text.upper()}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f'Confidence: {status_text}', (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Model: {model_type}', (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
