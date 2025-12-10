#deploy.py
##Real-time webcam deployment for the Material Stream Identification (MSI) System.

import argparse
import time
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from feature_extraction import extract_features
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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
               svm_threshold=0.6, knn_dist_threshold=0.5, cam_index=0,
               frame_size=(128, 128), show_fps=True):

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
    is_svm = False
    is_knn = False
    
    if isinstance(model, SVC):
        is_svm = True
    if isinstance(model, KNeighborsClassifier):
        is_knn = True

    print('Model type:', 'SVM' if is_svm else ('k-NN' if is_knn else type(model)))

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera index ' + str(cam_index))

    last_time = time.time()
    fps = 0.0

    window_name = 'MSI - Live (press q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed; exiting')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, frame_size)

            feat = extract_features(resized)
            feat = feat.reshape(1, -1)
            if scaler is not None:
                feat = scaler.transform(feat)

            label_idx = None
            conf_val = 0.0
            extra = ''

            if is_svm:
                label_idx, conf_val = svm_predict_with_reject(model, feat, threshold=svm_threshold)
                if label_idx is None:
                    extra = f' (rejected, prob={conf_val:.3f})'
                else:
                    extra = f' (prob={conf_val:.3f})'
            elif is_knn:
                pred, conf_val, mean_dist = knn_predict_with_reject(model, feat, dist_threshold=knn_dist_threshold)
                label_idx = pred
                if label_idx is None:
                    extra = f' (rejected, mean_dist={mean_dist:.3f})'
                else:
                    extra = f' (inv-dist-conf={conf_val:.3f})'
            else:
                # Generic model: try predict + probability
                try:
                    probs = model.predict_proba(feat)
                    p = float(np.max(probs))
                    pred = int(np.argmax(probs))
                    if p >= svm_threshold:
                        label_idx = pred
                        conf_val = p
                        extra = f' (prob={p:.3f})'
                    else:
                        label_idx = None
                        conf_val = p
                        extra = f' (rejected, prob={p:.3f})'
                except Exception:
                    pred = int(model.predict(feat)[0])
                    label_idx = pred
                    conf_val = 1.0
                    extra = ''

            label_text = pretty_label(label_idx, class_map)

            # draw text on frame
            text = f'Pred: {label_text}{extra}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # draw FPS
            if show_fps:
                now = time.time()
                dt = now - last_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps != 0 else (1.0 / dt)
                last_time = now
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to best model (best_model.pkl or svm_model.pkl / knn_model.pkl)')
    p.add_argument('--scaler', required=False, help='Path to scaler.pkl (StandardScaler). Optional but recommended')
    p.add_argument('--class_map', required=False, help='Path to class_map JSON or training_report.json containing class_map')
    p.add_argument('--svm_threshold', type=float, default=0.6, help='Probability threshold for SVM to accept a prediction')
    p.add_argument('--knn_dist_threshold', type=float, default=0.5, help='Distance threshold for k-NN to accept a prediction (lower = stricter)')
    p.add_argument('--cam', type=int, default=0, help='Camera index for cv2.VideoCapture')
    p.add_argument('--width', type=int, default=128, help='Width to resize frames for feature extraction')
    p.add_argument('--height', type=int, default=128, help='Height to resize frames for feature extraction')
    p.add_argument('--no-fps', dest='show_fps', action='store_false', help='Disable FPS overlay')
    args = p.parse_args()

    run_camera(args.model, scaler_path=args.scaler, class_map_path=args.class_map,
               svm_threshold=args.svm_threshold, knn_dist_threshold=args.knn_dist_threshold,
               cam_index=args.cam, frame_size=(args.width, args.height), show_fps=args.show_fps)
