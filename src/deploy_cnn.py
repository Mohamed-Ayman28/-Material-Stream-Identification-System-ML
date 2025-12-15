"""
Real-time CNN Deployment for Webcam
Uses trained CNN model for real-time material classification
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
import argparse
from pathlib import Path


class CNNDeployer:
    def __init__(self, model_path, class_map_path, img_size=(224, 224), 
                 confidence_threshold=0.5):
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading CNN model from: {model_path}")
        if model_path.endswith('.h5'):
            self.model = keras.models.load_model(model_path)
        else:
            self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load class map
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)
        self.class_map = {int(k): v for k, v in self.class_map.items()}
        print(f"Classes: {self.class_map}")
        
        # Class colors for visualization (BGR format)
        self.class_colors = {
            'glass': (255, 200, 100),      # Light blue
            'paper': (200, 200, 200),      # Light gray
            'cardboard': (100, 150, 255),  # Orange
            'plastic': (100, 255, 100),    # Green
            'metal': (200, 200, 0),        # Cyan
            'trash': (50, 50, 50)          # Dark gray
        }
        
    def preprocess_frame(self, frame):
        """Preprocess frame for CNN prediction"""
        # Resize
        img = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and preprocess
        img = img.astype(np.float32) / 255.0
        img = keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame):
        """Predict material class for frame"""
        # Preprocess
        img = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        
        # Get prediction
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx]
        class_name = self.class_map[pred_idx]
        
        # Get top 3
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [(self.class_map[i], predictions[0][i]) for i in top_3_idx]
        
        return class_name, confidence, top_3
    
    def draw_results(self, frame, class_name, confidence, top_3, fps):
        """Draw prediction results on frame"""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Main prediction
        if confidence >= self.confidence_threshold:
            color = self.class_colors.get(class_name, (255, 255, 255))
            text = f"{class_name.upper()}: {confidence*100:.1f}%"
            status = "CONFIDENT"
            status_color = (0, 255, 0)
        else:
            color = (100, 100, 100)
            text = f"{class_name}: {confidence*100:.1f}%"
            status = "UNCERTAIN"
            status_color = (0, 165, 255)
        
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_BOLD, 
                    1.2, color, 3)
        cv2.putText(frame, f"Status: {status}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Top 3 predictions
        cv2.putText(frame, "Top 3:", (20, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        for i, (cls, conf) in enumerate(top_3):
            y_pos = 140 + i * 20
            text = f"{i+1}. {cls}: {conf*100:.1f}%"
            cv2.putText(frame, text, (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save screenshot", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (200, 200, 200), 1)
        
        return frame
    
    def run_camera(self, camera_id=0, save_dir='screenshots'):
        """Run real-time detection on camera feed"""
        # Create save directory
        Path(save_dir).mkdir(exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "="*60)
        print("CNN REAL-TIME DETECTION")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("="*60 + "\n")
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            
            # Predict
            class_name, confidence, top_3 = self.predict(frame)
            
            # Draw results
            display_frame = self.draw_results(frame, class_name, confidence, top_3, fps)
            
            # Show frame
            cv2.imshow('CNN Material Detection', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"{save_dir}/cnn_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
                print(f"  Class: {class_name} ({confidence*100:.2f}%)")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera stopped.")


def main():
    parser = argparse.ArgumentParser(description='Real-time CNN material detection')
    parser.add_argument('--model', type=str, default='models/cnn_model.h5',
                        help='Path to trained CNN model')
    parser.add_argument('--class-map', type=str, default='models/cnn_class_map.json',
                        help='Path to class mapping JSON')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for model input (default: 224)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--save-dir', type=str, default='screenshots_cnn',
                        help='Directory to save screenshots')
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = CNNDeployer(
        model_path=args.model,
        class_map_path=args.class_map,
        img_size=(args.img_size, args.img_size),
        confidence_threshold=args.threshold
    )
    
    # Run camera
    deployer.run_camera(
        camera_id=args.camera,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
