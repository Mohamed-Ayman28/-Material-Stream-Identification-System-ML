"""
collect_custom_data.py - Collect custom training images with your camera
This will help the model learn YOUR specific materials
"""

import cv2
import os
import time
from pathlib import Path
import argparse


def collect_images_for_class(class_name, num_images, output_dir, cam_index=0):
    """Collect training images for a specific class"""
    
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {cam_index}')
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f'\n{"="*60}')
    print(f'Collecting {num_images} images for: {class_name.upper()}')
    print(f'{"="*60}')
    print('\nInstructions:')
    print('  - Hold the material in front of the camera')
    print('  - Move it around, rotate it, change angles')
    print('  - Try different lighting and backgrounds')
    print('  - Press SPACE to capture an image')
    print('  - Press Q to finish early')
    print(f'\nImages will be saved to: {class_dir}')
    print(f'\nTarget: {num_images} images\n')
    
    window_name = f'Collecting: {class_name} - Press SPACE to capture, Q to finish'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    collected = 0
    existing_files = list(class_dir.glob('*.jpg'))
    start_idx = len(existing_files)
    
    try:
        while collected < num_images:
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame')
                break
            
            # Add text overlay
            remaining = num_images - collected
            status = f'{collected}/{num_images} collected - {remaining} remaining'
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)
            cv2.putText(frame, 'Press SPACE to capture', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw capture area guide
            h, w = frame.shape[:2]
            margin = 50
            cv2.rectangle(frame, (margin, margin), (w-margin, h-margin), 
                         (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to capture
                filename = class_dir / f'{class_name}_{start_idx + collected:04d}.jpg'
                cv2.imwrite(str(filename), frame)
                collected += 1
                print(f'✓ Captured {collected}/{num_images}: {filename.name}')
                
                # Visual feedback
                time.sleep(0.2)  # Prevent accidental double-capture
                
            elif key == ord('q'):  # Q to quit early
                print(f'\nStopping early. Collected {collected} images.')
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f'\n{"="*60}')
    print(f'Finished! Collected {collected} images for {class_name}')
    print(f'{"="*60}\n')
    
    return collected


def main():
    parser = argparse.ArgumentParser(
        description='Collect custom training images with your camera'
    )
    parser.add_argument('--output-dir', default='custom_dataset',
                       help='Directory to save images')
    parser.add_argument('--images-per-class', type=int, default=30,
                       help='Number of images to collect per class')
    parser.add_argument('--cam', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--classes', nargs='+', 
                       default=['paper', 'cardboard', 'plastic', 'metal', 'glass', 'trash'],
                       help='Classes to collect')
    
    args = parser.parse_args()
    
    print('\n' + '='*60)
    print('CUSTOM TRAINING DATA COLLECTION')
    print('='*60)
    print(f'\nThis will help fix misclassification problems!')
    print(f'\nYou will collect {args.images_per_class} images for each class:')
    for cls in args.classes:
        print(f'  - {cls}')
    
    print(f'\nTotal images to collect: {len(args.classes) * args.images_per_class}')
    print(f'Output directory: {args.output_dir}')
    
    input('\nPress ENTER to start...')
    
    total_collected = 0
    
    for class_name in args.classes:
        collected = collect_images_for_class(
            class_name, 
            args.images_per_class, 
            args.output_dir,
            args.cam
        )
        total_collected += collected
        
        if collected < args.images_per_class:
            print(f'\nWarning: Only collected {collected}/{args.images_per_class} for {class_name}')
        
        # Pause between classes
        if class_name != args.classes[-1]:
            print(f'\nGet ready for next class...')
            input('Press ENTER to continue...')
    
    print('\n' + '='*60)
    print('COLLECTION COMPLETE!')
    print('='*60)
    print(f'\nTotal images collected: {total_collected}')
    print(f'Saved to: {args.output_dir}')
    print('\nNext steps:')
    print(f'  1. Extract features: python src\\main.py --mode extract --input-dir {args.output_dir} --features-file custom_features.npz')
    print(f'  2. Train models: python src\\train_knn_svm.py --features custom_features.npz --output-dir models')
    print(f'  3. Test: python src\\deploy.py --model models\\ensemble_knn_svm.pkl --scaler models\\scaler.pkl --class-map models\\class_map.json')
    print('\n' + '='*60 + '\n')


if __name__ == '__main__':
    main()
