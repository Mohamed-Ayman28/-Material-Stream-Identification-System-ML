"""
collect_training_data.py
Interactive webcam tool to collect your own training images.
This will help create a dataset in your actual environment (lighting, camera, background).
"""

import argparse
import cv2
import os
from pathlib import Path
import time


def collect_images_for_class(class_name, output_dir, num_images=30, countdown=3):
    """
    Collect training images for a specific class using webcam.
    
    Args:
        class_name: Name of the material class (e.g., 'glass', 'paper')
        output_dir: Directory to save images
        num_images: Number of images to capture per class
        countdown: Countdown seconds before each capture
    """
    # Create output directory
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("\n" + "="*60)
    print(f"COLLECTING IMAGES FOR: {class_name.upper()}")
    print("="*60)
    print(f"Target: {num_images} images")
    print(f"\nInstructions:")
    print(f"1. Hold the {class_name} object in front of the camera")
    print(f"2. Press SPACE to start capturing")
    print(f"3. Move the object to different angles/positions between captures")
    print(f"4. Press 'q' to finish early")
    print(f"5. Press 'r' to retake the last image")
    print("="*60)
    
    images_captured = 0
    capturing = False
    last_capture_time = 0
    last_image_path = None
    
    window_name = f'Collecting: {class_name} (Press SPACE to start, Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Display frame with instructions
        display_frame = frame.copy()
        
        # Add text overlay
        if not capturing:
            text = f"Press SPACE to start capturing {class_name}"
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            current_time = time.time()
            time_since_last = current_time - last_capture_time
            
            if time_since_last < countdown:
                # Show countdown
                remaining = int(countdown - time_since_last) + 1
                text = f"Get ready... {remaining}"
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # Capture image
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{images_captured+1:03d}_{timestamp}.jpg"
                filepath = class_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                images_captured += 1
                last_image_path = filepath
                last_capture_time = current_time
                
                print(f"✓ Captured {images_captured}/{num_images}: {filename}")
                
                # Flash effect
                cv2.rectangle(display_frame, (0, 0), 
                            (display_frame.shape[1], display_frame.shape[0]), 
                            (255, 255, 255), 20)
        
        # Progress bar
        progress_width = int((images_captured / num_images) * 600)
        cv2.rectangle(display_frame, (10, 60), (610, 80), (200, 200, 200), -1)
        cv2.rectangle(display_frame, (10, 60), (10 + progress_width, 80), (0, 255, 0), -1)
        
        progress_text = f"{images_captured}/{num_images} images"
        cv2.putText(display_frame, progress_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - start/continue capturing
            if not capturing:
                capturing = True
                last_capture_time = time.time()
                print(f"\nStarting capture sequence...")
        
        elif key == ord('q'):  # Q - quit
            print(f"\nStopping early. Captured {images_captured} images.")
            break
        
        elif key == ord('r'):  # R - retake last image
            if last_image_path and last_image_path.exists():
                last_image_path.unlink()
                images_captured -= 1
                print(f"✗ Deleted last image. Count: {images_captured}/{num_images}")
                last_image_path = None
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Finished! Collected {images_captured} images for '{class_name}'")
    print(f"Saved to: {class_dir}")
    return images_captured


def main():
    parser = argparse.ArgumentParser(description='Collect training data using webcam')
    parser.add_argument('--output-dir', type=str, default='my_training_data',
                        help='Directory to save collected images')
    parser.add_argument('--images-per-class', type=int, default=30,
                        help='Number of images to collect per class')
    parser.add_argument('--countdown', type=int, default=2,
                        help='Countdown seconds between captures')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash'],
                        help='List of classes to collect')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TRAINING DATA COLLECTION TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Images per class: {args.images_per_class}")
    print(f"Classes: {', '.join(args.classes)}")
    print("="*60)
    
    total_images = 0
    
    for class_name in args.classes:
        input(f"\n>>> Press ENTER to start collecting '{class_name}' images...")
        
        count = collect_images_for_class(
            class_name, 
            args.output_dir, 
            args.images_per_class,
            args.countdown
        )
        total_images += count
        
        print(f"\nCompleted: {class_name}")
        time.sleep(1)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"Total images collected: {total_images}")
    print(f"Saved to: {args.output_dir}")
    print("\nNext steps:")
    print(f"1. Review images in '{args.output_dir}' folder")
    print(f"2. Delete any bad/blurry images")
    print(f"3. Run: python src/main.py --mode extract --input-dir {args.output_dir} --features-file my_features.npz")
    print(f"4. Run: python src/main.py --mode train --features-file my_features.npz --model-type both --fast")
    print("="*60)


if __name__ == '__main__':
    main()
