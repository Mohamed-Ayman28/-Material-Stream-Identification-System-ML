"""
balance_dataset.py - Data Augmentation to Balance Dataset

PDF Requirement: "Making all class counts nearly the same size, e.g., 500, using data augmentation"

This script augments the dataset to balance all 6 primary classes to approximately 500 images each
using rotation, flipping, scaling, brightness adjustment, and noise addition.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from augmentation import rotateImage, flipImage, zoomImage, changeBrightness, addNoise


def augment_image_random(image):
    """
    Apply random augmentation to an image
    
    Augmentation techniques (as per PDF requirements):
    - Rotation: Random angles
    - Flipping: Horizontal/vertical
    - Scaling: Zoom in/out
    - Color jitter: Brightness variation
    - Noise: Gaussian noise
    
    Returns:
        Augmented image
    """
    # Randomly choose augmentation technique
    augmentation_type = random.choice(['rotate', 'flip', 'zoom', 'brightness', 'noise', 'combined'])
    
    if augmentation_type == 'rotate':
        angle = random.choice([15, 30, 45, 60, 90, 180, 270])
        return rotateImage(image, angle)
    
    elif augmentation_type == 'flip':
        mode = random.choice(['horizontal', 'vertical'])
        return flipImage(image, mode)
    
    elif augmentation_type == 'zoom':
        zoom_factor = random.uniform(1.1, 1.5)
        return zoomImage(image, zoom_factor)
    
    elif augmentation_type == 'brightness':
        factor = random.uniform(0.7, 1.3)
        return changeBrightness(image, factor)
    
    elif augmentation_type == 'noise':
        return addNoise(image)
    
    elif augmentation_type == 'combined':
        # Apply 2-3 random augmentations
        img = image.copy()
        num_augmentations = random.randint(2, 3)
        techniques = random.sample(['rotate', 'flip', 'brightness'], num_augmentations)
        
        for tech in techniques:
            if tech == 'rotate':
                angle = random.choice([15, 30, 45])
                img = rotateImage(img, angle)
            elif tech == 'flip':
                mode = random.choice(['horizontal', 'vertical'])
                img = flipImage(img, mode)
            elif tech == 'brightness':
                factor = random.uniform(0.8, 1.2)
                img = changeBrightness(img, factor)
        
        return img
    
    return image


def balance_dataset(dataset_path, target_per_class=500, output_path=None):
    """
    Balance dataset by augmenting underrepresented classes
    
    Args:
        dataset_path: Path to dataset root
        target_per_class: Target number of images per class (default: 500)
        output_path: Output path for balanced dataset (if None, augment in place)
    
    Returns:
        Dictionary with augmentation statistics
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path
    else:
        output_path = Path(output_path)
    
    classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    stats = {}
    
    print("\n" + "="*80)
    print("DATASET BALANCING - DATA AUGMENTATION")
    print("PDF Requirement: Balance all classes to ~500 images each")
    print("="*80 + "\n")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in classes:
        class_path = dataset_path / class_name
        output_class_path = output_path / class_name
        
        # Create output directory
        output_class_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        original_count = len(image_files)
        total_original += original_count
        
        # Calculate how many augmented images we need
        needed = max(0, target_per_class - original_count)
        
        print(f"\n{class_name.upper()}:")
        print(f"  Original: {original_count} images")
        print(f"  Target:   {target_per_class} images")
        print(f"  Need:     {needed} augmented images ({needed/original_count*100:.1f}% increase)")
        
        # Copy original images if output path is different
        if output_path != dataset_path:
            for img_file in tqdm(image_files, desc=f"  Copying {class_name}"):
                shutil.copy2(img_file, output_class_path / img_file.name)
        
        # Generate augmented images
        if needed > 0:
            augmented = 0
            with tqdm(total=needed, desc=f"  Augmenting {class_name}") as pbar:
                while augmented < needed:
                    # Randomly select an image to augment
                    img_file = random.choice(image_files)
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        continue
                    
                    # Apply random augmentation
                    aug_img = augment_image_random(img)
                    
                    # Save augmented image
                    base_name = img_file.stem
                    ext = img_file.suffix
                    aug_filename = f"{base_name}_aug_{augmented:04d}{ext}"
                    aug_path = output_class_path / aug_filename
                    
                    cv2.imwrite(str(aug_path), aug_img)
                    augmented += 1
                    pbar.update(1)
            
            total_augmented += augmented
        
        final_count = original_count + needed
        stats[class_name] = {
            'original': original_count,
            'augmented': needed,
            'final': final_count,
            'augmentation_percentage': (needed / original_count * 100) if original_count > 0 else 0
        }
        
        print(f"  Final:    {final_count} images ✅")
    
    print("\n" + "="*80)
    print("AUGMENTATION SUMMARY")
    print("="*80)
    print(f"Total original images:   {total_original}")
    print(f"Total augmented images:  {total_augmented}")
    print(f"Total final images:      {total_original + total_augmented}")
    print(f"Overall augmentation:    {total_augmented/total_original*100:.1f}%")
    print(f"\nPDF Requirement: Minimum 30% augmentation ✅ SATISFIED" if total_augmented/total_original >= 0.30 else "\nPDF Requirement: Minimum 30% augmentation ❌ NOT MET")
    print("="*80 + "\n")
    
    return stats


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset using data augmentation')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--target', type=int, default=500,
                        help='Target number of images per class (default: 500)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for balanced dataset (default: augment in place)')
    
    args = parser.parse_args()
    
    # Run balancing
    stats = balance_dataset(args.dataset, args.target, args.output)
    
    # Save statistics
    import json
    stats_file = Path(args.output if args.output else args.dataset) / 'augmentation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()
