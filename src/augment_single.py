"""
augment_single.py
Augment a single image file with various transformations.
"""

import argparse
import cv2
from pathlib import Path
from augmentation import (
    rotateImage, 
    flipImage, 
    zoomImage, 
    changeBrightness, 
    random_noise,
    combinedAugmentation
)


def main():
    parser = argparse.ArgumentParser(description='Augment a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='augmented_images',
                        help='Directory to save augmented images')
    parser.add_argument('--num-aug', type=int, default=5,
                        help='Number of augmented versions to create')
    parser.add_argument('--mode', type=str, default='combined',
                        choices=['combined', 'rotate', 'flip', 'zoom', 'brightness', 'noise'],
                        help='Augmentation mode')
    parser.add_argument('--angle', type=float, default=30,
                        help='Rotation angle (for rotate mode)')
    parser.add_argument('--flip-dir', type=str, default='horizontal',
                        choices=['horizontal', 'vertical', 'both'],
                        help='Flip direction (for flip mode)')
    parser.add_argument('--zoom-factor', type=float, default=1.3,
                        help='Zoom factor (for zoom mode)')
    parser.add_argument('--brightness-factor', type=float, default=1.2,
                        help='Brightness factor (for brightness mode)')
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    print(f"Loaded image: {args.image}")
    print(f"Image shape: {img.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original filename
    img_path = Path(args.image)
    base_name = img_path.stem
    ext = img_path.suffix
    
    # Save original
    original_path = output_dir / f"{base_name}_original{ext}"
    cv2.imwrite(str(original_path), img)
    print(f"\nSaved original: {original_path}")
    
    # Apply augmentations
    print(f"\nGenerating {args.num_aug} augmented versions...")
    
    for i in range(args.num_aug):
        if args.mode == 'combined':
            aug_img = combinedAugmentation(img.copy())
            aug_name = f"{base_name}_combined_{i+1}{ext}"
        
        elif args.mode == 'rotate':
            angle = args.angle if i == 0 else args.angle * (i + 1) / args.num_aug * 60
            aug_img = rotateImage(img.copy(), angle)
            aug_name = f"{base_name}_rotate_{int(angle)}_{i+1}{ext}"
        
        elif args.mode == 'flip':
            aug_img = flipImage(img.copy(), args.flip_dir)
            aug_name = f"{base_name}_flip_{args.flip_dir}_{i+1}{ext}"
        
        elif args.mode == 'zoom':
            zoom_f = args.zoom_factor + (i * 0.1)
            aug_img = zoomImage(img.copy(), zoom_f)
            aug_name = f"{base_name}_zoom_{zoom_f:.1f}_{i+1}{ext}"
        
        elif args.mode == 'brightness':
            bright_f = args.brightness_factor + (i * 0.1 - 0.3)
            aug_img = changeBrightness(img.copy(), max(0.3, bright_f))
            aug_name = f"{base_name}_bright_{bright_f:.1f}_{i+1}{ext}"
        
        elif args.mode == 'noise':
            aug_img = random_noise(img.copy())
            aug_name = f"{base_name}_noise_{i+1}{ext}"
        
        # Save augmented image
        aug_path = output_dir / aug_name
        cv2.imwrite(str(aug_path), aug_img)
        print(f"  [{i+1}/{args.num_aug}] Saved: {aug_path}")
    
    print(f"\n✅ Augmentation complete!")
    print(f"Total images created: {args.num_aug + 1} (1 original + {args.num_aug} augmented)")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
