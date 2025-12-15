"""
Check and remove corrupted images from dataset
"""
import os
from PIL import Image
from pathlib import Path
import argparse

def check_images(data_dir):
    """Check all images in directory and remove corrupted ones"""
    corrupted = []
    total = 0
    
    print(f"Scanning images in: {data_dir}")
    print("="*60)
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Try to open and verify
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    # Reopen and check if loadable
                    with Image.open(file_path) as img:
                        img.load()
                        
                except Exception as e:
                    print(f"Corrupted: {file_path}")
                    print(f"  Error: {e}")
                    corrupted.append(file_path)
    
    print("\n" + "="*60)
    print("SCAN RESULTS")
    print("="*60)
    print(f"Total images scanned: {total}")
    print(f"Corrupted images found: {len(corrupted)}")
    
    if corrupted:
        print("\nCorrupted images:")
        for path in corrupted:
            print(f"  - {path}")
        
        response = input("\nDelete corrupted images? (y/n): ")
        if response.lower() == 'y':
            for path in corrupted:
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
            print(f"\nRemoved {len(corrupted)} corrupted images")
    else:
        print("\nNo corrupted images found! Dataset is clean.")
    
    return corrupted

def main():
    parser = argparse.ArgumentParser(description='Check dataset for corrupted images')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Directory to check')
    args = parser.parse_args()
    
    check_images(args.data_dir)

if __name__ == '__main__':
    main()
