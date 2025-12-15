"""
Remove corrupted images from dataset automatically
"""
import os
from PIL import Image
from pathlib import Path

def remove_corrupted_images(data_dir):
    """Check all images and remove corrupted ones"""
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
                        
                except Exception:
                    corrupted.append(file_path)
    
    print(f"\nFound {len(corrupted)} corrupted images out of {total} total")
    
    # Delete corrupted images
    deleted = 0
    for path in corrupted:
        try:
            os.remove(path)
            deleted += 1
            print(f"Deleted: {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
    
    print("\n" + "="*60)
    print(f"Removed {deleted} corrupted images")
    print(f"Remaining clean images: {total - deleted}")
    print("="*60)
    
    return deleted

if __name__ == '__main__':
    remove_corrupted_images('dataset')
