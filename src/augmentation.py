##  purpose of this file is to Contains all image augmentation functions used to expand and balance the dataset.
## Supports: rotation, flipping, zoom, brightness change, noise, and combined augmentation.


import cv2
import numpy as np
import random

def rotateImage(image,angle):
    hieght, width = image.shape[:2]
    matrix= cv2.getRotationMatrix2D((width/2, hieght/2), angle, 1)
    rotatedImage = cv2.warpAffine(image, matrix, (width, hieght))   
    return rotatedImage


def flipImage(image, mode):
    if mode == 'horizontal':
        return cv2.flip(image, 1)
    elif mode == 'vertical':
        return cv2.flip(image, 0)
    elif mode == 'both':
        return cv2.flip(image, -1)
    else:
        raise ValueError("Mode should be 'horizontal', 'vertical', or 'both'")


def zoomImage(image, zoom_factor):
    if zoom_factor < 1.0:
        raise ValueError("Zoom factor should be greater than 1.0")
    
    height, width = image.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    
    y1 = (height - new_height) // 2
    x1 = (width - new_width) // 2
    y2 = y1 + new_height
    x2 = x1 + new_width
    
    cropped_image = image[y1:y2, x1:x2]
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_image

def changeBrightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,2] = hsv[:,:,2]*factor
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image


def addNoise(img):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def random_noise(img):
    """Alias for addNoise"""
    return addNoise(img)



def combinedAugmentation(image):
    # Random rotation between -30 to 30 degrees
    angle = random.uniform(-30, 30)
    image = rotateImage(image, angle)
    
    # Random flip
    flip_modes = ['horizontal', 'vertical', 'both', None]
    mode = random.choice(flip_modes)
    if mode:
        image = flipImage(image, mode)
    
    # Random zoom between 1.0 to 1.5
    zoom_factor = random.uniform(1.0, 1.5)
    image = zoomImage(image, zoom_factor)
    
    # Random brightness change between 0.5 to 1.5
    brightness_factor = random.uniform(0.5, 1.5)
    image = changeBrightness(image, brightness_factor)
    
    # Add random noise
    image = random_noise(image)
    
    return image