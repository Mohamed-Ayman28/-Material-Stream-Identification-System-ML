"""
improve_features.py - Enhanced feature extraction for better material classification
Adds more discriminative features without requiring new data collection
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from scipy import ndimage


def extract_enhanced_features(image):
    """
    Extract enhanced features with better material discrimination
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        1D feature vector
    """
    features = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. ENHANCED COLOR FEATURES (more detailed)
    # RGB color moments
    for channel in range(3):
        ch = image[:, :, channel]
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.median(ch),
            np.percentile(ch, 25),
            np.percentile(ch, 75)
        ])
    
    # HSV color features
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for channel in range(3):
        ch = hsv[:, :, channel]
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.median(ch)
        ])
    
    # 2. TEXTURE FEATURES (critical for material classification)
    
    # Enhanced LBP with multiple radii
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
        features.extend(lbp_hist[:10])  # Top 10 bins
    
    # Enhanced GLCM for texture
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Normalize gray image for GLCM
    gray_norm = (gray / 16).astype(np.uint8)  # Reduce to 16 levels
    
    for distance in distances:
        glcm = graycomatrix(gray_norm, [distance], angles, levels=16, symmetric=True, normed=True)
        
        # Extract multiple GLCM properties
        features.append(graycoprops(glcm, 'contrast')[0, 0])
        features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        features.append(graycoprops(glcm, 'homogeneity')[0, 0])
        features.append(graycoprops(glcm, 'energy')[0, 0])
        features.append(graycoprops(glcm, 'correlation')[0, 0])
    
    # 3. EDGE AND STRUCTURE FEATURES (important for cardboard vs paper)
    
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.mean(edges))
    features.append(np.std(edges))
    features.append(np.sum(edges > 0) / edges.size)  # Edge density
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    features.append(np.mean(gradient_magnitude))
    features.append(np.std(gradient_magnitude))
    features.append(np.max(gradient_magnitude))
    
    # 4. VARIANCE AND ROUGHNESS (cardboard is rougher than paper)
    
    # Local variance (window-based)
    kernel_size = 5
    mean_filter = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    sqr_filter = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = sqr_filter - mean_filter**2
    
    features.append(np.mean(variance))
    features.append(np.std(variance))
    features.append(np.median(variance))
    
    # 5. FREQUENCY DOMAIN FEATURES (FFT)
    
    # FFT to analyze texture frequency
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Radial frequency profile
    h, w = gray.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    for radius in [5, 10, 20, 40]:
        mask = (r >= radius - 2) & (r <= radius + 2)
        features.append(np.mean(magnitude_spectrum[mask]))
    
    # 6. MATERIAL-SPECIFIC FEATURES
    
    # Shininess/Reflectance (plastic and metal are shinier)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    shininess = np.sum(gray > 200) / gray.size
    features.append(shininess)
    
    # Uniformity (paper/cardboard more uniform than trash)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    features.append(entropy)
    
    # Surface roughness indicator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roughness = np.var(laplacian)
    features.append(roughness)
    
    # 7. STATISTICAL MOMENTS
    
    # Higher order moments
    features.append(np.mean((gray - np.mean(gray))**3))  # Skewness
    features.append(np.mean((gray - np.mean(gray))**4))  # Kurtosis
    
    return np.array(features, dtype=np.float32)


def extract_features(image):
    """
    Main feature extraction function (enhanced version)
    Compatible with existing code
    """
    return extract_enhanced_features(image)


if __name__ == '__main__':
    # Test feature extraction
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            features = extract_features(resized)
            print(f'Extracted {len(features)} features')
            print(f'Feature range: [{features.min():.3f}, {features.max():.3f}]')
        else:
            print(f'Could not load image: {sys.argv[1]}')
    else:
        print('Usage: python improve_features.py <image_path>')
