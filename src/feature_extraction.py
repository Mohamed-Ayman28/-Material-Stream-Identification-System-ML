"""
# feature_extraction.py
Enhanced feature extraction for material classification including:
- HOG (Histogram of Oriented Gradients) - captures shape/texture
- Color Histogram (HSV + RGB) - captures color distribution  
- LBP (Local Binary Patterns) - captures micro-textures
- Edge features - captures material boundaries
- Statistical features - captures intensity distribution
- GLCM texture features - captures spatial relationships
- Gabor filters - captures texture at multiple orientations
- Improved preprocessing and normalization for robustness
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray


##############################
# Image Preprocessing
##############################
def preprocess_image(img):
    """
    Preprocess image for robust feature extraction.
    - Normalize lighting
    - Enhance contrast
    """
    # Convert to LAB color space for better lighting normalization
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img_normalized


##############################
# HOG Features (Texture/Shape) - Enhanced
##############################
def extract_hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    """
    Extract Histogram of Oriented Gradients features.
    Adjusted parameters for better material discrimination.
    """
    gray = rgb2gray(img)
    
    # Normalize the grayscale image
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    fd = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True,
    )
    return fd


##############################
# Enhanced Color Features
##############################
def extract_color_features(img, bins=16):
    """
    Extract color features from both RGB and HSV color spaces.
    HSV is especially good for material classification as it separates
    color from intensity (important for materials under different lighting).
    """
    # RGB histograms
    rgb_hist = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        rgb_hist.append(hist)
    rgb_hist = np.concatenate(rgb_hist)
    
    # HSV histograms (better for material properties)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_hist = []
    for i in range(3):
        if i == 0:  # Hue channel (0-180)
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180])
        else:  # Saturation and Value (0-256)
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hsv_hist.append(hist)
    hsv_hist = np.concatenate(hsv_hist)
    
    # Color moments (mean, std, skewness for each channel)
    color_moments = []
    for i in range(3):
        channel = img[:, :, i]
        color_moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean(np.abs(channel - np.mean(channel))**3)**(1/3)  # skewness
        ])
    
    # Combine all color features
    color_features = np.concatenate([rgb_hist, hsv_hist, color_moments])
    color_features = color_features.astype('float32')
    
    # Normalize
    if np.sum(color_features) > 0:
        color_features /= (np.linalg.norm(color_features) + 1e-8)
    
    return color_features


##############################
# LBP Texture Features (Faster single-scale)
##############################
def extract_lbp_multiscale(img):
    """
    Extract Local Binary Pattern features (optimized for speed).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Single scale for speed
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float32')
    hist /= (hist.sum() + 1e-8)
    
    return hist


##############################
# Edge and Gradient Features
##############################
def extract_edge_features(img):
    """
    Extract edge-based features. Materials have different edge characteristics:
    - Glass: smooth edges, high reflectivity gradients
    - Paper: rough edges, low contrast
    - Metal: sharp edges, high contrast
    - Cardboard: rough, corrugated texture edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude and direction
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_dir = np.arctan2(sobely, sobelx)
    
    # Edge statistics
    edge_features = [
        np.mean(edge_mag),
        np.std(edge_mag),
        np.max(edge_mag),
        np.percentile(edge_mag, 75),
        np.percentile(edge_mag, 90)
    ]
    
    # Canny edge density (measure of edge strength)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    edge_features.append(edge_density)
    
    return np.array(edge_features, dtype='float32')


##############################
# Statistical Texture Features
##############################
def extract_statistical_features(img):
    """
    Extract statistical features that capture intensity distribution.
    Different materials reflect light differently.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')
    
    features = [
        np.mean(gray),           # Average brightness
        np.std(gray),            # Contrast
        np.var(gray),            # Variance
        np.min(gray),            # Darkest point
        np.max(gray),            # Brightest point
        np.median(gray),         # Median intensity
        np.percentile(gray, 25), # 1st quartile
        np.percentile(gray, 75), # 3rd quartile
    ]
    
    # Entropy (measure of randomness/texture complexity)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    features.append(entropy)
    
    return np.array(features, dtype='float32')


##############################
# Gabor Filter Features (Optimized)
##############################
def extract_gabor_features(img, num_orientations=4, num_frequencies=2):
    """
    Extract Gabor filter features (optimized for speed).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    # Downsample for speed
    gray_small = cv2.resize(gray, (64, 64))
    
    gabor_features = []
    
    # Fewer frequencies for speed
    frequencies = [0.1, 0.25]
    
    for freq in frequencies:
        for theta in range(num_orientations):
            theta_rad = theta / float(num_orientations) * np.pi
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (11, 11),  # Smaller kernel
                sigma=2.0, 
                theta=theta_rad, 
                lambd=1.0/freq, 
                gamma=0.5, 
                psi=0
            )
            
            # Apply filter
            filtered = cv2.filter2D(gray_small, cv2.CV_32F, kernel)
            
            # Extract statistics
            gabor_features.extend([
                np.mean(filtered),
                np.std(filtered)
            ])
    
    return np.array(gabor_features, dtype='float32')


##############################
# Material-Specific Features
##############################
def extract_material_specific_features(img):
    """
    Extract features specifically useful for material classification:
    - Reflectivity/shininess indicators
    - Texture roughness
    - Transparency indicators
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    features = []
    
    # 1. Shininess indicator (high local variance in bright areas)
    bright_mask = gray > np.percentile(gray, 75)
    if np.sum(bright_mask) > 0:
        shininess = np.std(gray[bright_mask])
    else:
        shininess = 0
    features.append(shininess)
    
    # 2. Texture roughness (gradient variation)
    roughness = np.std(magnitude) / (np.mean(magnitude) + 1e-8)
    features.append(roughness)
    
    # 3. Local contrast (good for distinguishing smooth vs rough materials)
    local_contrast = np.mean(cv2.Laplacian(gray, cv2.CV_32F)**2)
    features.append(local_contrast)
    
    # 4. Saturation features (colorfulness)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype('float32')
    features.extend([
        np.mean(saturation),
        np.std(saturation),
        np.percentile(saturation, 90)
    ])
    
    # 5. Value (brightness) distribution
    value = hsv[:, :, 2].astype('float32')
    features.extend([
        np.mean(value),
        np.std(value),
        np.max(value) - np.min(value)  # Dynamic range
    ])
    
    return np.array(features, dtype='float32')
def extract_glcm_features(img):
    """
    Extract Gray-Level Co-occurrence Matrix features.
    GLCM captures spatial relationships in texture - excellent for materials.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Normalize to 0-255 and convert to uint8
    gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype('uint8')
    
    # Compute GLCM at different angles
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    
    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = []
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        glcm_features.extend([
            np.mean(values),
            np.std(values)
        ])
    
    return np.array(glcm_features, dtype='float32')


##############################
# Combined Feature Vector - Optimized
##############################
def extract_features(img, hog_params=None, use_preprocessing=True):
    """
    Extract comprehensive but optimized feature vector.
    Features are selected for maximum discriminative power.
    
    Args:
        img: Input RGB image
        hog_params: Optional HOG parameters
        use_preprocessing: Whether to apply preprocessing (recommended)
    
    Returns:
        Feature vector (numpy array)
    """
    if hog_params is None:
        hog_params = {}
    
    # Preprocess image for robustness
    if use_preprocessing:
        img_processed = preprocess_image(img)
    else:
        img_processed = img
    
    # Extract all features (optimized for speed and performance)
    f_hog = extract_hog(img_processed, **hog_params)           # ~600 features  
    f_color = extract_color_features(img_processed, bins=8)    # ~50 features
    f_lbp = extract_lbp_multiscale(img_processed)              # ~10 features
    f_edge = extract_edge_features(img_processed)              # 6 features
    f_stats = extract_statistical_features(img_processed)      # 9 features
    f_glcm = extract_glcm_features(img_processed)              # 10 features
    f_gabor = extract_gabor_features(img_processed)            # 16 features
    f_material = extract_material_specific_features(img_processed)  # 9 features
    
    # Combine all features
    features = np.concatenate([
        f_hog, 
        f_color, 
        f_lbp, 
        f_edge, 
        f_stats, 
        f_glcm,
        f_gabor,
        f_material
    ])
    
    # Final normalization for numerical stability
    features = features.astype('float32')
    
    # L2 normalization
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features


def build_feature_matrix(images, hog_params=None):
    """Build feature matrix for multiple images."""
    features = []
    for img in images:
        feat = extract_features(img, hog_params)
        features.append(feat)
    return np.array(features)