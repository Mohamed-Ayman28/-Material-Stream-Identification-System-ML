"""
# feature_extraction.py
Handles feature extraction from raw images including:
- HOG (Histogram of Oriented Gradients)
- Color Histogram
- LBP (Local Binary Patterns)

"""
from matplotlib.pyplot import gray
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from turtle import fd


def extract_hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
 gray = rgb2gray(img)
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
# Color Histogram Features
##############################
def extract_color_hist(img, bins=(8, 8, 8)):
 chans = cv2.split(img)
 hist = []
 h = None
 for ch in chans:
  h = np.histogram(ch, bins=bins[0], range=(0, 256))[0]
 hist.append(h)
 hist = np.concatenate(hist).astype('float32')
 hist /= hist.sum() + 1e-8
 return hist

##############################
# LBP Texture Features
##############################
def extract_lbp(img, P=8, R=1):
 gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 lbp = local_binary_pattern(gray, P, R, method='uniform')
 n_bins = int(lbp.max() + 1)
 hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
 hist = hist.astype('float32')
 hist /= hist.sum() + 1e-8
 return hist

##############################
# Combined Feature Vector
##############################
def extract_features(img, hog_params=None):
 if hog_params is None:
  hog_params = {}

  f1 = extract_hog(img, **hog_params)
  f2 = extract_color_hist(img)
  f3 = extract_lbp(img)
  return np.concatenate([f1, f2, f3])
 
def build_feature_matrix(images, hog_params=None):
 features = []
 for img in images:
  feat = extract_features(img, hog_params)
  features.append(feat)
 return np.array(features)
