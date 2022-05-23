from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time
from Exatract_Features import *
from Finding_cars import *
# Get test image
test_img = mpimg.imread('./test_images/test4.jpg')

# Set parameters for find_cars function
ystart = 400
ystop = 660
scale = 1.5
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

# Get output rectangles surrounding the cars we found
rectangles = find_cars(test_img, ystart, ystop, scale, colorspace,
                       hog_channel, svc, None, orient, pix_per_cell,
                       cell_per_block, None, None)

# Print home many rectangles we found in the image
print(len(rectangles), 'potential cars found in image')

# Draw boxes where cars are located in test image
test_img_rects = draw_boxes(test_img, rectangles)

# Plot the new image
plt.figure(figsize=(10,10))
plt.imshow(test_img_rects)