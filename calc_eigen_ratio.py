import numpy as np
import cv2

from main import get_main_contour
from classifier import ShotAngleClassifier


sac = ShotAngleClassifier()
contour = get_main_contour('../ray_seg.png')
sz = len(contour)
data_pts = np.empty((sz, 2), dtype=np.float64)

for i in range(data_pts.shape[0]):
    data_pts[i, 0] = contour[i, 0, 0]
    data_pts[i, 1] = contour[i, 0, 1]
# Perform PCA analysis
mean = np.empty((0))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
# Store the center of the object
cntr = (int(mean[0, 0]), int(mean[0, 1]))
p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
      cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
      cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

dist1 = np.sqrt((cntr[0] - p1[0]) ** 2 + (cntr[1] - p1[1]) ** 2)
dist2 = np.sqrt((cntr[0] - p2[0]) ** 2 + (cntr[1] - p2[1]) ** 2)

print(dist1/dist2)


