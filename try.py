#%%
'''
Author: Khush Shah
This file is a Proof of concept file for the testing of OpenCV library.
'''

#%%
'''Loading the libraries'''
import cv2

#Reading the image from the local system
img = cv2.imread("C:/Users/nupur/computer/Desktop/donuts.png")
resized_image = cv2.resize(img, (400, 400))

# Display the resized image
cv2.imshow('Resized Image', resized_image)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)

edges = cv2.Canny(img, 100, 200)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the image
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_image = img.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

# Display the contours
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
import numpy as np
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to get non-contoured areas
non_contoured_area = cv2.bitwise_and(img, img, mask=mask_inv)

# Example: Convert non-contoured areas to grayscale (or any other processing)
non_contoured_area_gray = cv2.cvtColor(non_contoured_area, cv2.COLOR_BGR2GRAY)

# Display the non-contoured area
cv2.imshow('Non-Contoured Area', non_contoured_area_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
