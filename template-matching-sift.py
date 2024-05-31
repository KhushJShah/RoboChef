#%%
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
#%%

#%%
img1 = cv2.imread('cloves.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('masala_box1.jpg',cv2.IMREAD_GRAYSCALE)


#%%
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#%%
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


# %%
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
 if m.distance < 0.75*n.distance:
    good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
# %%


img = cv2.imread('masala_box1.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv2.imread('cloves1.jpg', cv2.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
 img = img2.copy()
 method = eval(meth)
 # Apply template Matching
 res = cv2.matchTemplate(img,template,method)
 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
 # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
 if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
 else:
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

 cv2.rectangle(img,top_left, bottom_right, 255, 2)
 plt.subplot(121),plt.imshow(res,cmap = 'gray')
 plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
 plt.subplot(122),plt.imshow(img,cmap = 'gray')
 plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
 plt.suptitle(meth)
 plt.show()
# %%
