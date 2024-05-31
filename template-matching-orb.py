#%%
'''
This python file contains a different approach to the spice detection. 
The approach is to first query the ingredient name in the image dataset or the file, and then match it with the spices in the box. 
If detected, then create a square box around it to detect the slot from whre it needs to be extracted.
'''

#%%
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
#%%
def find_ingredient_image(ingredient_name, dataset_path):
    for file in os.listdir(dataset_path):
         if ingredient_name.lower() in file.lower() and (file.endswith(".jpg") or file.endswith(".png")):
            return cv2.imread(os.path.join(dataset_path, file))
    return None

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


#%%

img = cv2.imread('masala_box2.jpg')
ingredient_name = 'turmeric'
dataset_path='C:/Users/nupur/computer/Desktop/RoboChef/RoboChef/'
query = find_ingredient_image(ingredient_name,dataset_path)
if query is None:
    print('Ingredient not found')
    exit()



# %%
# Create ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(query, None)

# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

best_match = matches[0]
best_kp = kp1[best_match.queryIdx].pt

# Draw bounding box around the best match
x, y = int(best_kp[0]), int(best_kp[1])
box_size = 50  # Size of the bounding box
cv2.rectangle(img, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 1)

# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# %%
good_matches = [m for m in matches if m.distance < 20]

# Extract keypoints from the good matches
good_kp = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.int32)

# Calculate the bounding box that encompasses all the best keypoints
if len(good_kp) > 0:
    x, y, w, h = cv2.boundingRect(good_kp)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# %%


# %%
