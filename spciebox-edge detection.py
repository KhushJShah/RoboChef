#%%
'''This python file analyzes the slots or the mini-boxes from the masala box'''

#%%
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('masala box.JPG')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()


# %%
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_2GRAY)

edges = cv2.Canny(gray_image, 100,200)

plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
#%%
plt.imshow(gray_image)
plt.axis('off')
plt.show()
# %%
# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Convert to RGB for displaying using matplotlib
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

# Display the image with contours
plt.imshow(contour_image_rgb)
plt.axis('off')
plt.show()

# %%
# Initialize list to store slot contours
slot_contours = []

# Filter contours based on area and shape
for contour in contours:
    area = cv2.contourArea(contour)
    if 1000 < area < 5000:  # Adjust the area range as needed
        # Approximate the contour to a polygon and check if it's roughly circular
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) > 5:  # Circular shapes have more points
            slot_contours.append(contour)

# Draw slot contours
slot_contour_image = image.copy()
cv2.drawContours(slot_contour_image, slot_contours, -1, (0, 255, 0), 2)

# Convert to RGB for displaying using matplotlib
slot_contour_image_rgb = cv2.cvtColor(slot_contour_image, cv2.COLOR_BGR2RGB)

# Display the image with slot contours
plt.imshow(slot_contour_image_rgb)
plt.axis('off')
plt.show()

# %%
# Analyze each slot
import numpy as np
for i, contour in enumerate(slot_contours):
    # Create a mask for the current slot
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the slot using the mask
    slot = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop the slot to its bounding box
    x, y, w, h = cv2.boundingRect(contour)
    slot_cropped = slot[y:y+h, x:x+w]

    # Convert to RGB for displaying using matplotlib
    slot_cropped_rgb = cv2.cvtColor(slot_cropped, cv2.COLOR_BGR2RGB)

    # Display the slot
    plt.imshow(slot_cropped_rgb)
    plt.axis('off')
    plt.title(f'Slot {i+1}')
    plt.show()

    # Further analysis for identifying masala can be added here
    # For example, using color histograms, texture analysis, or a pre-trained classifier

# %%
circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=20, maxRadius=30)

# Ensure at least some circles were found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Draw circles on the image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

    # Display the image with circles
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
# %%
# Ensure at least some circles were found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Draw squares on the image
    for (x, y, r) in circles:
        cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)

    # Display the image with squares
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# %%
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

# Canny Edge Detection
edges = cv2.Canny(blurred_image, 100, 200)

# Hough Circle Transform
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=50, minRadius=20, maxRadius=60)

# Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # Draw the rectangle bounding the circle
        cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)

# Convert to RGB for displaying using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with detected circles
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
# %%
