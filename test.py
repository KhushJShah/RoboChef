import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def load_model(weights_path):
    """
    Load the YOLO model with the specified weights.
    """
    model = YOLO(weights_path)
    return model

def predict_and_visualize(model, image_path):
    """
    Perform prediction on the given image and visualize the results.
    """
    # Perform prediction
    results = model(image_path)

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Visualize results
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.names
        scores = result.boxes.conf.cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            caption = f"{label} {score:.2f}"
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            ax.text(x1, y1, caption, color='red', fontsize=12, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

    plt.show()

if __name__ == '__main__':
    # Define the paths
    weights_path = 'C:/Users/Computer vision/Desktop/RoboChef/runs/detect/train16/weights/best.pt'
    test_image_path = 'E:/RoboChef/clove_cinnamon1.jpg'  

    # Load the model
    model = load_model(weights_path)

    # Perform prediction and visualize results
    predict_and_visualize(model, test_image_path)
