from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(weights_path):
    """
    Load the YOLO model with the specified weights.
    """
    model = YOLO(weights_path)
    return model

def predict_and_save(model, image_folder, output_folder):
    """
    Perform prediction on the images in the specified folder and save the results.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            
            # Perform prediction
            results = model.predict(image_path, save=False, imgsz=640, conf=0.25)
            
            # Load the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.axis('off')
            
            # Draw bounding boxes, labels, and sequence numbers
            sequence_number = 1
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.names
                scores = result.boxes.conf.cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = map(int, box)
                    caption = f"{sequence_number}. {label} {score:.2f}"
                    sequence_number += 1
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
                    ax.text(x1, y1, caption, color='red', fontsize=12, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

            # Save the result
            output_path = os.path.join(output_folder, image_name)
            plt.savefig(output_path)
            plt.close(fig)

if __name__ == '__main__':
    # Define the paths
    weights_path = 'C:/Users/Computer vision/Desktop/RoboChef/runs/detect/train16/weights/best.pt'
    # Folder with images for inference
    image_folder = 'E:/spices.v2i.yolov8'
    
    # Output folder for results
    output_folder = 'E:/spices.v2i.yolov8/results1'

    # Load the model
    model = load_model(weights_path)

    # Perform prediction and save results
    predict_and_save(model, image_folder, output_folder)
