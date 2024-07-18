from ultralytics import YOLO
import os

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
            
            # Perform prediction and save the results
            model.predict(image_path, save=True, imgsz=640, conf=0.5, project=output_folder, name='results', exist_ok=True)



if __name__ == '__main__':
    # Define the paths
    weights_path = 'C:/Users/Computer vision/Desktop/RoboChef/runs/detect/train16/weights/best.pt'
    # Folder with images for inference
    image_folder = 'E:/spices.v2i.yolov8'
    
    # Output folder for results
    output_folder = 'E:/spices.v2i.yolov8/results'

    # Load the model
    model = load_model(weights_path)

    # Perform prediction and save results
    predict_and_save(model, image_folder, output_folder)
