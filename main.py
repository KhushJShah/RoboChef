from ultralytics import YOLO
import torch

def main():
    # Load the pre-trained model
    model = YOLO('c:/Users/Computer vision/Desktop/RoboChef/runs\detect/train13/weights/best.pt')
    yaml_path = 'E:/RoboChef/spice_detection_dataset_yolo1/spice_detection_dataset_yolo1/data1.yaml'

    # Determine the device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the model
    model.train(data=yaml_path, epochs=20, imgsz=640, device=device)

    # Evaluate the model
    results = model.val()
    print(results)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
