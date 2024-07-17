from ultralytics import YOLO
import torch

def main():
    # Load the pre-trained model
    model = YOLO('yolov8n.pt')
    yaml_path = 'E:/RoboChef/data.yaml'

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
