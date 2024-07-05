from ultralytics import YOLO # type: ignore

# Load the pre-trained model
model = YOLO('runs/detect/train3/weights/best.pt')
yaml_path = 'E:/RoboChef/spice_detection_dataset_yolo1/spice_detection_dataset_yolo1/data1.yaml'
# Train the model
model.train(data=yaml_path, epochs=20, imgsz=640)

# Evaluate the model
results = model.val()
print(results)
