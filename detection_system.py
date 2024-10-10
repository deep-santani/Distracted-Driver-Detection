import torch
from matplotlib import pyplot as plt
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load and preprocess image
img = Image.open('image.jpg')

# Perform inference
results = model(img)

# Display results
results.show()

# Print detection details
print(results.pandas().xyxy[0])
