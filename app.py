from ultralytics import YOLO
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
import torch
from urllib.request import urlretrieve

# Replace with the actual URL of your model weights
model_url = "https://github.com/SumayyahAlbarakati/Violence-Detector/blob/main/best.pt"

# Download the model weights
try:
    model_filename = "weights.pt"  # Temporary filename for download
    urlretrieve(model_url, model_filename)
    print(f"Model weights downloaded to: {model_filename}")
except Exception as e:
    print(f"Error downloading model weights: {e}")
    exit(1)  # Or raise an exception if desired

# Load the model using torch.hub.load (adapt if needed)
model = YOLO(model_filename)  # Replace YOLO with your actual model class
print("Model loaded successfully!")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Get the uploaded image
    file = request.files['image']
    if file:
      # Read the image using OpenCV
      img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

      # Perform object detection
      results = model(img)

    detections = []
    for r in results:
      boxes = r.boxes
      for box in boxes:
        b = box.xyxy[0]  # Get box coordinates
        c = box.cls  # Get class ID
        conf = box.conf  # Get confidence score

        detections.append({
          'name': model.names[int(c)],
          'confidence': float(conf),
          'bbox': [int(x) for x in b]  # Convert to integers
        })
        # Extract coordinates from b
        x_min, y_min, x_max, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        label = f"{model.names[int(c)].title()}: {float(conf):.2f}"
        
        # Draw the rectangle
        if(model.names[int(c)]=="violence"):
          color = (0, 0, 255)
        else:
          color = (0, 255, 0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)  # Green rectangle, thickness 2
        
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x_min
        text_y = y_min - text_size[1] - 2  # Adjust y-coordinate for text position
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Encode the annotated image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Render the template with detections and image data
    return render_template('index.html', image=img_base64, detections=detections)

  # Render the template for uploading the image
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
