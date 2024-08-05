from ultralytics import YOLO
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64

# Replace with the actual path to your best.pt weights file
model_weights = r"\models_train\best.pt"

# Load the model using torch.hub.load with the custom configuration file
model = YOLO(model_weights)
# prompt: having the yolov8 best.pt file how to make flask code that take input image from user and return detected frame and it's name and confidence level


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
