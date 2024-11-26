import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify
import requests
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'object_classifier_model.pth'
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features

# Load the state dictionary to determine the number of classes
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
num_classes = state_dict['classifier.3.weight'].size(0)

# Define the classifier layer with the correct number of classes
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes)
)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()

# Define the image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class names (replace with your actual class names)
class_names = {
    0: 'Cardboard',
    1: 'Glass',
    2: 'Metal',
    3: 'Paper',
    4: 'Plastic',
    5: 'Trash',
    # Add more class names as needed
}

# Function to fetch and preprocess the image
def fetch_and_preprocess_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    else:
        return None

# Function to make a prediction
def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return preds.item()

# Function to send a request to move the servo
def move_servo(servo_degree):
    url = f'http://192.168.0.100/servo?servo={servo_degree}'
    response = requests.get(url)
    return response.status_code

# Define the prediction endpoint
@app.route('/predict', methods=['GET'])
def predict_image():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({'error': 'Image URL is required'}), 400

    image = fetch_and_preprocess_image(image_url)
    if image is None:
        return jsonify({'error': 'Failed to fetch the image'}), 500

    predicted_label_index = predict(image)
    predicted_label_name = class_names.get(predicted_label_index, 'Unknown')

    # Convert the image to base64 for preview
    response = requests.get(image_url)
    image_preview = base64.b64encode(response.content).decode('utf-8')

    # Move the servo based on the predicted label index
    if(predicted_label_index == 0):
        servo_status = move_servo(15)
    else:
        servo_status = move_servo(predicted_label_index * 15 * 2)

    return jsonify({
        'image_preview': image_preview,
        'predicted_label': predicted_label_name,
        # 'servo_status': servo_status,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
