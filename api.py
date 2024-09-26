import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import base64

# Load the trained model
model = load_model('object_classifier_model.keras')

# Define constants
IMAGE_SIZE = (384, 512)  # Swap the dimensions

# Create a Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Function to preprocess the image
def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(f"Image shape after preprocessing: {img_array.shape}")  # Debugging information
    return img, img_array

# Function to encode the image as a base64 string
def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Define the prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    image_url = request.args.get('image_url')

    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        img, img_array = preprocess_image(image_url)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Assuming you have the class indices from the training data generator
        class_indices = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
        predicted_label = class_indices[predicted_class]

        # Encode the image as a base64 string
        img_base64 = encode_image_to_base64(img)

        return jsonify({'predicted_label': predicted_label, 'image_preview': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
