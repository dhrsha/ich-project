import os
import numpy as np
from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import io

app = Flask(__name__)

class_labels = {
    0: 'non-hemorrhage',
    1: 'hemorrhage'
}

# Load the Keras model
model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request has file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img = img.resize((150, 150))  # Assuming model input size is 150x150
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Get prediction label and probability
    predicted_class = np.argmax(predictions[0])
    class_label = class_labels[predicted_class]
    probability = predictions[0][predicted_class]

    # Return prediction result as JSON response
    return jsonify({
        'class_label': class_label,
        'probability': float(probability)  # Convert to float
    })

if __name__ == '__main__':
    app.run(debug=True)
