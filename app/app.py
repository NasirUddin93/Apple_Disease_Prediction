from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/apple_model.h5'
model = load_model(MODEL_PATH)

# Define class names (adjust according to your dataset)
class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!'

    file = request.files['file']

    if file.filename == '':
        return 'No image selected for uploading!'

    # Save the uploaded file
    file_path = os.path.join('app/static/uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Return result
    return render_template('index.html',
                           prediction_text=f'Predicted Class: {predicted_class}',
                           image_path=file_path)


# Run app
if __name__ == '__main__':
    app.run(debug=True)
