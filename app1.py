import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask, request, render_template, send_from_directory

# Load both models
brain_tumor_model = load_model('brain_tumor_model.keras')
alzheimers_model = load_model('alzh_detection_model.keras')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

# Class mappings with added index 4
brain_tumor_classes = {
    0: 'Glioma Tumor',
    1: 'Metigloma Tumor',
    2: 'No Tumor',
    3: 'Pituatary Tumor',
    4: 'Invalid Photo'  
}

alzheimers_classes = {
    0: 'Non Dementia',
    1: 'Mild Dementia',
    2: 'Moderate Dementia',
    3: 'Very Mild Dementia',
    4: 'invalid Photo' 
}

# Confidence threshold for determining unknown images
CONFIDENCE_THRESHOLD = 0.9  

# Home route - choose detection type
@app.route('/')
def home():
    return render_template('index1.html')

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'detection_type' not in request.form:
        return 'No file or detection type selected', 400

    file = request.files['file']
    detection_type = request.form['detection_type']

    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read and check if it's a valid image
    img = cv2.imread(file_path)
    if img is None:
        return 'Invalid image file', 400  # Return error if image is not valid

    if detection_type == 'brain_tumor':
        img = cv2.resize(img, (150, 150))  # Brain tumor model expects 150x150 images
        img_array = np.array(img, dtype=np.float32).reshape(1, 150, 150, 3)  # Convert to proper format
        model = brain_tumor_model
        class_mapping = brain_tumor_classes
        condition_label = "Brain Tumor Type"

    elif detection_type == 'alzheimers':
        img = cv2.resize(img, (128, 128))  # Alzheimer's model expects 128x128 images
        img_array = np.array(img, dtype=np.float32).reshape(1, 128, 128, 3)
        model = alzheimers_model
        class_mapping = alzheimers_classes
        condition_label = "Alzheimer's Condition"

    else:
        return 'Invalid detection type', 400

    # Make predictions
    predictions = model.predict(img_array)
    max_prob = predictions.max()  # Get the highest probability
    indices = predictions.argmax()  # Get the index of the highest probability


    if max_prob < CONFIDENCE_THRESHOLD:
        indices = 4

    condition = class_mapping.get(indices, 'Invalid Condition')

    return render_template('result1.html', condition_label=condition_label, condition=condition, image_filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
