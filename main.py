from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

app = Flask(__name__)

# Load the models once at the beginning
models = {
    'lenet5': load_model('50lenet5.keras'),
    'ran2dev': load_model('50ran2dev.keras')
}

# Class labels for Ranjana script
class_labels = [
    'अ', 'आ', 'अ:', 'ऐ', 'अं', 'औ', 'ब', 'भ', 'च', 'छ', 'ड',
    'द', 'ध', 'ढ', 'ए', '८', '५', '५', 'ग', 'घ', 'ज्ञ',
    'ह', 'इ', 'ई', 'ज', 'झ', 'क', 'ख', 'क्ष', 'ल', 'लृ', 'lrii',
    'म', 'न,', '९', 'ण', 'न', 'ञ', 'ओ', '१', 'प', 'फ',
    'र', 'ऋ', 'rii', 'ष', 'स', '७', 'श', '६', 'ट', 'ठ',
    '३', 'त्र', 'त', 'थ', '२', 'उ', 'ऊ', 'व', 'य', '०',
]

def prepare_image(image, model_name):
    # Preprocess the image: resize, convert to grayscale, and normalize
    image = Image.open(image)
    
    # Convert image to grayscale
    image = image.convert('L')
    
    if model_name == 'ran2dev':
        # Resize the image to 64x64 for ran2dev
        image = image.resize((64, 64))
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Ensure it's in the correct shape for the model (64, 64, 1)
        image = np.reshape(image, (1, 64, 64, 1))  # Grayscale input shape for ran2dev
    else:
        # Process image for lenet5 (grayscale, 32x32)
        image = image.resize((32, 32))  # Resize to 32x32 for lenet5
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.reshape(image, (1, 32, 32, 1))  # Grayscale input for lenet5

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selected"})

    image_file = request.files['image']
    model_name = request.form['model']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"})

    if model_name not in models:
        return jsonify({"error": "Invalid model selected"})

    try:
        # Prepare the image
        image = prepare_image(image_file, model_name)

        # Load the selected model and predict
        model = models[model_name]
        prediction = model.predict(image)

        # Log predictions
        print(f"Raw predictions: {prediction}")

        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100  # Confidence score in percentage

        # Get the corresponding label
        predicted_label = class_labels[predicted_class]

        # Return the predicted character and confidence score
        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        # Log any errors
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred: {e}"})

if __name__ == '__main__':
    app.run(debug=True)
