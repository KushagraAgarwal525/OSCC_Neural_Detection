from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the pretrained Keras model
model = load_model('model.keras')

def preprocess_image(image_bytes, target_size):

    """Preprocess the image for prediction using OpenCV"""
    # Convert the image bytes to a numpy array
    np_img = np.frombuffer(image_bytes, np.uint8) 
    # Decode the image array to OpenCV format
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Convert the image to an array
    image = img_to_array(image)
    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        image_bytes = file.read()
        # Preprocess the image
        processed_image = preprocess_image(image_bytes, target_size=(300, 300)) 
        
        prediction = model.predict(processed_image).flatten()[0]
        output = float(prediction)

        if output < 0.5:
            result = f"RESULT: {(1-output)*100:.2f}% NEGATIVE"
        else:
            result = f"RESULT: {output*100:.2f}% POSITIVE"

        return jsonify({"message": result}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>', methods=['GET'])
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
