

from flask import *
import json
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import cv2
import base64
app = Flask(__name__)




def image_processing(image):
    img = cv2.imread(image, -1)
    json_file = open('cnn_model_83.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn_model_83.h5")
    print("Loaded model from disk")
    img.resize(120,120,3)
    CLASSES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Hispa', 'Leaf Blast', 'Leaf Smut']
    img = np.array(img).reshape(-1, 120, 120, 3)
    prediction = loaded_model.predict(img)
    classes = np.argmax(prediction,axis=1)
    print(prediction)
    return CLASSES[classes[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        result = image_processing(file_path)
        return result   
    return None

if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0')
