from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import glob

app = Flask(__name__)

# Überprüfen, ob das Verzeichnis 'uploads' existiert und erstellen, falls nicht
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funktion zum Bereinigen des Upload-Ordners
def clean_uploads():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for file in files:
        try:
            os.remove(file)
            print(f'{file} wurde gelöscht.')
        except Exception as e:
            print(f'Fehler beim Löschen von {file}: {e}')

# Modell laden
model = load_model('final_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file"
    
    file = request.files['file']
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        prediction_class = "malignant" if prediction[0][0] > 0.5 else "benign"
        
        return render_template('result.html', prediction=prediction_class)

if __name__ == '__main__':
    # Bereinigen des Upload-Ordners vor dem Start der App
    clean_uploads()
    app.run(debug=True)
