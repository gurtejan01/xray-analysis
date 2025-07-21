import os
import gdown

MODEL_PATH = 'unet_epoch_30.pth'
MODEL_URL = 'https://drive.google.com/uc?id=11GnEWwB0HWViMY2zTOuvgBYDc-MY0-3U'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists.")

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from inference import generate_output_image  # uses the inference.py function

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    original_image_url = None
    output_image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate output image (overlayed anomaly heatmap)
            output_filename = 'output_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            generate_output_image(filepath, output_path)

            original_image_url = url_for('static', filename='uploads/' + filename)
            output_image_url = url_for('static', filename='uploads/' + output_filename)

    return render_template('index.html',
                           original_image=original_image_url,
                           output_image=output_image_url)

if __name__ == '__main__':
    download_model()  # Download model if not present
    app.run(debug=True)
