from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and class mapping
model = load_model('best_model.keras')
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Reverse dictionary: index to label
labels = {v: k for k, v in class_indices.items()}


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(300, 300))  # Resize to 300x300
    img_array = image.img_to_array(img) / 255.0             # Normalize
    img_array = np.expand_dims(img_array, axis=0)           # Shape: (1, 300, 300, 3)
    
    pred = model.predict(img_array)
    pred_class = np.argmax(pred[0])                         # Get predicted class index
    return labels[pred_class]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = model_predict(filepath)
            return render_template("index.html", prediction=prediction, image=file.filename)
        else:
            return "No file selected."
    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
