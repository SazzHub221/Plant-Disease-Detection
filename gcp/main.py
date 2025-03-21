from flask import Flask, request,jsonify
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = 'plant-disease-model-2'
class_names = ["Early Blight", "Late Blight", "Healthy"]

model = None

app = Flask(__name__)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

@app.route("/", methods=["POST"])
def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/Keras1.keras",
            "/tmp/Keras1.keras"
        )
        print("Model downloaded to /tmp/Keras1.keras")
        model = tf.keras.models.load_model("/tmp/Keras1.keras")

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image / 255
    img_array = np.expand_dims(image, axis=0)

    prediction = model.predict(img_array)
    print(prediction)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * (np.max(prediction)), 2)

    return jsonify({"class": predicted_class, "confidence": confidence})