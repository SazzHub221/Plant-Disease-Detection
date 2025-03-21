import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from mangum import Handler

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Kearas_MODEL = tf.keras.models.load_model("saved_models/Keras1.keras")
# Saved_MODEL = tf.saved_model.load("D:\Capstone\Project\saved_models\Saved1")
CLASS_NAMES = ["Early Blight","Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = Kearas_MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence  = np.max(prediction[0])
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

# In main.py, update the last part:
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

handler = Handler(app)