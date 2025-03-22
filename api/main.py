import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "*",  # Allow all origins for deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

# Try different model formats until one works
model_paths = [
    "saved_models/Keras1.keras",
    "saved_models/simple_model.keras",
    "saved_models/simple_model.h5",
    "saved_models/simple_model"
]

Kearas_MODEL = None

for model_path in model_paths:
    try:
        print(f"Attempting to load model from {model_path}")
        if model_path.endswith(".keras") or model_path.endswith(".h5"):
            Kearas_MODEL = tf.keras.models.load_model(model_path)
        else:
            Kearas_MODEL = tf.saved_model.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        break
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")

# If all loading attempts failed, use fallback model
if Kearas_MODEL is None:
    print("All model loading attempts failed. Using fallback model")
    # Create a simple model directly
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    Kearas_MODEL = tf.keras.Model(inputs=inputs, outputs=outputs)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running. Use /predict endpoint to analyze plant leaves."}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        
        # Resize image if needed
        if image.shape[0] != 256 or image.shape[1] != 256:
            from PIL import Image
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((256, 256))
            image = np.array(pil_image)
        
        img_batch = np.expand_dims(image, 0)
        
        # Normalize image
        img_batch = img_batch / 255.0
        
        prediction = Kearas_MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {
            "error": str(e),
            "class": "Error",
            "confidence": 0.0
        }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # Use 8080 for Cloud Run
    
    # Display clear message about server running
    print(f"\n{'='*50}")
    print(f"🌱 Plant Disease Detection API Server is starting up!")
    print(f"{'='*50}")
    print(f"Server URL: http://0.0.0.0:{port}")
    print(f"To test the API locally: http://localhost:{port}/docs")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'='*50}\n")
    
    # Make sure to bind to 0.0.0.0, not 127.0.0.1
    uvicorn.run(app, host="0.0.0.0", port=port)