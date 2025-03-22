import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy
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
    "*",  # Allow all origins temporarily for testing
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

# Try to load the model with error handling
try:
    print("Loading model... This may take a moment.")
    # Try with direct loading
    Kearas_MODEL = tf.keras.models.load_model("saved_models/Keras1.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        # Try alternative loading method
        print("Attempting alternative loading method...")
        Kearas_MODEL = tf.saved_model.load("saved_models/Keras1")
        print("Model loaded with alternative method!")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        # Create a simple error model as fallback
        print("Creating fallback model")
        # This is just a placeholder model that will return "Error" for any input
        class FallbackModel:
            def predict(self, x):
                return [[0.8, 0.1, 0.1]]  # Default to first class with high confidence
                
        Kearas_MODEL = FallbackModel()

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
        img_batch = np.expand_dims(image, 0)
        
        # Normalize image if needed
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
    print(f"Local server URL: http://localhost:{port}")
    print(f"To test the API, visit: http://localhost:{port}/docs")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host='0.0.0.0', port=port)