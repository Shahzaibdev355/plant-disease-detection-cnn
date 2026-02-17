import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from groq import Groq
from dotenv import load_dotenv



# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)  # Allow React frontend

# Load environment variables
load_dotenv()


# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Load Model Once (IMPORTANT)
# -----------------------------
MODEL_PATH = "cnn_model.h5"
LABEL_PATH = "label_transform.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
label_binarizer = pickle.load(open(LABEL_PATH, "rb"))

IMAGE_SIZE = (128, 128)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predictions = model.predict(image)[0]
    class_index = np.argmax(predictions)
    confidence = float(predictions[class_index] * 100)
    class_label = label_binarizer.classes_[class_index]

    return class_label, confidence


# -----------------------------
# API Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Plant Disease API is running 🚀"
    })



@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Convert file to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    label, confidence = predict_image(image)

    description = generate_description(label)

    return jsonify({
        "disease": label,
        "confidence": round(confidence, 2),
        "description": description
    })



def generate_description(disease_name):
    try:
        prompt = f"""
        The plant disease detected is: {disease_name}.
        Give a short 3-4 line explanation of:
        - What it is
        - How to cure it
        - How to prevent it
        Keep it simple and practical for farmers.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=200,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print("Groq error:", e)
        return "No description available."



# -----------------------------
# Run Server
# -----------------------------
# if __name__ == "__main__":
#     app.run(debug=True, port=4000)

# for spaces
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
