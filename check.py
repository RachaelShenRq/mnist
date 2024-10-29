import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_model(model_path="./model.h5"):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_np = np.array(img).astype("float32") / 255  # Normalize
    img_np = np.expand_dims(img_np, axis=(0, -1))  # Add batch and channel dimensions
    return img_np

def predict(model, img_path):
    img_tensor = preprocess_image(img_path)
    prediction = model.predict(img_tensor)
    predicted_class = np.argmax(prediction)
    print(f"Predicted class for {img_path}: {predicted_class}")

"""if __name__ == "__main__":
    model = load_model()
    for i in range(10):
        predict(model, f"./data/num{i}.png")

check(".data/num0.png")"""

if __name__ == "__main__":
    model = load_model()
    for i in range(10):
        predict(model, f"./data/num{i}.png")

    # Correct the last line by calling 'predict' instead of 'check'
    # predict(model, "./data/num0.png")
