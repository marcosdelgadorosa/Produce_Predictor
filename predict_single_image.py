import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "fruit_freshness_multitask.keras"
IMG_SIZE = 128
FRUITS = ["apple", "banana", "carrot", "orange", "tomato"]

def load_and_preprocess_image(image_path):
    # Loads and preprocesses the image
    img = tf.io.read_file(image_path)

    # Can handle png and jpeg
    try:
        img = tf.image.decode_png(img, channels=3)
    except:
        img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)
    return img


def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    img = load_and_preprocess_image(image_path)

    preds = model.predict(img)

    fruit_logits = preds[0][0]
    fresh_logits = preds[1][0]

    fruit_idx = np.argmax(fruit_logits)
    fruit_name = FRUITS[fruit_idx]
    fruit_conf = float(np.max(fruit_logits))

    freshness = "fresh" if fresh_logits > 0.5 else "rotten"
    freshness_conf = float(fresh_logits if fresh_logits > 0.5 else 1 - fresh_logits)

    print("\n============= PREDICTION =============")
    print(f"Produce Type: {fruit_name} ({fruit_conf:.3f} confidence)")
    print(f"Freshness: {freshness} ({freshness_conf:.3f} confidence)")
    print("======================================\n")


if __name__ == "__main__":
    
    test_image = r"C:\Users\marco\Downloads\Produce and Freshness Predictor\Test Images\Tomato_Rotten.jpg" # CHANGE THIS to the image you want to test
    predict_image(test_image)
