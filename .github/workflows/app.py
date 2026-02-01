import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# =================================================
# Page Config
# =================================================
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# =================================================
# Title
# =================================================
st.title("✍️ Handwritten Digit Recognition")
st.caption("Draw a digit (0–9) and let the neural network predict it")

# =================================================
# Load & Train Model (Cached)
# =================================================
@st.cache_resource
def load_model():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)

    y_train = keras.utils.to_categorical(y_train, 10)

    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=8,
        batch_size=128,
        verbose=0
    )

    return model

model = load_model()

# =================================================
# Canvas Input
# =================================================
st.subheader("Draw Here")

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# =================================================
# Prediction Logic
# =================================================
if st.button("Predict Digit 🚀"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]   # grayscale
        img = Image.fromarray(img).resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 784)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"🧠 Predicted Digit: **{digit}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    else:
        st.warning("Please draw a digit first.")

# =================================================
# Footer
# =================================================
st.caption("Built with TensorFlow & Streamlit")
