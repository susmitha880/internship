import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("model/digit_model.h5", compile=False)

st.title("✍️ Handwritten Digit Recognition")

# ================= PREPROCESS FUNCTION =================
def preprocess_image(img):
    # Resize to bigger size for better processing
    img = cv2.resize(img, (100, 100))

    # Convert to binary (black & white)
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    # Find bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size > 0:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Resize digit to 20x20
    img = cv2.resize(img, (20, 20))

    # Create 28x28 black image
    final_img = np.zeros((28, 28))

    # Place digit in center
    final_img[4:24, 4:24] = img

    # Normalize
    final_img = final_img / 255.0

    return final_img.reshape(1, 28, 28, 1)

# Sidebar
option = st.sidebar.selectbox("Choose Input Method", ["Draw Digit", "Upload Image"])

# ================= DRAW DIGIT =================
if option == "Draw Digit":
    st.subheader("Draw a digit below 👇")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=8,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0]
        img = img.astype('uint8')

        img = preprocess_image(img)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.write(f"### 🧠 Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")

        st.image(img.reshape(28, 28), caption="Processed Image")

# ================= UPLOAD IMAGE =================
elif option == "Upload Image":
    st.subheader("Upload an image of a digit")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=200)

        img = np.array(image)

        # Invert colors (important for MNIST)
        img = cv2.bitwise_not(img)

        img = preprocess_image(img)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.write(f"### 🧠 Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")

        st.image(img.reshape(28, 28), caption="Processed Image")