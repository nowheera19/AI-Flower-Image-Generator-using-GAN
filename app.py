
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="BloomGen AI",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 BloomGen AI")
st.subheader("Generate your own AI flower images")
st.write(
    "Create new flower-like images using a Deep Convolutional "
    "Generative Adversarial Network (DCGAN)."
)

# Model path
MODEL_PATH = Path(__file__).parent / "flower_generator.keras"

@st.cache_resource
def load_generator():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "flower_generator.keras is missing. "
            "Place the trained model inside the project folder."
        )

    if MODEL_PATH.stat().st_size == 0:
        raise ValueError("The generator model file is empty.")

    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

# Load model
try:
    generator = load_generator()
    st.success("Flower generator loaded successfully! 🌷")
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# User controls
st.write("### Customize your generation")

number_of_images = st.slider(
    "Number of flowers to generate",
    min_value=1,
    max_value=16,
    value=4
)

if st.button("✨ Generate Flowers", type="primary"):
    with st.spinner("Creating your flowers..."):
        noise = tf.random.normal(
            shape=(number_of_images, 100)
        )

        generated_images = generator(
            noise, training=False
        ).numpy()

        st.write("### Your AI-generated flowers")

        columns = st.columns(4)

        for i, image_array in enumerate(generated_images):
            # Convert model output from [-1, 1] to [0, 255]
            image_array = (image_array + 1.0) / 2.0
            image_array = np.clip(
                image_array * 255, 0, 255
            ).astype(np.uint8)

            image = Image.fromarray(image_array)

            with columns[i % 4]:
                st.image(
                    image,
                    caption=f"Generated Flower {i + 1}",
                    use_container_width=True
                )

                # Download individual image
                from io import BytesIO

                buffer = BytesIO()
                image.save(buffer, format="PNG")

                st.download_button(
                    label="Download image",
                    data=buffer.getvalue(),
                    file_name=f"ai_flower_{i + 1}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )

st.divider()
st.caption(
    "Built with TensorFlow, Streamlit and DCGAN | "
    "Generated images are AI-created and may not look realistic."
)
