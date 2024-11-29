from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import openai
import requests

def generate_synthetic_image(prompt, api_key):
    openai.api_key = api_key
    try:
        st.info("Generating synthetic image...")
        response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        synthetic_image = Image.open(BytesIO(response.content)).convert("RGB")
        st.success("Synthetic image generated successfully!")
        return synthetic_image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def segment_asset(image):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours of the pipeline
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the pipeline area
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)

    return Image.fromarray(mask)

def overlay_defect(background_image, synthetic_image, mask, alpha=0.2):
    # Resize synthetic image
    synthetic_image = synthetic_image.resize(background_image.size, Image.Resampling.LANCZOS)

    # Convert images to numpy arrays
    background_np = np.array(background_image)
    synthetic_np = np.array(synthetic_image)
    mask_np = np.array(mask) / 255

    # Blend images, focusing on the masked area
    blended_np = background_np.copy()
    blended_np[mask_np == 1] = (background_np * (1 - alpha) + synthetic_np * alpha)[mask_np == 1]

    return Image.fromarray(blended_np)

def main():
    st.title("Synthetic Defect Generation for Assets")

    with st.sidebar:
        openai_api_key = st.text_input("Insert your OpenAI API key:", type="password")
        st.markdown("-------")

    uploaded_file = st.file_uploader("Upload an image of the asset (equipment):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        background_image = Image.open(uploaded_file).convert("RGB")
        st.image(background_image, caption="Uploaded Image", use_column_width=True)

        prompt = st.text_input("Describe the defect (e.g., 'heavy rust on the pipeline'):")

        if st.button("Generate Synthetic Defect"):
            if openai_api_key and prompt:
                synthetic_image = generate_synthetic_image(prompt, openai_api_key)
                if synthetic_image:
                    mask = segment_asset(background_image)
                    if mask:
                        result_image = overlay_defect(background_image, synthetic_image, mask)
                        st.image(result_image, caption="Synthetic Image with Defect", use_column_width=True)
            else:
                st.error("Please provide a valid API key and prompt.")

if __name__ == "__main__":
    main()
