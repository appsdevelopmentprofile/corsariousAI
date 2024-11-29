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
    image_np = np.array(image)
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(mask)
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        return None

def overlay_defect(background_image, synthetic_image, mask, alpha=0.2):  # Lower alpha for subtlety
    # Resize synthetic image
    synthetic_image = synthetic_image.resize(background_image.size, Image.Resampling.LANCZOS)

    # Convert images to numpy arrays
    background_np = np.array(background_image)
    synthetic_np = np.array(synthetic_image)

    # Reshape mask to match background channels
    mask_np = np.expand_dims(np.array(mask) / 255, axis=-1)
    mask_np = np.repeat(mask_np, 3, axis=-1)

    # Blend images
    blended_np = (background_np * (1 - mask_np) + synthetic_np * mask_np * alpha).astype(np.uint8)

    # Convert back to PIL image
    blended_image = Image.fromarray(blended_np)
    return blended_image

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
