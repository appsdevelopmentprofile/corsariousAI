from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import openai
import requests

# Function to generate a synthetic image based on a prompt
def generate_synthetic_image(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        synthetic_image = Image.open(BytesIO(response.content)).convert("RGB")
        return synthetic_image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Function to segment the asset (equipment) using thresholding
def segment_asset(image):
    image_np = np.array(image)
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(mask)
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        return None

# Function to overlay the defect on the segmented asset
def overlay_defect(background_image, synthetic_image, mask, alpha=0.1):
    # Resize synthetic image
    synthetic_image = synthetic_image.resize(background_image.size, Image.Resampling.LANCZOS)

    # Convert images to numpy arrays
    background_np = np.array(background_image)
    synthetic_np = np.array(synthetic_image)

    # Reshape mask (optional)
    mask_np = np.expand_dims(np.array(mask) / 255, axis=-1)  # Normalize and add channel
    mask_np = np.repeat(mask_np, 3, axis=-1)  # Repeat for 3 channels

    # Blend images
    blended_np = (background_np * (1 - mask_np) + synthetic_np * mask_np * alpha).astype(np.uint8)

    # Convert back to PIL image
    blended_image = Image.fromarray(blended_np)
    return blended_image

# Streamlit UI
def main():
    st.title("Synthetic Defect Generation for Assets")

    # Sidebar for API key input
    with st.sidebar:
        openai_api_key = st.text_input("Insert your OpenAI API key:", type="password")
        st.markdown("-------")

    # Upload original image
    uploaded_file = st.file_uploader("Upload an image of the asset (equipment):",
