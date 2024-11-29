import openai
import os
from PIL import Image, ImageOps
import requests
from io import BytesIO
import streamlit as st
import numpy as np
import cv2  # For mask operations


# Function to generate synthetic image with defect using OpenAI API
def generate_synthetic_image(prompt, api_key):
    openai.api_key = api_key
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    return image_url


# Function to download and process synthetic image
def download_image(image_url):
    response = requests.get(image_url)
    synthetic_image = Image.open(BytesIO(response.content))
    return synthetic_image


# Function to segment the asset (equipment) using a simple segmentation technique (mockup)
def segment_asset(image):
    # Convert the image to grayscale and apply thresholding (mockup example)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold
    
    # Convert mask to PIL image
    mask_pil = Image.fromarray(mask)
    return mask_pil


# Function to overlay the synthetic defect on the segmented asset
def overlay_defect(background_image, synthetic_image, mask, alpha=0.7):
    # Resize synthetic image to match background size
    synthetic_image = synthetic_image.resize(background_image.size, Image.Resampling.LANCZOS)
    mask = mask.resize(background_image.size, Image.Resampling.LANCZOS)
    
    # Convert images to numpy arrays
    background_np = np.array(background_image)
    synthetic_np = np.array(synthetic_image)
    mask_np = np.array(mask) / 255  # Normalize mask to [0, 1]
    
    # Blend images only in masked regions
    blended_np = (background_np * (1 - mask_np) + synthetic_np * mask_np * alpha).astype(np.uint8)
    
    # Convert back to PIL image
    blended_image = Image.fromarray(blended_np)
    return blended_image


# Streamlit UI
st.title("Synthetic Defect Generation for Assets")

with st.sidebar:
    openai_api_key = st.text_input("Insert your OpenAI API key:", type="password")
    st.markdown("-------")

# Upload original image
uploaded_file = st.file_uploader("Upload an image of the asset (equipment):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    background_image = Image.open(uploaded_file).convert("RGB")
    st.image(background_image, caption="Uploaded Image", use_column_width=True)

    # Text input for prompt
    prompt = st.text_input("Describe the defect (e.g., 'heavy rust on the pipeline'):")

    # Generate synthetic defect image
    if st.button("Generate Synthetic Defect"):
        if openai_api_key and prompt:
            # Generate synthetic defect image
            image_url = generate_synthetic_image(prompt, openai_api_key)
            synthetic_image = download_image(image_url)
            
            # Segment the asset
            mask = segment_asset(background_image)

            # Overlay defect
            result_image = overlay_defect(background_image, synthetic_image, mask)

            # Display result
            st.image(result_image, caption="Synthetic Image with Defect", use_column_width=True)
        else:
            st.error("Please provide a valid API key and prompt.")
