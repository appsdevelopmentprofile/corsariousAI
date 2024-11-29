from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import streamlit as st

# Function to generate a synthetic image placeholder based on a prompt (replace with actual implementation)
def generate_synthetic_image(prompt):
    try:
        # Placeholder: Replace this with your image generation implementation
        st.warning("Image generation functionality is not implemented. Returning a placeholder image.")
        placeholder_image = Image.new("RGB", (1024, 1024), color=(255, 0, 0))  # Red placeholder image
        return placeholder_image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Function to segment the asset (equipment) using thresholding
def segment_asset(image):
    # Convert the image to grayscale
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to create a binary mask
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Convert the mask to a PIL image
    mask_pil = Image.fromarray(mask)
    return mask_pil

# Function to overlay the defect on the segmented asset
def overlay_defect(background_image, synthetic_image, mask, alpha=0.7):
    # Resize synthetic image and mask to match the background size
    synthetic_image = synthetic_image.resize(background_image.size, Image.Resampling.LANCZOS)
    mask = mask.resize(background_image.size, Image.Resampling.LANCZOS)
    
    # Convert images to numpy arrays
    background_np = np.array(background_image)
    synthetic_np = np.array(synthetic_image)
    mask_np = np.array(mask) / 255  # Normalize mask to [0, 1]
    
    # Blend images only in masked regions
    blended_np = (background_np * (1 - mask_np) + synthetic_np * mask_np * alpha).astype(np.uint8)
    
    # Convert back to a PIL image
    blended_image = Image.fromarray(blended_np)
    return blended_image

# Streamlit UI
def main():
    st.title("Synthetic Defect Generation for Assets")

    # Upload original image
    uploaded_file = st.file_uploader("Upload an image of the asset (equipment):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        background_image = Image.open(uploaded_file).convert("RGB")
        st.image(background_image, caption="Uploaded Image", use_column_width=True)

        # Text input for prompt
        prompt = st.text_input("Describe the defect (e.g., 'heavy rust on the pipeline'):")

        # Generate and overlay synthetic defect if prompt is provided
        if st.button("Generate Synthetic Defect"):
            if prompt:
                # Generate synthetic image
                synthetic_image = generate_synthetic_image(prompt)
                if synthetic_image:
                    # Segment the asset
                    mask = segment_asset(background_image)
                    
                    # Overlay defect
                    result_image = overlay_defect(background_image, synthetic_image, mask)
                    
                    # Display result
                    st.image(result_image, caption="Synthetic Image with Defect", use_column_width=True)
            else:
                st.error("Please provide a valid prompt.")

if __name__ == "__main__":
    main()
