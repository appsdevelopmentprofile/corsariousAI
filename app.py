import openai
import os
from PIL import Image
import requests
from io import BytesIO
import streamlit as st

# Function to generate the synthetic image using OpenAI API
def generate_image(prompt, api_key):
    # Set up the OpenAI API key
    openai.api_key = api_key
    
    # Use the DALL-E model to generate an image based on the prompt
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    # Extract the image URL from the response
    image_url = response['data'][0]['url']
    
    return image_url

# Function to download the image
def download_image(image_url, save_path):
    # Download the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Save the image
    image.save(save_path)
    return save_path

# Streamlit UI
with st.sidebar:
    openai_api_key = st.text_input("Insert your API key:")
    st.markdown("-------")

# Title
st.title("Synthetic Image Generation with Structural Consistency")

# Dropdown menus for asset and defect selection
asset = st.selectbox("Select an asset", ["Pipeline"])
defect = st.selectbox("Select a defect", ["Rust"])

# Text input for prompt
user_features = st.text_input("Describe the new features you want to add", value="Heavy rust, corrosion, weathered, damaged, old, abandoned")

# Placeholder for image upload and display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        # Display the image in Streamlit
        st.image(image, caption="Uploaded Image", use_column_width=False)

# Generate button and synthetic image logic
if st.button("Generate Synthetic Image"):
    if openai_api_key:
        if uploaded_file is not None:
            # Generate a prompt that includes both the structure and the new features
            image_description = f"An image of a {asset} with the structure similar to the uploaded image, but showing {defect}. {user_features}."
            
            # Generate the synthetic image
            image_url = generate_image(image_description, openai_api_key)
            
            with col2:
                st.subheader("Synthetic Image")
                # Display the synthetic image in Streamlit
                st.image(image_url, caption="Synthetic Image", use_column_width=True)
        else:
            st.warning("Please upload an image to use as a reference.")
    else:
        st.warning("Please enter your OpenAI API key.")
