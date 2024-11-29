import openai
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import streamlit as st

# Function to generate synthetic features using OpenAI's API
def generate_synthetic_features(prompt, api_key):
    # Set up the OpenAI API key
    openai.api_key = api_key
    
    # Use DALL-E to generate the synthetic image
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    # Extract the image URL
    image_url = response['data'][0]['url']
    
    # Download the generated image
    response = requests.get(image_url)
    synthetic_image = Image.open(BytesIO(response.content))
    
    return synthetic_image

# Function to blend the uploaded image and synthetic features
def blend_images(background_image, synthetic_image, alpha=0.5):
    # Resize synthetic image to match background size
    synthetic_image = synthetic_image.resize(background_image.size, Image.ANTIALIAS)
    
    # Blend the images together
    blended_image = Image.blend(background_image, synthetic_image, alpha)
    
    return blended_image

# Streamlit UI
with st.sidebar:
    openai_api_key = st.text_input("Insert your API key:")
    st.markdown("-------")

# Title
st.title("Synthetic Image Generation with Background Preservation")

# File uploader for background image
uploaded_file = st.file_uploader("Upload a background image...", type=["jpg", "png", "jpeg"])

# Text input for synthetic features prompt
prompt = st.text_input("Describe the new features for the synthetic image:", value="Heavy rust, corrosion, weathered, damaged, old, abandoned")

# Display the uploaded image
if uploaded_file is not None:
    # Open the image using PIL
    background_image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(background_image, caption="Uploaded Background Image", use_column_width=True)

# Generate button
if st.button("Generate Synthetic Image"):
    if openai_api_key and uploaded_file:
        # Generate synthetic features
        st.write("Generating synthetic features...")
        synthetic_features = generate_synthetic_features(prompt, openai_api_key)
        
        # Blend synthetic features into the uploaded background
        st.write("Blending synthetic features with the background image...")
        blended_image = blend_images(background_image, synthetic_features, alpha=0.5)
        
        # Display the final blended image
        st.image(blended_image, caption="Generated Synthetic Image", use_column_width=True)
        
        # Option to download the blended image
        buffered = BytesIO()
        blended_image.save(buffered, format="PNG")
        st.download_button(
            label="Download Synthetic Image",
            data=buffered.getvalue(),
            file_name="synthetic_image.png",
            mime="image/png"
        )
    else:
        st.error("Please upload a background image and enter your OpenAI API key.")
