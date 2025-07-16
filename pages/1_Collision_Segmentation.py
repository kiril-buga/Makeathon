import os
from io import BytesIO
from tempfile import NamedTemporaryFile, gettempdir
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw
import streamlit as st
from transformers import pipeline
import numpy as np
import random


def random_color():
    return tuple(np.random.randint(0, 255, size=3).tolist())

# Define a helper function to calculate the centroid of a mask
def calculate_centroid(mask):
    indices = np.argwhere(mask)
    y, x = indices.mean(axis=0).astype(int)
    return x, y

def segment_image(image_file):
    """Segments objects in the provided image, accepting either a file path or in-memory image."""
    image = Image.open(image_file).convert('RGB')

    # Use transformers pipeline for segmentation
    segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
    results = segmenter(image)

    # Create a copy to draw on
    segmented_image = image.copy()
    draw = ImageDraw.Draw(segmented_image, "RGBA")

    # To store segment colors
    segment_colors = {}

    for result in results:
        mask = result['mask']
        mask_np = np.array(mask)

        # Get a color for the segment
        segment_id = result.get('segment_id', random.randint(0, 1000)) # Use segment_id if available
        if segment_id not in segment_colors:
            # Generate a random color with some transparency
            segment_colors[segment_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150)

        color = segment_colors[segment_id]

        # Create a colored mask image
        colored_mask = Image.new("RGBA", mask.size, color)
        
        # Apply the mask
        segmented_image.paste(colored_mask, (0, 0), mask)

    return segmented_image


st.header("Please upload an image of an airport.")

file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if file:
    # display the image
    st.image(file, caption="Uploaded Image.", use_container_width=True)
    with st.spinner(text="Segmenting aircraft components..."):
        # segment the image
        segmented_image = segment_image(file)
        st.image(segmented_image, caption="Segmented Image.", use_container_width=True)

# if __name__ == "__main__":
#     set_tokens()
#     segment_image("../data/example_airport.jpg")

