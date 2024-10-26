from io import BytesIO
from tempfile import NamedTemporaryFile, gettempdir

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


def random_color():
    return tuple(np.random.randint(0, 255, size=3).tolist())

# Define a helper function to calculate the centroid of a mask
def calculate_centroid(mask):
    indices = np.argwhere(mask)
    y, x = indices.mean(axis=0).astype(int)
    return x, y

def segment_image(image_path: str):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Load the instance segmentation pipeline
    instance_segmentation = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
    # Perform instance segmentation on the image
    result = instance_segmentation(image)

    segmented_image = image.copy()
    draw = ImageDraw.Draw(segmented_image)
    # Loop through the results and draw each segment with a random color and label
    for segment in result:
        mask = np.array(segment['mask'])
        color = random_color()
        label = segment['label']

        # Overlay the mask with a random color
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    segmented_image.putpixel((x, y), color)

        # Calculate the centroid of the segment
        centroid_x, centroid_y = calculate_centroid(mask)

        # Draw the label at the centroid
        draw.text((centroid_x, centroid_y), label, fill=(255, 255, 255))

    blended_image = Image.blend(image.convert("RGBA"), segmented_image.convert("RGBA"), alpha=0.5)

    # Display the segmented image only
    # plt.figure(figsize=(40, 40))
    # plt.imshow(blended_image)
    # # plt.title('Segmented Image with Labels and Original Image Underneath')
    # plt.axis('off')  # Hide axes
    # plt.show()

    # Save the blended image to a BytesIO buffer
    buffer = BytesIO()
    blended_image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


st.header("Please upload an image of an airport.")

file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if file:
    image = Image.open(file).convert('RGB')
    st.image(file, caption="Uploaded Image.", use_column_width=True)
    with st.spinner(text="Segmenting aircraft components..."):
        # segment the image
        segmented_image = segment_image(file)

    st.image(segmented_image, caption="Segmented Image.", use_column_width=True)

# if __name__ == "__main__":
    # segment_image("../data/example_airport.jpg")

