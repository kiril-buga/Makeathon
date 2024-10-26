import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import requests

image_path = "../data/example_aircraft.png"
image = Image.open(image_path)

# Install the required package

instance_segmentation = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

# Perform instance segmentation on the image
result = instance_segmentation(image)

# Create a new image to store the masks
segmented_image = Image.new('RGB', image.size, (0, 128, 0))
draw = ImageDraw.Draw(segmented_image)


# Define a function to generate random colors
def random_color():
    return tuple(np.random.randint(0, 255, size=3).tolist())


# Define a helper function to calculate the centroid of a mask
def calculate_centroid(mask):
    indices = np.argwhere(mask)
    y, x = indices.mean(axis=0).astype(int)
    return x, y


font = ImageFont.load_default()

# Loop through the results and draw each segment with a random color and label
for segment in result:
    if segment["label"] == "airplane":
        mask = np.array(segment['mask'])
        label = segment['label']

        # Overlay the mask with a random color
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    segmented_image.putpixel((x, y), image.getpixel((x, y)))

        # Calculate the centroid of the segment
        centroid_x, centroid_y = calculate_centroid(mask)

        # Draw the label at the centroid
        draw.text((centroid_x, centroid_y), label, fill=(255, 255, 255))

segmented_image.save("../segmented_airplane/segmented_airplane.png")
