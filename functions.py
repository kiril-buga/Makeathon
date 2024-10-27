from io import BytesIO

import streamlit as st
import os
import base64
import requests


from PIL import Image, ImageDraw
from dotenv import load_dotenv
from huggingface_hub import InferenceApi, InferenceClient
from sympy.logic.algorithms.z3_wrapper import encoded_cnf_to_z3_solver
from transformers import pipeline
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool

image_path = 'data/example_aircraft.png'


def set_tokens():
    global HUGGINGFACEHUB_API_TOKEN
    global GROQ_API_KEY
    load_dotenv()
    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None:
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        print(f"Token is added")
    if os.getenv("GROQ_API_KEY") is not None:
        GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    if (os.getenv("AZURE_OPENAI_ENDPOINT") is not None
            or os.getenv("AZURE_OPENAI_API_KEY") is not None):
        AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
        AZUER_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    else:
        raise Exception("No API Token Provided!")

def use_huggingface_endpoint(_model, _temperature: 0.5, _max_new_tokens: int = 1024):
    callbacks = [StreamingStdOutCallbackHandler()]  # Callback for streaming output
    llm = HuggingFaceEndpoint(
        repo_id=_model,
        max_new_tokens=_max_new_tokens,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        task="text-generation",
        temperature=_temperature,
        repetition_penalty=1.03,
        callbacks=callbacks,
        do_sample=False,
        # stop_sequences=["<|eot_id|>"],
        streaming=True,
        timeout=1000,
    )
    return ChatHuggingFace(llm=llm,  model_id=_model, timeout=1000, streaming=True, verbose=True) # max_new_tokens=_max_new_tokens

def use_huggingface_object_detection(_model, _temperature: 0.5, _max_new_tokens: int = 1024):
    callbacks = [StreamingStdOutCallbackHandler()]  # Callback for streaming output
    llm = HuggingFaceEndpoint(
        repo_id=_model,
        max_new_tokens=_max_new_tokens,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        task="object-detection",
        temperature=_temperature,
        repetition_penalty=1.03,
        callbacks=callbacks,
        do_sample=False,
        # stop_sequences=["<|eot_id|>"],
        streaming=True,
        timeout=1000,
    )
    return llm

def get_image_caption(image_path):
    """Generates a short caption for the provided image...."""
    encoded_image = encode_image(image_path)

    #image = Image.open(image_path).convert('RGB')
    # model_name = "Salesforce/blip-image-captioning-large"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # image_to_text = pipeline("image-to-text", model=model_name) image_to_text(image)

    prompt = [
        AIMessage(content="You are a bot that is a technical expert at analyzing aircraft images."),
        HumanMessage(content=[
            {"type": "text", "text": "Provide the model of the aircraft. Then, provide a list of all aircraft elements you see in the image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{encoded_image}"
                },
            },
        ])
    ]

    llm = use_huggingface_endpoint(model_name, 0.2)
    response = llm.bind(max_tokens=1024, temperature=0.2).invoke(prompt)

    return response.content


# Function for image summaries
def summarize_image(encoded_image):
    """Summarizes the contents of the provided image."""
    model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = use_huggingface_endpoint(model, 0.5)

    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = llm.invoke(prompt)
    return response.content


def detect_objects(image_file):
    """Detects objects in the provided image, accepting either a file path or in-memory image."""

    # Handle in-memory file (BytesIO) or file path
    if isinstance(image_file, BytesIO):
        image = Image.open(image_file).convert('RGB')
        # Extract raw bytes for the client
        image_file.seek(0)  # Ensure we're at the start
        image_bytes = image_file.read()
    else:
        image = Image.open(image_file).convert('RGB')
        # Read file from path for raw bytes
        with open(image_file, "rb") as f:
            image_bytes = f.read()

    model_name = "facebook/detr-resnet-50"
    client = InferenceClient(
        model_name,
        token=HUGGINGFACEHUB_API_TOKEN,
    )

    # Send the raw bytes to object_detection
    results = client.object_detection(image=image_bytes)
    draw = ImageDraw.Draw(image)
    detections = ""

    for result in results:
        box = result.box
        label = result.label
        score = result.score
        detections += f"[{int(box.xmin)}, {int(box.ymin)}, {int(box.xmax)}, {int(box.ymax)}] {label} {float(score)}\n"

        x0, y0, x1, y1 = box.xmin, box.ymin, box.xmax, box.ymax
        draw.rectangle([x0, y0, x1, y1], outline='red', width=5)
        draw.text((x0, y0), label, fill='white')

    return image, detections

def classify(image, model, class_names):
    """Classifies the provided image...."""

    return 'dummy_class_name', 0.0

def query_image_huggingface(filename):
    import requests
    with open(filename, "rb") as f:
        image_data = f.read()
    response = requests.post(API_URL, headers=headers, files={"file": image_data})
    return response.json()


def encode_image(image_input):
    """Encodes an image to a base64 string."""
    # Check if the input is a file path or BytesIO object
    if isinstance(image_input, str):  # File path
        with open(image_input, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_input, BytesIO):  # In-memory file
        encoded_string = base64.b64encode(image_input.getvalue()).decode("utf-8")
    else:
        raise TypeError("Unsupported image input type")

    return encoded_string

if __name__ == '__main__':
    set_tokens()
    detections = detect_objects(image_path)
    print(detections)