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

@st.cache_data
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

    llm = use_huggingface_endpoint(model_name, 0.3)
    response = llm.bind(max_tokens=1024).invoke(prompt)

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


def detect_objects(image_path):
    """Detects objects in the provided image...."""
    image = Image.open(image_path).convert('RGB')

    encoded_image = encode_image(image_path)

    model_name = "facebook/detr-resnet-50"
    # Hugging Face API URL for DETR model
    client = InferenceClient(
        model_name,
        token=HUGGINGFACEHUB_API_TOKEN,
    )

    results = client.object_detection(image_path)
    draw = ImageDraw.Draw(image)
    detections = ""
    for result in results:
        box = result.box
        label = result.label
        score = result.score
        detections += f"[{int(box.xmin)}, {int(box.ymin)}, {int(box.xmax)}, {int(box.ymax)}]"
        detections += f"{label}"
        detections += f" {float(score)}\n"

        x0, y0, x1, y1 = box.xmin, box.ymin, box.xmax, box.ymax
        draw.rectangle([x0, y0, x1, y1], outline='red', width=5)
        draw.text((x0, y0), label, fill='white', font_size=24)

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == '__main__':
    set_tokens()
    detections = detect_objects(image_path)
    print(detections)