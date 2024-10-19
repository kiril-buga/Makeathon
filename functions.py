import os

from PIL import Image
from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace

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
    set_tokens()
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
        streaming=False,
        timeout=1000,
    )
    return ChatHuggingFace(llm=llm,  model_id=_model, timeout=1000, streaming=False, verbose=True) # max_new_tokens=_max_new_tokens

def get_image_caption(image_path):
    """Generates a short caption for the provided image...."""
    image = Image.open(image_path).convert('RGB')

    model_name = "Salesforce/blip-image-captioning-large"

    image_to_text = pipeline("image-to-text", model=model_name)

    return image_to_text(image)

def get_image_caption2(image_path):
    """Generates a short caption for the provided image...."""

    import base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    image = Image.open(image_path).convert('RGB')

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    prompt = [
        AIMessage(content="You are a bot that is a technical expert at analyzing aircraft images."),
        HumanMessage(content=[
            {"type": "text", "text": "Provide the model of the airplane. You need to mention all airplane parts you see in the image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{encoded_image}"
                },
            },
        ])
    ]

    llm = use_huggingface_endpoint(model_name, 0.5)
    response = llm.bind(max_tokens=1024).bind().invoke(prompt)

    return response.content

    image_to_text = pipeline("image-to-text", model=model_name)

    return image_to_text(image)


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
    pass

def query_image_huggingface(filename):
    import requests
    with open(filename, "rb") as f:
        image_data = f.read()
    response = requests.post(API_URL, headers=headers, files={"file": image_data})
    return response.json()

if __name__ == '__main__':
    caption = get_image_caption2(image_path)
    print(caption)