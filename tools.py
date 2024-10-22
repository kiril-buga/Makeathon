from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from langchain_core.tools import tool

import functions

class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        return functions.get_image_caption(img_path)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        return functions.detect_objects(img_path)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")