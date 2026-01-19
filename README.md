# Makeathon - Let's Fly ✈️

An AI-powered aircraft inspection solution developed for **Swiss Airlines' Challenge 3: Drone Aircraft Inspection**.  This prototype integrates a multimodal agentic chatbot with computer vision capabilities to revolutionize aircraft maintenance and inspection workflows.

## 🚀 Prototype Demo
https://github.com/user-attachments/assets/65b2fa79-2c2a-42f0-bef2-5b673efb15dd

The prototype is deployed at **[letsfly.streamlit.app](https://letsfly.streamlit.app/)**

## 🎯 Project Overview

Let's Fly addresses the challenge of automated drone-based aircraft inspection through an intelligent agent chatbot system. The application combines state-of-the-art computer vision models with conversational AI to:

- **Aircraft Component Detection**: Automatically identify and locate aircraft parts (engines, wings, tails) in images captured by drones or smartphones
- **Image Segmentation**: Separate and identify different elements in images including backgrounds, objects, and people
- **Aircraft Classification**: Identify aircraft types, models, and specific components with high accuracy
- **Interactive Q&A**:  Ask natural language questions about uploaded aircraft images and receive detailed analysis
- **Object Extraction**: Extract specific objects from images while filtering out backgrounds and people

**Live Demo Highlights:**
- Tested in real-time during the Makeathon event
- Validated with actual airplane images and physical mock-ups
- Demonstrated successful component detection on engines, wings, and tails

## 🛠️ Technology Stack

- **Frontend**:  Streamlit
- **AI/ML Models**: 
  - Meta Llama 3.2 11B Vision Instruct (image captioning & conversational AI)
  - Custom-trained models for airplane component detection
  - HuggingFace Transformers (object detection, image segmentation)
- **Frameworks**:  
  - LangChain (agent orchestration & memory)
  - LangGraph (workflow management)
  - PyTorch (model training)
- **APIs**: HuggingFace Inference API

## 📁 Project Structure

```
Makeathon/
├── Hello.py                           # Main Streamlit app entry point
├── functions.py                       # Core utility functions for AI models
├── tools.py                          # LangChain tools for image analysis
├── pages/
│   ├── 1_Collision_Segmentation.py   # Image segmentation page
│   └── 2_Aircraft_Classification. py  # Aircraft type classification page
├── notebooks/
│   ├── image_segmentation.ipynb      # Image segmentation experiments
│   ├── object_detection.ipynb        # Object detection prototyping
│   ├── train_model.ipynb             # Model training notebook
│   ├── LLM_prototyping.ipynb         # LLM experimentation
│   └── langgraph.ipynb               # LangGraph workflow development
├── data/                             # Sample aircraft images
└── vectordb/                         # Vector database storage
```

## 🔧 Key Features

### 1. **Multimodal Agentic Chatbot**
- Powered by Meta Llama 3.2 11B Vision model
- Conversational interface for intuitive aircraft inspection
- Context-aware responses with memory of previous interactions
- Real-time analysis of smartphone or drone-captured images

### 2. **Advanced Image Segmentation**
Intelligent segmentation to identify and separate:
- **Backgrounds**:  Automatic identification and removal of irrelevant scenery
- **Objects**: Precise detection and isolation of aircraft components (engines, wings, tails)
- **People**: Filtering out personnel from inspection images
- Clean extraction of relevant aircraft parts for focused analysis

### 3. **Component Detection & Classification**
Custom-trained AI model to:
- Detect airplane engines with bounding boxes
- Identify wing structures and configurations
- Recognize tail assemblies and stabilizers
- Classify aircraft types and models with confidence scores

### 4. **Object Extraction**
- Extract specific components from complex images
- Isolate individual parts for detailed analysis
- Support for images from smartphones, drones, or fixed cameras

## 💡 Use Cases for Swiss Airlines

- **Drone-Based Inspections**:  Automated analysis of drone-captured aircraft images
- **Pre-Flight Checks**: Quick visual inspection of critical components
- **Component Documentation**: Identify and catalog aircraft parts from inspection images
- **Training & Documentation**:  Educational tool for maintenance personnel
- **Quality Control**: Automated inspection workflows in hangars and maintenance facilities
- **Inspection Reports**: Generate detailed visual reports with component identification

## 📊 Model Performance

- **Custom Training**: Models trained specifically on airplane components (engines, wings, tails)
- **Real-Time Processing**: Optimized for quick analysis during live inspections
- **High Accuracy**: Validated during live demo with physical mock-ups and real aircraft images
- **Extensible**:  Notebooks included for fine-tuning on airline-specific aircraft fleets

## 🏆 Makeathon Achievement

**Swiss Airlines Challenge 3: Drone Aircraft Inspection**

This prototype was developed and deployed during the Makeathon hackathon event, where it was successfully demonstrated to a live audience. The system processed real-time smartphone images of airplane mock-ups, showcasing its practical viability for production deployment.

## 👥 Team

Developed during the Makeathon hackathon event for Swiss Airlines.

## 🙏 Acknowledgments

- **Swiss Airlines** for presenting the drone inspection challenge
- **HuggingFace** for providing powerful AI models and infrastructure
- **Meta** for the Llama 3.2 Vision model
- **Streamlit** for the rapid prototyping framework
- **Makeathon organizers** for hosting the event

---

**Note**:  This is a prototype developed for demonstration purposes during a hackathon. For production deployment in safety-critical aviation applications, additional validation, testing, and certification would be required per aviation industry standards.