from tempfile import NamedTemporaryFile, gettempdir

import streamlit as st
from PIL import Image
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import functions
from tools import ImageCaptionTool, ObjectDetectionTool


##############################
### initialize agent #########
##############################
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = functions.use_huggingface_endpoint(model, 0.0)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stoppy_method='generate'
)

st.title('Swiss Airlines Chatbot')

st.header("Please upload an image of an aircraft")

file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if file:
    # display the image
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # classify the image
   # class_name, conf_score = functions.classify(image, model, class_names)

    # write classification results
    # st.write("## {}".format(class_name))
    # st.write("Confidence score: {:.2f}".format(conf_score))



    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(dir=gettempdir(), delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name

        caption = functions.get_image_caption(image_path)
        st.write(caption)

        # text input
        user_question = st.text_input("Ask a question about the aircraft in the image: ")

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run(f'{user_question}, this is the image path: {image_path}')
                st.write(response)