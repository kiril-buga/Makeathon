import streamlit as st
#from keras.models import load_model
from PIL import Image

from functions import *

st.title('Hello World')

st.header("Please upload an image of an aircraft")

file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if file:
    # display the image
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # classify the image
    class_name, conf_score = classify(image, model, class_names)

    # write classification results
    st.write("## {}".format(class_name))
    st.write("Confidence score: {:.2f}".format(conf_score))

    # text input
    user_question = st.text_input("Ask a question about the aircraft in the image:")

    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                st.write(response)