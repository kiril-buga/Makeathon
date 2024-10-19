import streamlit as st

st.title('Hello World')

st.header("Please upload an image of an aircraft")

file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if file:
    # display the image
    st.image(file, caption="Uploaded Image.", use_column_width=True)

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