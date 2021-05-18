# -*- coding: utf-8 -*-


from tensorflow.keras.models import load_model
#model = load_model('https://github.com/priyankajagota/final_zoom/blob/main/my_model_MRI_Jupy.h5')

import streamlit as st
from multiapp import MultiApp
import streamlit as st
import keras



def foo():
    st.image ("ezgif.com-gif-maker.gif")
         
    st.write("Welcome to Medicinal Robot Web Application")
    st.write("It is used to predict the presence of Brain Tumor through MRI images !")


import keras
from PIL import Image, ImageOps
import numpy as np


def ImageClassificationModel(img,model):
    # Step 1
    model = keras.models.load_model('my_model_MRI_Jupy.h5')

    # Step 2
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Step 3
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Step 4
    data[0] = normalized_image_array

    # Step 5
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

def bar():
    uploaded_file = st.file_uploader("Please Choose MRI image of brain", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded brain MRI image', use_column_width=True)
        st.write("")
        st.write("Result........")
        label = ImageClassificationModel(image, 'my_model_MRI_Jupy.h5')
        if label == 1:
            st.write("Brain tumor is present")
        else:
            st.write("Brain tumor is not present")
def Author():
    st.image ("MEDIC CARE.png")
    st.write("Hello everyone !")
    st.write("I am extremely excited to share the Medicinal Robot web application with all of you. As we know, Brain Tumors are one of the deadliest diseases. It's important to diagnose it at a very early stage.Magnetic Resonance Imaging (MRI) is the most widely used method to identify brain tumors. With the help of Medicinal Robot, MRI images are analyzed to predict the presence or absence of Brain Tumor. This application has above 95% accuracy. However, before getting to any conclusion, please consult doctors first. You are using this web app at your own risk! ")
    st.write('Best Regards')
    st.write('Priyanka Jagota')
    
app = MultiApp()
app.add_app("Welcome Page", foo)
app.add_app("Developer's Desk", Author)
app.add_app("Brain Tumor Predictor", bar)
app.run()
