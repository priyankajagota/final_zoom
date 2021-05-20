# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


from tensorflow.keras.models import load_model
#model = load_model('my_model_MRI_Jupy.h5')

import streamlit as st
from multiapp import MultiApp
import streamlit as st
import keras
import streamlit_theme as stt
# Primary accent for interactive elements
st.markdown(
    """
    <style>
    .reportview-container {
        background: '#1f1f1f'
    }
   .sidebar .sidebar-content {
        background: '#d1d1d1'
    }
    </style>
    """,
    unsafe_allow_html=True
)

#backgroundColor = '#273346'
# Background color for sidebar and most interactive widgets
#secondaryBackgroundColor = '#B9F1C0'
# Color used for almost all text
#textColor = '#FFFFFF'
# Font family for all text in the app, except code blocks
# Accepted values (serif | sans serif | monospace) 
# Default: "sans serif"
#font = "sans serif"



def foo():
    st.image ("ezgif.com-gif-maker.gif")
         
    st.write("Welcome to Medicinal Robot Web Application")
    st.write("It can distinguish the Pituitary tumor, Meningioma tumor, and Glioma tumor. And can also detect the presence or absence of brain tumors of other types through MRI images! !")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: '#1f1f1f'
    }
   .sidebar .sidebar-content {
        background: '#d1d1d1'
    }
    </style>
    """,
    unsafe_allow_html=True
    )
  


import keras
from PIL import Image, ImageOps
import numpy as np


def ImageClassificationModel(img,model):
    # Step 1
    model = keras.models.load_model('actual_my_model_MRI_Jupy_cancer_yes_no.h5')

    # Step 2
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
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
    uploaded_file = st.file_uploader("Please Choose MRI image of brain",type=["jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded brain MRI image', use_column_width=True)
        st.write("")
        st.write("Scanning........")
        label = ImageClassificationModel(image, 'actual_my_model_MRI_Jupy_cancer_yes_no.h5')
        if label == 0:
            st.write("Brain tumor is present")
        else:
            st.write("Brain tumor is not present")
def ImageClassificationModel1(img,model):
        model = keras.models.load_model('my_model_MRI_Jupy_types_of_cancer.h5')
        data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
        image = img
        #image sizing
        size = (150, 150)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        # Step 3
        image_array = np.asarray(image)
        # Normalize the image
        #Normalize_image = (image_array.astype(np.float32) / 127.0) - 1
        # Step 4
        data[0] = image_array
        # Step 5
        prediction = model.predict(data)
        return np.argmax(prediction) # return position of the highest probability
def bar1():
         uploaded_file = st.file_uploader("Please Choose MRI image of brain", type=["jpg","jpeg"])
         if uploaded_file is not None:
             image = Image.open(uploaded_file)
             st.image(image, caption='Uploaded brain MRI image', use_column_width=True)
             st.write("")
             st.write("Result........")
             label = ImageClassificationModel1(image, 'my_model_MRI_Jupy_types_of_cancer.h5')
             print(label)
             #st.write(label)
             if label == 0:
                 st.write("Either the Brain tumor is not present or the brain tumor of some other type is present.")
                 st.write("In order to verify, please click on the Brain Tumor Predictor(in general) web page!")
             elif label==1:
                      st.write("Pituitary tumor tumor is  present")
             elif label==2:
                    st.write("Meningioma tumor is  present")
             elif label==3:
                   st.write(" Glioma tumor  is  present")
def Author():
    st.image ("MEDIC CARE.png")
    st.write("Hello everyone !")
    st.write("I am extremely excited to share the Medicinal Robot web application with all of you. As we know, Brain Tumors are one of the deadliest diseases. It's important to diagnose it at a very early stage.Magnetic Resonance Imaging (MRI) is the most widely used method to identify brain tumors. With the help of Medicinal Robot, MRI images are analyzed to predict the presence or absence of Brain Tumor. This application has above 95% accuracy. However, before getting to any conclusion, please consult doctors first. You are using this web app at your own risk! ")
    st.write('Best Regards')
    st.write('Priyanka Jagota')
    st.write('       ')
    st.write('Email Address : medicinalrobot@gmail.com')
    
    
app = MultiApp()
app.add_app("Welcome Page", foo)
app.add_app("Developer's Desk", Author)
app.add_app("Brain Tumor Type Predictor", bar1)
app.add_app("Brain Tumor Predictor", bar)
app.run()
