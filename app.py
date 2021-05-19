# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:26:45 2021

@author: priya
"""

from tensorflow.keras.models import load_model

import streamlit as st
from multiapp import MultiApp
import streamlit as st
import keras

def foo():
    st.image ("ezgif.com-gif-maker.gif")
         
    st.write("Welcome to Medic Care Web Application")
    st.write("It is used to predict the presence of Brain Tumor through MRI images !")
#model1 = keras.models.load_model('my_model_MRI_Jupy_types_of_cancer.h5')
#img = scipy.ndimage.imread('suza.jpg', mode='RGB')


import keras
from PIL import Image, ImageOps
import numpy as np
#model1=keras.models.save_model('my_model_MRI_jupy.h5','C:\\Users\priya\Documents\Final_Capstone_Files_Updated\archive')



def ImageClassificationModel(img,model):
    model = keras.models.load_model('my_model_MRI_Jupy_types_of_cancer.h5')
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Step 3
    image_array = np.asarray(image)
    # Normalize the image
    Normalize_image = (image_array.astype(np.float32) / 127.0) - 1

    # Step 4
    data[0] = image_array
    # Step 5
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

def bar():
    uploaded_file = st.file_uploader("Please Choose MRI image of brain", type=["jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded brain MRI image', use_column_width=True)
        st.write("")
        st.write("Result........")
        label = ImageClassificationModel(image, 'my_model_MRI_Jupy_types_of_cancer.h5')
        print(label)
        #.write(label)
        if label == 0:
            st.write("Brain tumor is not present")
        elif label==1:
            st.write("Pituitary tumor tumor is  present")
        elif label==2:
             st.write("Meningioma tumor is  present")
        elif label==3:
            st.write(" Glioma tumor  is  present")
            
def Author():
    st.image ("MEDIC CARE (1).png")
    st.write("Hello everyone !")
    st.write("I am extremely excited to share the Medic Care web application with all of you. As we know, Brain Tumors are one of the deadliest diseases. It's important to diagnose it at a very early stage.Magnetic Resonance Imaging (MRI) is the most widely used method to identify brain tumors. With the help of medic care, MRI images are analyzed to predict the presence or absence of Brain Tumor. This application has above 95% accuracy. However, before getting to any conclusion, please consult doctors first. You are using this web app at your own risk! ")
    st.write('Best Regards')
    st.write('Priyanka Jagota')
    
app = MultiApp()
app.add_app("Welcome Page", foo)
app.add_app("Developer's Desk", Author)
app.add_app("Brain Tumor Predictor", bar)
app.run()

