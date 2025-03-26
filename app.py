# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

# import os
# os.environ["KERAS_BACKEND"] = "jax"
import keras
import streamlit as st
from multiapp import MultiApp
import streamlit as st
from tensorflow import keras 
#from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.models import load_model, Sequential
# from tensorflow.python.keras.layers import LSTM, Dense

#@st.cache(allow_output_mutation=True, max_entries=100, ttl=10800,suppress_st_warning=True)

def foo():
    st.image ("ezgif.com-gif-maker.gif")
    st.write("Welcome to Medicinal Robot Web Application")
    st.write("It can distinguish the Pituitary tumor, Meningioma tumor, and Glioma tumor. And can also detect the presence or absence of brain tumors of other types through MRI images! !")
  


from PIL import Image, ImageOps
import numpy as np
#@st.cache(suppress_st_warning=True,allow_output_mutation=True, max_entries=100, ttl=10800)

def ImageClassificationModel(img,model):
    # Step 1
    model = keras.models.load_model('actual_my_model_MRI_Jupy_cancer_yes_no.h5')

    # Step 2
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Step 3
    image_array = np.asarray(image)
    
    # Step 4
    data[0] = image_array

    # Step 5
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability
#@st.cache(suppress_st_warning=True,allow_output_mutation=True, max_entries=100, ttl=10800)

def bar():
    #st.write(' It will predict the presence of Pituitary tumor, Meningioma tumor and Glioma tumors in the brain')
    uploaded_file = st.file_uploader("Please Choose MRI image of brain",type=["jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded brain MRI image', use_container_width=True)
        st.write("")
        st.write("Scanning........")
        label = ImageClassificationModel(image, 'actual_my_model_MRI_Jupy_cancer_yes_no.h5')
        if label == 0:
            st.write("Brain tumor is present")
        else:
            st.write("Brain tumor is not present")
#@st.cache(suppress_st_warning=True,allow_output_mutation=True, max_entries=100, ttl=10800)

def ImageClassificationModel1(img,model):
        model = keras.models.load_model('my_model_MRI_Jupy_types_of_cancer.h5')
        data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
        image = img
        #image sizing
        size = (150, 150)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        # Step 3
        image_array = np.asarray(image)
        # Normalize the image
        #Normalize_image = (image_array.astype(np.float32) / 127.0) - 1
        # Step 4
        data[0] = image_array
        # Step 5
        prediction = model.predict(data)
        return np.argmax(prediction) # return position of the highest probability

#@st.cache(suppress_st_warning=True,allow_output_mutation=True, max_entries=100, ttl=10800)
def bar1():
            # Dashboard Main Panel
            # Dashboard Main Panel
    col = st.columns((1.5, 4.5, 2), gap='medium')
    with col[1]:
            st.markdown('##### BRAIN TUMOR PREDICTOR')
            uploaded_file = st.file_uploader("Upload MRI image of Brain", type=["jpg","jpeg"])
      
            if uploaded_file is not None:
               image = Image.open(uploaded_file).convert('RGB')
               st.image(image, caption='Uploaded brain MRI image', use_container_width=True)
               st.write("")
               st.write("Result........")
               label = ImageClassificationModel1(image, 'my_model_MRI_Jupy_types_of_cancer.h5')
               print(label)
               if label == 0:
                   st.write("Either the Brain tumor is not present or the brain tumor of some other type is present.")
                   st.write("In order to verify, please click on the Brain Tumor Predictor(in general) web page!")
               elif label==1:
                   st.write("Pituitary tumor tumor is  present")
               elif label==2:
                   st.write("Meningioma tumor is  present")
               elif label==3:
                   st.write(" Glioma tumor  is  present")
    with col[2]:
               with st.expander('About', expanded=True):
                    st.write('''
                 - Devolper Desk: Brain Tumors are one of the deadliest diseases. It's important to diagnose it at a very early stage.Magnetic Resonance Imaging (MRI) is the most widely used method to identify brain tumors. With the help of Medicinal Robot, MRI images are analyzed to predict the presence or absence of Brain Tumor. This application has above 95% accuracy. However, before getting to any conclusion, please consult doctors first. You are using this web app at your own risk!
                 - :red[**Problem Statement**]: It can distinguish among Pituitary tumor, Meningioma tumor, and Glioma tumor.
                 - :red[**How to Use**]: Upload the MRI scan of brain in either JPG or JPEG format. 
                 ''')
          
#@st.cache(suppress_st_warning=True,allow_output_mutation=True, max_entries=100, ttl=10800)

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
app.add_app("Brain Tumor Predictor(in general)", bar)
app.run()
