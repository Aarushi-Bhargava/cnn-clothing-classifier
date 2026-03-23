# type: ignore

#streamlit

import streamlit as st
import keras
from PIL import Image #to process image formats
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__)) #gets path to main.py file instead of hardcoding it
model_path = f"{working_dir}/trained_model/trained_fashion_mnist_model.h5" #concatenating

#load the pre-trained model
model = keras.models.load_model(model_path)

#define class labels for fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L') #converting to greyscale, L = luminiscence
    img_array = np.array(img) /225.0
    img_array = img_array.reshape((1, 28, 28, 1)) #getting prediction for 1 image, not 10 images at once
    return img_array

#streamlit app
st.title('Classify your Closet - CNN')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            #preprocessing the uploaded image
            img_array = preprocess_image(uploaded_image)

            #making a predicting using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')