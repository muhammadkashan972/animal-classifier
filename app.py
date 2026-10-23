import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.title("🐾 AI Animal Detector Web App")
st.write("Kashan ka banaya hua Animal Classification Model!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('animal_classifier_model.h5')

try:
    model = load_model()
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")

class_names = ['d', 'f', 'l', 's', 'z']
animal_full_names = {
    'd': 'Deer (Hiran)',
    'f': 'Fox (Lomri)',
    'l': 'Lion (Sher)',
    's': 'Stag (Barasingha)',
    'z': 'Zebra'
}

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict Animal'):
        with st.spinner('Analyzing image...'):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            
            predicted_code = class_names[np.argmax(predictions)]
            confidence = 100 * np.max(score)
            
            st.success(f"### Animal Name: {animal_full_names[predicted_code]}")
            st.info(f"### Confidence Score: {confidence:.2f}%")
