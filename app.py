import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Modeli yükleyin
model = tf.keras.models.load_model('best.keras')

# Sınıf isimlerinizi içeren liste
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweet potato', 'tomato', 'turnip', 'watermelon'
]

def prepare_image(image):
    img = image.resize((150, 150))  # Modelin beklediği boyut
    img = img.convert('RGB')  # Renkli formatta olmalı
    img_array = np.array(img) / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekleyin
    return img_array

def predict_image(image):
    img_array = prepare_image(image)
    predictions = model.predict(img_array)
    return predictions

def get_prediction_result(predictions):
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    return predicted_label, confidence

# Streamlit uygulaması
st.title('Meyve ve Sebze Sınıflandırıcı')

uploaded_file = st.file_uploader("Bir meyve veya sebze resmi yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Resim.', use_column_width=True)
    st.write("Tahmin ediliyor...")
    
    predictions = predict_image(image)
    predicted_label, confidence = get_prediction_result(predictions)
    
    st.write(f"Tahmin: {predicted_label}")
    st.write(f"Olasılık: %{confidence:.2f}")
