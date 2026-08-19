import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo entrenado
@st.cache_resource
def get_model():
    model = load_model('modelo_cifar10.h5')
    return model

model = get_model()

class_names = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Interfaz de la App
st.title("🧠 Examen - Computación en la Nube")
st.subheader("Clasificación de Imágenes con CNN y CIFAR-10")

# REQUISITO: Incluir tu nombre dentro de la aplicación
st.markdown("**Desarrollado por:** *[Jose Eduardo Sabillon]*")
st.markdown("---")

st.write("Sube una imagen o toma una foto para que el modelo de IA identifique el objeto.")

# Opción para subir imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)
    
    # Preprocesar la imagen para el modelo (32x32 píxeles)
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    
    # Si la imagen tiene canal alfa (RGBA), convertirla a RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    
    # Realizar predicción
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    st.markdown("### Resultados del Análisis:")
    st.success(f"**Predicción:** {predicted_class}")
    st.info(f"**Confianza:** {confidence * 100:.2f}%")
