import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import json

# --- Configuration ---
st.set_page_config(
    page_title="Natural Scene Classifier",
    page_icon="🏞️",
    layout="centered"
)


# --- Load Model and Class Labels ---
# This function now builds the model structure and loads the pre-trained weights.
@st.cache_resource
def load_keras_model():
    """Build the model architecture and load the pre-trained weights."""
    try:
        # 1. Recreate the exact same model architecture from your training script
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            
            Dense(128, activation='relu'),
            Dense(6, activation='softmax') # 6 classes
        ])
        
        # 2. Load the weights from the converted file
        # Make sure 'model_weights.weights.h5' is in the same directory
        model.load_weights('model_weights.weights.h5')
        
        # It's important to compile the model after loading weights,
        # otherwise, it won't be able to make predictions.
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
                      
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure you have run the conversion script and the 'model_weights.weights.h5' file exists.")
        return None

@st.cache_data
def load_class_labels():
    """Load the class labels from JSON."""
    try:
        with open('class_labels.json', 'r') as f:
            labels_map = json.load(f)
            return {int(k): v for k, v in labels_map.items()}
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocess the uploaded image to fit model input requirements."""
    img = image.resize((150, 150))
    # Ensure image is in RGB format, discarding alpha channel if present
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main App Interface ---
st.title("🏞️ CNN for Natural Scene Classification")
st.write("By Alex Fried and Wyatt Lindseth")
st.markdown(
    """
    Upload an image of a natural scene, and this app will predict its category 
    using a Convolutional Neural Network (CNN). The model is trained to recognize 
    six types of scenes: **buildings, forest, glacier, mountain, sea, and street.**
    """
)

# Load resources
model = load_keras_model()
class_labels = load_class_labels()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)

if model and class_labels and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_label = class_labels.get(predicted_class_index, "Unknown Class").replace('_', ' ').capitalize()

    st.success(f"**Prediction:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    st.write("### Prediction Probabilities")
    prob_dict = {class_labels[i].capitalize(): prediction[0][i] for i in range(len(class_labels))}
    st.bar_chart(prob_dict)

elif not model or not class_labels:
    st.warning("Waiting for the model and class labels to load...")
