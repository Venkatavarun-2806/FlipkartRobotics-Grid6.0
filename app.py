import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import torch
from paddleocr import PaddleOCR
import cv2
import os
import re

# Load models
tensorflow_model = load_model('m.h5')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
ocr = PaddleOCR(use_angle_cls=True, lang='en', cpu_threads=1)

# Class names for TensorFlow model
class_names = {
    0: 'fresh_apple',
    1: 'fresh_banana',
    2: 'fresh_bitter_gourd',
    3: 'fresh_capsicum',
    4: 'fresh_orange',
    5: 'fresh_tomato',
    6: 'stale_apple',
    7: 'stale_banana',
    8: 'stale_bitter_gourd',
    9: 'stale_capsicum',
    10: 'stale_orange',
    11: 'stale_tomato'
}

# Helper functions
def preprocess_image(image_file):
    img = load_img(image_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_expiry_date(text):
    date_patterns = [
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Expiry Date: 20/07/2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Expiry Date: 20 MAY 2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 2024/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Expiry Date: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}))',  # Best Before: 2025
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Best Before: 20/07/2024
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Best Before: 20 MAY 2024
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 2024/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Best Before: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))', 
    r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}[0O]\d{2}))',  # Consume Before: 3ODEC2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}\d{2}))',  # Consume Before: 30DEC23
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Consume Before: 20/07/2024
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Consume Before: 20 MAY 2024
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 2024/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Consume Before: 2024/07/20
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Exp: 20/07/2024
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Exp: 20 MAY 2024
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 2024/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Exp: 2024/07/20
    r"Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})",
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 16 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp. Date: 15/12/2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 15 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date cdsyubfuyef 15 MAR 2O30 (with typo)
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{2})',  # 20/07/24
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{2})',  # 20 MAY 24
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024-07-20
    r'(\d{4}[A-Za-z]{3,}\d{2})',  # 2024MAY20
    r'(\d{2}[A-Za-z]{3,}\d{4})',  # 20MAY2024
    r'(?:DX3\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
    r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
    r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
    r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',
    r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',
    r"Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})",
    r"EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]"
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    return "No expiry date found"

def perform_ocr_on_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    ocr_results = ocr.ocr(img)
    text = " ".join([word[1][0] for line in ocr_results for word in line])
    expiry_date = extract_expiry_date(text)
    return text, expiry_date

# Streamlit UI
st.title("Flipkart Robotics Services")

# Sidebar navigation
current_task = st.session_state.get('current_task', 'Home')
task = st.sidebar.radio("Choose a task", ["Home", "Object Detection", "Text Extraction", "Freshness Detection"])

# Clear uploaded file and query params when task changes
if task != current_task:
    st.session_state['current_task'] = task
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file']
    st.query_params.clear()
    st.rerun()

# Handle page reload using query parameters
query_params = st.query_params
uploaded = query_params.get("uploaded", False)

if task == "Home":
    st.header("Welcome to Flipkart Robotics Services")
    st.markdown("""
    - *Object Detection*: Detect objects in an image.
    - *Text Extraction*: Extract text and expiry dates from an image.
    - *Freshness Detection*: Classify the freshness of fruits or vegetables.
    """)

elif task == "Object Detection":
    st.header("Object Detection")
    st.markdown("""
    - This Detect the object in the images.
    - The objects like Person,Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck,Bird, Cat, Dog, Horse, Sheep, Cow, Backpack, Umbrella, Handbag, Tie, Suitcase,Bottle
    """)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], 
                                   on_change=lambda: st.query_params.update({"uploaded": True}),
                                   key="object_detection_uploader")

    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = yolo_model(img)
        results.render()
        st.image(results.ims[0], caption="Detected Objects", use_column_width=True)
        st.success(f"Detected {len(results.xyxy[0])} objects.")
        st.query_params.clear()

elif task == "Text Extraction":
    st.header("Text Extraction and Expiry Date")
    st.markdown("""
    - This extract the readable text on the product and also checks the expiry date.
    - Specifically identify expiry dates in various formats:
        - DD/MM/YYYY
        - YYYY/MM/DD
        - DD MMM YYYY
    """)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], 
                                   on_change=lambda: st.query_params.update({"uploaded": True}),
                                   key="text_extraction_uploader")

    if uploaded_file:
        extracted_text, expiry_date = perform_ocr_on_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.subheader("Extracted Text")
        st.write(extracted_text)
        st.subheader("Expiry Date")
        st.write(expiry_date)
        st.query_params.clear()

elif task == "Freshness Detection":
    st.header("Freshness Detection")
    st.markdown("""
    - Analyze fruits and vegetables for freshness and Classify items into fresh or stale categories
    - It perdicts Apples,Bananas,Bitter Gourds,Capsicums,Oranges,Tomatoes 
    """)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], 
                                   on_change=lambda: st.query_params.update({"uploaded": True}),
                                   key="freshness_detection_uploader")

    if uploaded_file:
        img = preprocess_image(uploaded_file)
        prediction = tensorflow_model.predict(img)
        class_index = np.argmax(prediction, axis=-1)[0]
        class_name = class_names.get(class_index, "Unknown")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.subheader("Freshness Status")
        st.write(f"Class: {class_name}")
        st.query_params.clear()