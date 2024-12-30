import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os
import pyodbc  # For database connection

# --- Database Connection Setup ---
def connect_to_db():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=rfosqlserver.database.windows.net;"
        "DATABASE=master;"
        "UID=rfotest;"
        "PWD=BTColombia2022.;"
    )
    return pyodbc.connect(conn_str)

# Function to insert tags into the database
def insert_tags_to_db(doc_name, tags):
    conn = connect_to_db()
    cursor = conn.cursor()

    for tag in tags:
        try:
            cursor.execute(
                "INSERT INTO dbo.EquTag (DocumentName, tag) VALUES (?, ?)",
                doc_name,
                tag
            )
        except pyodbc.IntegrityError:
            st.warning(f"Duplicate entry for DocumentName: {doc_name}")
    
    conn.commit()
    conn.close()

# --- Main Application ---
st.set_page_config(
    page_title="corsarious",
    layout="wide",
    page_icon="🧑‍⚕️"
)

reader = easyocr.Reader(['en'], verbose=True)
model_path = "yolov5s.pt"
model = YOLO(model_path)

st.title("P&ID Instrumentation and Symbol Detection")
uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png", "PNG"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = img.copy()
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR")

    # YOLO Detection
    st.subheader("Symbol Detection with YOLOv5 (yolov5s.pt)")
    results = model(img)

    # Annotate and extract tags
    tags = []
    for *xyxy, conf, cls in results[0].boxes.data:
        label = model.names[int(cls)]
        x_min, y_min, x_max, y_max = map(int, xyxy)
        tags.append(label)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    st.image(img, caption="YOLO Annotated Image", use_column_width=True)

    # EasyOCR and Shape Detection
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    instrument_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 500 and 50 < h < 500:
            instrument_shapes.append((x, y, w, h))
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(original_img, channels="BGR")
    cols = st.columns(3)

    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        tags.append(extracted_text)
        with cols[i % 3]:
            st.image(cropped_shape, caption=f"Shape {i + 1}")
            st.write(f"Text: {extracted_text}")

    # Insert tags into database
    document_name = os.path.splitext(uploaded_file.name)[0]
    insert_tags_to_db(document_name, tags)
    st.success(f"Tags have been successfully added to the database for Document: {document_name}")
