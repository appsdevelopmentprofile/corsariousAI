import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os
import pickle
from pathlib import Path
import streamlit_authenticator as stauth


# --- Set page configuration ---
st.set_page_config(
    page_title="corsarious",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --- USER AUTHENTICATION ---
names = ["corsarious"]
usernames = ["corsarious"]

# Load hashed passwords
file_path = Path("hashed_pw.pkl")
try:
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)
except FileNotFoundError:
    st.error("Password file not found.")
except Exception as e:
    st.error(f"An error occurred while loading passwords: {e}")

authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords, "corsarious", "corsarious", cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", "main")

# If the user clicks "Logout", update session to reflect the logout
if "logged_out" not in st.session_state:
    st.session_state["logged_out"] = False  # Initial state for tracking logout

if authentication_status == False:
    st.error("Username/password is incorrect")

elif authentication_status == None:
    st.warning("Please enter your username and password")

elif authentication_status:
    # ---- LOGOUT AND GREETING ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    # Sidebar
    selected = st.sidebar.selectbox("Select Module", ["Doc Intelligence", "Field AI Assistant", "AI Testing"])

        # Split layout into two columns: left for selected content, right for chatbot
    col1, col2 = st.columns([2, 1])

    with col1:

    
        # Main Modules
        if selected == "Doc Intelligence":
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], verbose=True)
            
            # Define the path to the YOLO model file (assuming it's in the same directory as the script)
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            
            # Load the YOLO model
            model = YOLO(model_path)
            
            # Streamlit app title
            st.title("P&ID Instrumentation and Symbol Detection")
            
            # File uploader for image input
            uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png", "PNG"])
            
            if uploaded_file is not None:
                # Read the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                original_img = img.copy()
            
                # Display the uploaded image
                st.subheader("Uploaded Image:")
                st.image(img, channels="BGR")
            
                # ONNX Symbol Detection (Using the YOLO model)
                st.subheader("Symbol Detection with YOLO (best.pt)")
            
                # Perform inference with the YOLO model
                results = model(img)
            
                # Display the results
                st.subheader("Detection Results:")
                
                # Access bounding boxes, labels, and confidence scores
                for *xyxy, conf, cls in results[0].boxes.data:  # Get bounding boxes and other info
                    label = model.names[int(cls)]
                    x_min, y_min, x_max, y_max = map(int, xyxy)  # Get bounding box coordinates
                    st.write(f"Detected: {label} with confidence {conf:.2f}")
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
                # Display annotated image with YOLO results
                st.image(img, caption="YOLO Annotated Image", use_column_width=True)
            
                # EasyOCR Text Detection and Instrument Shapes
                st.subheader("Text Extraction and Shape Detection")
            
                # Preprocessing for contours
                gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(edges, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                # Detect and annotate instrument shapes
                instrument_shapes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 50 < w < 500 and 50 < h < 500:  # Adjust thresholds as needed
                        instrument_shapes.append((x, y, w, h))
                        cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
                # Detect circles using Hough Circle Transform
                gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
                circles = cv2.HoughCircles(
                    gray_blur,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=50,
                    param1=50,
                    param2=30,
                    minRadius=10,
                    maxRadius=50
                )
            
                # Draw circles on the original image
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        center = (circle[0], circle[1])  # x, y center
                        radius = circle[2]  # radius
                        cv2.circle(original_img, center, radius, (0, 255, 0), 2)
            
                # Display detected shapes and text
                st.subheader("Processed Image with Detected Shapes and Circles")
                st.image(original_img, channels="BGR")
            
                # Extract text from detected shapes
                st.subheader("Extracted Text from Detected Shapes and Circles")
                cols = st.columns(3)
            
                for i, (x, y, w, h) in enumerate(instrument_shapes):
                    cropped_shape = img[y:y + h, x:x + w]
                    text = reader.readtext(cropped_shape, detail=0)
                    extracted_text = " ".join(text) if text else "No text detected"
                    with cols[i % 3]:
                        st.image(cropped_shape, caption=f"Shape {i + 1}")
                        st.write(f"Text: {extracted_text}")
            
                if circles is not None:
                    for i, circle in enumerate(circles[0, :]):
                        x, y, r = circle
                        cropped_circle = original_img[y-r:y+r, x-r:x+r]
                        if cropped_circle.size > 0:
                            text = reader.readtext(cropped_circle, detail=0)
                            extracted_text = " ".join(text) if text else "No text detected"
                            with cols[(i + len(instrument_shapes)) % 3]:
                                st.image(cropped_circle, caption=f"Circle {i + 1}")
                                st.write(f"Text: {extracted_text}")
        
        elif selected == "Field AI Assistant":

            import os
            import streamlit as st
            import sounddevice as sd
            import numpy as np
            import scipy.io.wavfile as wav
            from gtts import gTTS
            from playsound import playsound
            from pydub import AudioSegment
            import io
            from docx import Document
            import speech_recognition as sr


            st.write("""There needs to be a pre trained model (only containing the checklist fromats) connected to the data lake

For every document that is added to the Data Lake, there will be a process of rename and auto mp3 generation.

Rename process: designed to have standarization and accurate diagnose. Can be done from OCR to identify tags, or maually from the operator setting the name once it is submitted. The proposed labeling is:
"project/type_of_document/operation/type of equipment/label of the equipment (or tag of the process).pdf"


Auto mp3 generarion: designed to ask the questionaire to the engineer before fillinf the checklist. It will be done with the same YoloV5 capacibilities to identfy objects, and text. Once test is identified, another library will generate the mp3 file with the same name.


WORKFLOW:

SUBMITTING THE CHECLIST FORMATS
Peter is an engineer, he submits the checlkits ot the data lake. The name of the file is "1-1_2-IN-70R902_CNPI25E_Cleanliness_and_Drying_Summary_19079_page_1.jpg"

The pre trained model makes the first text recognitin, with feature recognition labels the file and renames it. For testing purposes, the new file will be named as: Braya_Checklist_Cleanliness_and_Drying_Piping_170R902.jpg

Once it is renamed, the pre trained model does the second text recognition to get the written questions, put them on text and the library generates the mp3 file


QUESTIONAIRE PROCESS FORM ENGINEER

3 weeks later, the project is completed, and Peter is requested to fill the checklist, is -30 in the beautiful province of Alberta

Peter turns on the virtual assisstant: "TURN ON QUESTIONS"

The system will have a mp3 with a present questionaire with 5 questions

# Question 1: what is the project?
# Question 2: what is the type of document?
# Quesion 3: what is the operation?
# Question 4: What is the type of equipment?
# Question 5: what is the label of the equipment (or tag of the process)?


EXAMPLE (Peter responds):


# Question 1: Braya
# Quesion 2: Checklist
# Question 3: Cleanliness_and_Drying
# Question 4: Piping
# Question 5: 170R902

Based on the responsed of the engineer, the system identifies from all thouands of documents that the engineers needs the next checklist: "Braya_Checklist_Cleanliness_and_Drying_Piping_170R902.jpg"

Then, the system automatically reproduces the MP3 qie the checklist questions: Braya_Checklist_Cleanliness_and_Drying_Piping_170R902.mp3""")
            
            # Function to play text-to-speech using gTTS and sounddevice
            def speak(text):
                tts = gTTS(text=text, lang='en')
                tts.save("output.mp3")
                playsound("output.mp3")
            
            # Function to record audio using sounddevice
            def record_audio(duration=10, sample_rate=16000):
                st.write("Recording... Speak now.")
                audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
                sd.wait()  # Wait for the recording to finish
                st.write("Recording finished.")
                return audio, sample_rate
            
            # Function to save recorded audio to a WAV file
            def save_audio(audio, sample_rate, filename="recorded.wav"):
                wav.write(filename, sample_rate, audio)
            
            # Function to transcribe speech using SpeechRecognition
            def transcribe_audio(filename):
                recognizer = sr.Recognizer()
                with sr.AudioFile(filename) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        return text
                    except sr.UnknownValueError:
                        return "Could not understand the audio."
                    except sr.RequestError as e:
                        return f"Error with speech recognition service: {e}"
            
            # Function to match equipment keywords
            def match_equipment(response, equipment_keywords):
                response = response.lower()
                for equipment, keywords in equipment_keywords.items():
                    if any(keyword in response for keyword in keywords):
                        return equipment
                return None
            
            # Main questionnaire function
            def field_engineer_questionnaire():
                st.title("Field Engineer Questionnaire")
            
                # Equipment categories and associated keywords
                equipment_keywords = {
                    "electric unit heater": ["electric", "unit", "heater"],
                    "air intake": ["air", "intake"],
                    "ridge vent": ["ridge", "vent"],
                    "exhaust air fan": ["exhaust", "fan"]
                }
            
                # Questions for identification
                identification_questions = [
                    "What is the equipment name?",
                    "What is the equipment ID?",
                    "What is the current status of the equipment? (Working or Faulty)"
                ]
            
                responses = {}
            
                for question in identification_questions:
                    speak(question)
                    st.write(question)
            
                    # Record audio response
                    if st.button(f"Record for: {question}"):
                        audio, sample_rate = record_audio()
                        save_audio(audio, sample_rate, "response.wav")
                        response_text = transcribe_audio("response.wav")
                        responses[question] = response_text
                        st.write(f"Recorded response: {response_text}")
            
                # Match the response to the equipment
                equipment_name = responses.get("What is the equipment name?", "").lower()
                matched_equipment = match_equipment(equipment_name, equipment_keywords)
            
                if matched_equipment:
                    st.write(f"Identified equipment: {matched_equipment}")
                    speak(f"The identified equipment is {matched_equipment}.")
                    st.write("Proceeding to the next stage for follow-up questions...")
                else:
                    st.write("Equipment type not recognized or supported.")
                    speak("Equipment type not recognized or supported.")
            
                st.write("All responses recorded:", responses)
                return responses
            
            # Follow-up questions stage
            def follow_up_questions_stage(equipment_name):
                st.title("Follow-Up Questions Stage")
            
                # Find MP3 file
                def find_mp3_file(equipment_name, base_path="local_dataset/mp3"):
                    mp3_filename = f"{equipment_name.replace(' ', '_')}.mp3"
                    mp3_path = os.path.join(base_path, mp3_filename)
                    if os.path.isfile(mp3_path):
                        return mp3_path
                    else:
                        st.write("No MP3 file found for the identified equipment.")
                        return None
            
                # Play MP3
                mp3_path = find_mp3_file(equipment_name)
                if mp3_path:
                    playsound(mp3_path)
            
                # Record response
                audio, sample_rate = record_audio()
                save_audio(audio, sample_rate, "follow_up.wav")
                response_text = transcribe_audio("follow_up.wav")
                return response_text
            
            # Main Streamlit application
            if __name__ == "__main__":
                st.sidebar.title("Navigation")
                app_mode = st.sidebar.selectbox("Choose the app mode", ["Questionnaire", "Follow-Up"])
            
                if app_mode == "Questionnaire":
                    field_engineer_questionnaire()
                elif app_mode == "Follow-Up":
                    example_equipment = "air intake"  # Example placeholder
                    follow_up_response = follow_up_questions_stage(example_equipment)
                    st.write(f"Final recorded response: {follow_up_response}")


            
    with col2:
                # --- CHATBOT IN RIGHT COLUMN ---
        st.subheader(f"Hello {name}, I am your upskiller!")
        
        # OpenAI API Key input
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Let's start with the demo. Write 'start' and click enter."}
            ]
        
        # Restart Chat button
        if st.button("Restart Chat"):
            # Clear chat messages to restart the conversation
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Let's start with the demo. Write 'start' and click enter."}
            ]
        
        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Accept user input
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
        
            # Send user message
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
        
            # Define bot response logic
            if prompt.strip().lower() == "start":
                # Start demo process
                msg = "Thank you! Let's add some sample files for demoing purposes. Do you have Orthos or Lidar files?"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                
                # Display "Yes" and "No" buttons
                has_files = st.radio("Do you have Orthos or Lidar files?", ["Yes", "No"])
                
                if has_files == "No":
                    # If customer doesn't have files, display links
                    no_files_response = (
                        "No problem! You can use these sample datasets:\n\n"
                        f"**TIFF**: [SAIT Orthos TIFF file](https://mysait-my.sharepoint.com/:i:/r/personal/rick_duchscher_sait_ca/Documents/UAV%20Projects/SAIT%20OGL%202018/SAIT_OGL_Orthos_2018/R004_C001.tif?csf=1&web=1&e=Q0f8v7)\n"
                        f"**LAS**: [SAIT Lidar LAS file](https://mysait-my.sharepoint.com/:u:/r/personal/rick_duchscher_sait_ca/Documents/UAV%20Projects/SAIT%20OGL%202018/SAIT_OGL_Lidar_2018/Colorized/Sait_CIR_NonGround.las?csf=1&web=1&e=Ky9ftA)"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": no_files_response})
                    st.chat_message("assistant").write(no_files_response)
        
                elif has_files == "Yes":
                    # If customer has files, prompt for the file location link
                    file_location_response = "Great! Please enter the link to where your files are located."
                    st.session_state.messages.append({"role": "assistant", "content": file_location_response})
                    st.chat_message("assistant").write(file_location_response)
        
                    # Display text box for customer to input file location link
                    user_file_link = st.text_input("Enter your file location link here:")
                    if user_file_link:
                        st.session_state.messages.append({"role": "user", "content": f"Customer provided file link: {user_file_link}"})
                        st.chat_message("user").write(f"Customer provided file link: {user_file_link}")
                        
                        # Proceed with using `user_file_link` in your main app functionality as needed
        
            else:
                # Other non-start responses - get response from OpenAI
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
