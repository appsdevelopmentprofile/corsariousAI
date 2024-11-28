import streamlit as st
import os
import pytesseract
from PIL import Image
from gtts import gTTS
import speech_recognition as sr
import tempfile
import re

# Set up the Streamlit app
st.title("Document Submission Prototype")

# File upload section
uploaded_file = st.file_uploader("Upload your file (e.g., 1-1_2-IN-70R902_CNPI25E_Cleanliness_and_Drying_Summary_19079_page_1.jpg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join("/content", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    # Rename the file
    renamed_file = "Braya_Checklist_Cleanliness_and_Drying_Piping_170R902.jpg"
    renamed_path = os.path.join("/content", renamed_file)
    os.rename(file_path, renamed_path)
    st.success(f"File renamed to: {renamed_file}")

    # Perform OCR to extract text under the label "Task"
    image = Image.open(renamed_path)
    text = pytesseract.image_to_string(image)

    # Simulate extracting specific text under "Task"
    task_label = "Task"
    extracted_text = ""
    if task_label in text:
        task_index = text.index(task_label) + len(task_label)
        extracted_text = text[task_index:].strip().split("\n")[0]
    else:
        st.error("Unable to find 'Task' in the document.")

    if extracted_text:
        st.write("Extracted Text under 'Task':")
        st.write(extracted_text)

        # Generate MP3 from the extracted text
        mp3_file = "/content/task_audio.mp3"
        tts = gTTS(text=extracted_text, lang='en')
        tts.save(mp3_file)

        # Display the MP3 file in Streamlit
        st.audio(mp3_file, format="audio/mp3")
        st.success("MP3 file generated and ready to play!")

        # Record engineer's responses
        st.write("Please answer the questions. Recording will stop when you say 'Q COMPLETED'.")
        if st.button("Start Recording"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Recording... Please speak clearly.")
                audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=30)

                try:
                    response_text = recognizer.recognize_google(audio_data)
                    st.write("Captured Response:", response_text)

                    if "Q COMPLETED" in response_text:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
                            temp_audio = gTTS(text=response_text.replace("Q COMPLETED", ""), lang='en')
                            temp_audio.save(temp_mp3.name)
                            st.audio(temp_mp3.name, format="audio/mp3")
                            st.success("Response saved and converted to MP3.")

                        # Tag responses with Yes or No and fill the checkboxes
                        response_tags = {
                            "Q1: What is the project?": None,
                            "Q2: What is the type of document?": None,
                            "Q3: What is the operation?": None,
                            "Q4: What is the type of equipment?": None,
                            "Q5: What is the label of the equipment (or tag of the process)?: None"
                        }

                        # Assuming that responses are in the format "Q1: ... Yes/No"
                        for question, _ in response_tags.items():
                            if question in response_text:
                                response_tags[question] = "Yes" if "Yes" in response_text else "No"

                        # Confirm checkbox filling
                        st.write("Responses tagged with Yes/No:")
                        for question, answer in response_tags.items():
                            st.write(f"{question}: {answer}")

                        # Confirm document saving
                        st.success("Document has been saved, autofill with Virtual Assistant completed.")

                except sr.UnknownValueError:
                    st.error("Could not understand the audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"Speech Recognition error: {e}")
