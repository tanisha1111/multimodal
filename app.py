import streamlit as st
import requests
from PIL import Image
import pytesseract
import soundfile as sf
from requests.exceptions import SSLError
import time

# Constants
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models"
HUGGINGFACE_API_KEY = "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"

# App Title
st.title("Lightweight Multimodal Analysis App")
st.sidebar.title("Select a Feature")

# Sidebar Navigation
feature = st.sidebar.radio(
    "Choose a functionality:",
    ("Text Translation", "Image OCR", "Audio-to-Text", "Sentiment Analysis"),
)

def query_huggingface_api(endpoint, payload):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    retries = 3  # Number of retry attempts
    for _ in range(retries):
        try:
            # Making the API request
            response = requests.post(f"{HUGGINGFACE_API_URL}/{endpoint}", headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()  # Return JSON response
        except SSLError as e:
            print(f"SSL Error: {e}. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    return None  # Return None if the API request fails after retries

if feature == "Text Translation":
    st.header("Text Translation")
    
    # Dropdowns for selecting source and target languages
    st.subheader("Select Languages")
    source_language = st.selectbox("Source Language:", ["English", "French", "Spanish", "German", "Italian", "Russian", "Chinese", "Japanese", "Hindi"])
    target_language = st.selectbox("Target Language:", ["French", "Spanish", "German", "Italian", "Russian", "Chinese", "Japanese", "English", "Hindi"])
    
    input_text = st.text_area("Enter text to translate:")
    
    # Define model mapping for source-target pairs
    model_mapping = {
        ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
        ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
        ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
        ("English", "Italian"): "Helsinki-NLP/opus-mt-en-it",
        ("English", "Russian"): "Helsinki-NLP/opus-mt-en-ru",
        ("English", "Chinese"): "Helsinki-NLP/opus-mt-en-zh",
        ("English", "Japanese"): "Helsinki-NLP/opus-mt-en-ja",
        ("English", "Hindi"): "Helsinki-NLP/opus-mt-en-hi",
        ("Hindi", "English"): "Helsinki-NLP/opus-mt-hi-en",
        ("French", "English"): "Helsinki-NLP/opus-mt-fr-en",
        ("Spanish", "English"): "Helsinki-NLP/opus-mt-es-en",
        ("German", "English"): "Helsinki-NLP/opus-mt-de-en",
        ("Italian", "English"): "Helsinki-NLP/opus-mt-it-en",
        ("Russian", "English"): "Helsinki-NLP/opus-mt-ru-en",
        ("Chinese", "English"): "Helsinki-NLP/opus-mt-zh-en",
        ("Japanese", "English"): "Helsinki-NLP/opus-mt-ja-en",
    }
    
    if st.button("Translate"):
        if not input_text:
            st.error("Please enter some text to translate.")
        elif (source_language, target_language) not in model_mapping:
            st.error("Translation for the selected language pair is not supported.")
        else:
            # Get the model name for the selected language pair
            model_name = model_mapping[(source_language, target_language)]
            
            # Call the Hugging Face API
            translation = query_huggingface_api(model_name, {"inputs": input_text})
            if "error" in translation:
                st.error(f"Error during translation: {translation['error']}")
            else:
                st.success(f"Translated Text ({target_language}): {translation[0]['translation_text']}")

elif feature == "Image OCR":
    st.header("Image OCR")
    uploaded_image = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"])
    if st.button("Extract Text"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            extracted_text = pytesseract.image_to_string(image)
            st.write(f"**Extracted Text:** {extracted_text}")
        else:
            st.error("Please upload an image.")

elif feature == "Audio-to-Text":
    st.header("Audio-to-Text")
    uploaded_audio = st.file_uploader("Upload an Audio File:", type=["wav", "mp3"])

    if st.button("Convert Audio to Text"):
        if uploaded_audio:
            # Read the audio file using soundfile and convert to base64
            audio_input, sample_rate = sf.read(uploaded_audio)
            audio_json = {"inputs": audio_input.tolist()}
            
            # Call the Hugging Face ASR API with retry logic
            asr_result = query_huggingface_api("openai/whisper-large-v2", audio_json)

            if asr_result:
                st.write(f"**Recognized Text:** {asr_result.get('text', 'No text recognized')}")
            else:
                st.error("Error processing the audio file. Please try again later.")
        else:
            st.error("Please upload an audio file.")

elif feature == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    input_text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if not input_text:
            st.error("Please enter some text.")
        else:
            emotions = query_huggingface_api("j-hartmann/emotion-english-distilroberta-base", {"inputs": input_text})
            if emotions and isinstance(emotions[0], list):
                # Flatten the list and find the dictionary with the highest score
                flattened_emotions = emotions[0]
                detected_emotion = max(flattened_emotions, key=lambda x: x["score"])["label"]
                st.success(f"Detected Emotion: {detected_emotion}")
            else:
                st.error("Invalid emotions data structure!")
