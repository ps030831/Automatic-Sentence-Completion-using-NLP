import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import speech_recognition as sr

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Streamlit App
st.title("Let Me Do It For You")

# Create an option for text input method (typing or speaking)
input_method = st.radio("Choose input method:", ("Type", "Speak"))

# Initialize input_text as an empty string
input_text = ""

# Function to recognize speech and return it as text
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    try:
        st.write("Processing speech...")
        # Recognize speech using Google's speech recognition API
        recognized_text = recognizer.recognize_google(audio)
        return recognized_text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that. Please try again."
    except sr.RequestError:
        return "Request to Google Speech Recognition failed."

# If the user chooses to type, display a text input box
if input_method == "Type":
    input_text = st.text_input("Type some text:")

# If the user chooses to speak, display a button to record speech
elif input_method == "Speak":
    if st.button("Speak Now"):
        input_text = recognize_speech()
        st.write(f"You said: **{input_text}**")

# User input for the number of words to predict
num_words = st.number_input("Number of words to predict", min_value=1, max_value=50, value=5)

# Function to predict logical next words with top-k and temperature control
def predict_next_words(input_text, num_words, top_k=50, temperature=0.7, num_return_sequences=4):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate the next num_words tokens with top-k sampling and temperature control
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + num_words,
            num_return_sequences=num_return_sequences,  # Return 4 different sequences
            do_sample=True,              # Enable sampling for more natural results
            top_k=top_k,                 # Top-k sampling to limit token choices
            temperature=temperature,     # Control randomness
            eos_token_id=tokenizer.eos_token_id,  # Stop at the end of a sentence
            pad_token_id=tokenizer.eos_token_id   # Ensure sentences are padded logically
        )
    
    # Decode the generated sequences and return the result
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True)[len(input_text):].strip() for seq in output]
    return generated_texts

# Only run the prediction if there is input text
if input_text:
    st.write("Generating four logical continuations...")
    # Generate the text
    generated_texts = predict_next_words(input_text, num_words)
    
    # Display the four options
    for i, option in enumerate(generated_texts, 1):
        st.write(f"Option {i}: **{option}**")
