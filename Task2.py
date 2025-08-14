import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load your audio file
with sr.AudioFile("sample.wav") as source:
    print("Listening...")
    audio = recognizer.record(source)

# Try converting speech to text
try:
    text = recognizer.recognize_google(audio)
    print("\n--- Transcribed Text ---\n")
    print(text)

except sr.UnknownValueError:
    print("Sorry, could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results; {e}")
