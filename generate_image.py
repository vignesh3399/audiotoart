import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
from PIL import Image

# Log in to Hugging Face with your token (use the token you copied from your Hugging Face account)
login("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")

# Load the model pipeline with your Hugging Face token
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    use_auth_token=True  # This ensures token is used when loading the model
)

# Move model to GPU if available for faster processing
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate an image based on a text prompt
def generate_image_from_text(prompt, size=(512, 512)):
    # Generate the image from the prompt
    image = pipe(prompt).images[0]

    # Resize the image if needed
    image = image.resize(size)

    # Show and save the image
    image.show()
    image.save("generated_image.png")
    print(f"Image saved as 'generated_image.png'.")

# Speech-to-text function using the SpeechRecognition library
def get_speech_input():
    recognizer = sr.Recognizer()
    
    # Use the microphone as the source for audio
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise with 1 second duration
        print("Adjusting for background noise...")
        audio = recognizer.listen(source)  # Capture the audio

        try:
            # Recognize the speech using Google Web Speech API
            speech_text = recognizer.recognize_google(audio)
            print(f"Speech recognized: {speech_text}")
            return speech_text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# Main function that ties everything together
def main():
    prompt = get_speech_input()  # Get speech input from the user
    if prompt:  # If we successfully get the speech text
        generate_image_from_text(prompt)
    else:
        print("No valid speech input detected, cannot generate image.")

if __name__ == "__main__":
    main()
