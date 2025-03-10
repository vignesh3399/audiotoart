from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
import io
from PIL import Image
import base64

app = Flask(__name__)

# Folder to save generated images
OUTPUT_FOLDER = 'generated_images'

# Ensure the folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Log in to Hugging Face with your token
login("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")

# Load the model pipeline with your Hugging Face token
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
)

# Move model to GPU if available for faster processing
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate an image based on a text prompt
def generate_image_from_text(prompt, size=(512, 512)):
    # Generate the image from the prompt
    image = pipe(prompt).images[0]
    
    # Resize the image if needed
    image = image.resize(size)
    
    # Convert image to a byte stream for sending over the web
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return byte_io

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Generate the image
    image_byte_io = generate_image_from_text(prompt)

    # Save the image to a file
    image_filename = os.path.join(OUTPUT_FOLDER, 'generated_image.png')
    with open(image_filename, 'wb') as f:
        f.write(image_byte_io.getvalue())

    # Return the URL of the saved image
    return jsonify({"image_url": f"/generated_images/generated_image.png"})

# Serve the images folder as static files
@app.route('/generated_images/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
