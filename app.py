from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import requests
import os
import PyPDF2

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
app.config["UPLOAD_FOLDER"] = "uploads"
HUGGINGFACE_API_KEY = "your-huggingface-token"  # Free API key from Huggingface

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract text from PDF
    text = extract_text(filepath)

    # Generate notes and MCQs using Hugging Face API
    notes = generate_summary(text)
    mcqs = generate_questions(text)

    return jsonify({"notes": notes, "mcqs": mcqs})


def extract_text(filepath):
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def generate_summary(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()[0]["summary_text"]


def generate_questions(text):
    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json={"inputs": f"Generate 5 multiple choice questions about: {text}"},
    )
    return response.json()[0]["generated_text"]


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
