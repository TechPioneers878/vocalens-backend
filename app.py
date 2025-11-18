# backend/app.py
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---- Load environment variables from .env (for local development only) ----
load_dotenv()  # Render ignores this, but locally it loads your .env file

# ---- CONFIGURE ----
# IMPORTANT: API key must ONLY come from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash"

# Fail fast if API key is missing
if not GEMINI_API_KEY:
    raise RuntimeError("❌ ERROR: GEMINI_API_KEY is NOT set in environment variables!")

# ---- Init ----
app = Flask(__name__)
CORS(app)  # Allow all during development; restrict in production

client = genai.Client(api_key=GEMINI_API_KEY)

@app.route("/api/gemini-proxy", methods=["POST"])
def gemini_proxy():
    """
    Accepts multipart/form-data:
      - images: one or more JPEG/PNG files (field name 'images')
      - query: text field
    Forwards everything to Gemini model and returns the text response.
    """
    query = request.form.get("query", "").strip()
    files = request.files.getlist("images")

    if not query and not files:
        return jsonify({"error": "Provide a query or at least one image"}), 400

    contents = []
    if query:
        contents.append(query)

    for f in files:
        data = f.read()
        contents.append(types.Part.from_bytes(data=data, mime_type=f.mimetype))

    try:
        # Send content to Gemini API
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents
        )

        # Extract only text parts
        text_out = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_out += part.text

        return jsonify({"result_text": text_out}), 200

    except Exception as e:
        return jsonify({
            "error": "Vendor request failed",
            "details": str(e)
        }), 502


if __name__ == "__main__":
    # LOCAL dev server (do NOT use in production)
    app.run(host="0.0.0.0", port=5000, debug=True)
