# Image Describer Web Application

A web application that allows users to upload an image and uses the Google Gemini AI model to describe its contents. Built with Python, FastAPI, and the Google GenAI SDK.

## Prerequisites
- Python 3.9+
- A Google Gemini API Key

## Setup and Installation

1. **Navigate to the project directory:**
   ```bash
   cd c:/use/gulp/image-describer
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**
   - Copy `.env.example` to a new file named `.env` (or simply create a `.env` file).
   - Open `.env` and paste your actual Gemini API key:
     ```env
     GEMINI_API_KEY=your_actual_api_key_here
     ```

4. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`.

## Features
- Clean and responsive UI built with Tailwind CSS.
- Drag-and-drop file upload for images.
- Integration with `gemini-2.5-flash` model using the new `google-genai` SDK.
- Handles various image formats (PNG, JPG, WEBP).
