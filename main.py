import os
import base64
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from pydantic_settings import BaseSettings
from io import BytesIO
from PIL import Image

class Settings(BaseSettings):
    gemini_api_key: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(title="Image Describer App")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize Gemini Client
if settings.gemini_api_key:
    client = genai.Client(api_key=settings.gemini_api_key)
else:
    client = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if not client:
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Gemini API key is not configured. Please add it to your .env file."
         })
         
    if not file.content_type.startswith("image/"):
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Uploaded file is not an image. Please upload a valid image file."
         })

    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Load the image using PIL
        image = Image.open(BytesIO(contents))
        
        # Encode image as base64 for display in template
        image_b64 = base64.b64encode(contents).decode('utf-8')
        image_data_uri = f"data:{file.content_type};base64,{image_b64}"
        
        # We pass the PIL image directly to the generate_content method
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=['Describe this image in detail.', image]
        )
        
        description = response.text
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "description": description,
            "filename": file.filename,
            "image_data": image_data_uri
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred: {str(e)}"
        })
