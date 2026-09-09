import os
import base64
from datetime import datetime
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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

# In-memory history for dashboard
analysis_history = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, tab: str = "home"):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": analysis_history,
        "active_tab": tab
    })

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if not client:
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Gemini API key is not configured. Please add it to your .env file.",
            "history": analysis_history,
            "active_tab": "describer"
         })
         
    if not file.content_type.startswith("image/"):
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Uploaded file is not an image. Please upload a valid image file (PNG, JPG, JPEG, WEBP).",
            "history": analysis_history,
            "active_tab": "describer"
         })

    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Load the image using PIL
        image = Image.open(BytesIO(contents))
        
        # Encode image as base64 for display in template
        image_b64 = base64.b64encode(contents).decode('utf-8')
        image_data_uri = f"data:{file.content_type};base64,{image_b64}"
        
        # Pass image to Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=['Describe this image in detail. Highlight the main subjects, action, colors, background, and mood.', image]
        )
        
        description = response.text
        timestamp = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        
        # Add to dashboard history (latest first)
        history_item = {
            "id": len(analysis_history) + 1,
            "filename": file.filename,
            "timestamp": timestamp,
            "description": description,
            "image_data": image_data_uri,
            "snippet": description[:140] + "..." if len(description) > 140 else description
        }
        analysis_history.insert(0, history_item)
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "description": description,
            "filename": file.filename,
            "image_data": image_data_uri,
            "history": analysis_history,
            "active_tab": "describer"
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred: {str(e)}",
            "history": analysis_history,
            "active_tab": "describer"
        })

@app.post("/delete-history/{item_id}")
async def delete_history(item_id: int):
    global analysis_history
    analysis_history = [item for item in analysis_history if item["id"] != item_id]
    return JSONResponse({"success": True})

@app.post("/clear-history")
async def clear_history():
    global analysis_history
    analysis_history.clear()
    return JSONResponse({"success": True})
