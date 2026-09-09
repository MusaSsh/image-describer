import os
import sqlite3
import base64
from datetime import datetime
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
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

app = FastAPI(title="VisionIQ — Image Describer & Studio")

# Ensure data directory exists for persistent SQLite database
os.makedirs("data", exist_ok=True)
DB_PATH = "data/history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                description TEXT NOT NULL,
                image_data TEXT NOT NULL,
                snippet TEXT NOT NULL
            )
        """)
        conn.commit()

def get_all_history(search_query: str = ""):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if search_query:
            cursor.execute("""
                SELECT * FROM history 
                WHERE filename LIKE ? OR description LIKE ?
                ORDER BY id DESC
            """, (f"%{search_query}%", f"%{search_query}%"))
        else:
            cursor.execute("SELECT * FROM history ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def insert_history_record(filename: str, timestamp: str, description: str, image_data: str, snippet: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (filename, timestamp, description, image_data, snippet)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, timestamp, description, image_data, snippet))
        conn.commit()
        return cursor.lastrowid

def delete_history_record(item_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ?", (item_id,))
        conn.commit()

def clear_all_history_records():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()

# Initialize DB on startup
init_db()

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
async def read_root(request: Request, tab: str = "home", q: str = ""):
    history = get_all_history(q)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": history,
        "active_tab": tab,
        "search_query": q
    })

@app.get("/analyze")
async def get_analyze_redirect():
    return RedirectResponse(url="/#describer", status_code=303)

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    history = get_all_history()
    if not client:
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Gemini API key is not configured. Please add it to your .env file.",
            "history": history,
            "active_tab": "describer"
         })
         
    if not file.content_type.startswith("image/"):
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Uploaded file is not an image. Please upload a valid image file (PNG, JPG, JPEG, WEBP).",
            "history": history,
            "active_tab": "describer"
         })

    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Load the image using PIL
        image = Image.open(BytesIO(contents))
        
        # Encode image as base64 for persistent storage & display
        image_b64 = base64.b64encode(contents).decode('utf-8')
        image_data_uri = f"data:{file.content_type};base64,{image_b64}"
        
        # Pass image to Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=['Describe this image in detail. Highlight the main subjects, action, colors, background, annotations/markers, and mood.', image]
        )
        
        description = response.text
        timestamp = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        snippet = description[:140] + "..." if len(description) > 140 else description
        
        # Save permanently to SQLite database
        new_id = insert_history_record(
            filename=file.filename,
            timestamp=timestamp,
            description=description,
            image_data=image_data_uri,
            snippet=snippet
        )
        
        # Reload updated persistent history
        updated_history = get_all_history()
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "description": description,
            "filename": file.filename,
            "image_data": image_data_uri,
            "history": updated_history,
            "active_tab": "describer",
            "saved_id": new_id
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred: {str(e)}",
            "history": history,
            "active_tab": "describer"
        })

@app.post("/delete-history/{item_id}")
async def delete_history(item_id: int):
    delete_history_record(item_id)
    return JSONResponse({"success": True})

@app.post("/clear-history")
async def clear_history():
    clear_all_history_records()
    return JSONResponse({"success": True})

@app.get("/export-report/{item_id}")
async def export_report(item_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if not row:
            return PlainTextResponse("Record not found", status_code=404)
        
        item = dict(row)
        report_content = f"""==================================================
VisionIQ — AI Vision Analysis Report
==================================================
File: {item['filename']}
Date Analyzed: {item['timestamp']}
ID: #{item['id']}
==================================================

AI DESCRIPTION & INSIGHTS:
--------------------------------------------------
{item['description']}

==================================================
Generated by VisionIQ with Google Gemini 2.5 Flash
==================================================
"""
        return PlainTextResponse(
            report_content,
            headers={
                "Content-Disposition": f"attachment; filename=visioniq_report_{item['id']}.txt"
            }
        )
