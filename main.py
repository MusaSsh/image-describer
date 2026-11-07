import os
import sqlite3
import base64
from datetime import datetime
from fastapi import FastAPI, Request, File, UploadFile, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from pydantic_settings import BaseSettings
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Settings(BaseSettings):
    gemini_api_key: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

app = FastAPI(title="VisionIQ — Image Describer & Studio")

# Ensure data directory exists for persistent SQLite database
DB_PATH = "/tmp/history.db"
try:
    if not os.environ.get("VERCEL"):
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        DB_PATH = os.path.join(data_dir, "history.db")
except Exception:
    DB_PATH = "/tmp/history.db"

# Database Initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        # History Table (no user_id dependency)
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
    finally:
        conn.close()

init_db()

def get_user_history(search_query: str = ""):
    conn = sqlite3.connect(DB_PATH)
    try:
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
    finally:
        conn.close()

def insert_history_record(filename: str, timestamp: str, description: str, image_data: str, snippet: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (filename, timestamp, description, image_data, snippet)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, timestamp, description, image_data, snippet))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def delete_history_record(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ?", (item_id,))
        conn.commit()
    finally:
        conn.close()

def clear_all_history_records():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
    finally:
        conn.close()

# Mount static files
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Gemini Client
if settings.gemini_api_key:
    client = genai.Client(api_key=settings.gemini_api_key)
else:
    client = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, tab: str = "home", q: str = ""):
    history = get_user_history(q)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "history": history,
            "active_tab": tab,
            "search_query": q
        }
    )

@app.get("/analyze")
async def get_analyze_redirect():
    return RedirectResponse(url="/#describer", status_code=303)

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    history = get_user_history()

    if not client:
         return templates.TemplateResponse(
            request=request, 
            name="index.html",
            context={
                "error": "Gemini API key is not configured. Please add it to your .env file.",
                "history": history,
                "active_tab": "describer"
            }
         )
         
    if not file.content_type.startswith("image/"):
         return templates.TemplateResponse(
            request=request, 
            name="index.html",
            context={
                "error": "Uploaded file is not an image. Please upload a valid image file (PNG, JPG, JPEG, WEBP).",
                "history": history,
                "active_tab": "describer"
            }
         )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        image_b64 = base64.b64encode(contents).decode('utf-8')
        image_data_uri = f"data:{file.content_type};base64,{image_b64}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=['Describe this image in detail. Highlight the main subjects, action, colors, background, annotations/markers, and mood.', image]
        )
        
        description = response.text
        timestamp = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        snippet = description[:140] + "..." if len(description) > 140 else description
        
        new_id = insert_history_record(
            filename=file.filename,
            timestamp=timestamp,
            description=description,
            image_data=image_data_uri,
            snippet=snippet
        )
        
        updated_history = get_user_history()
        
        return templates.TemplateResponse(
            request=request, 
            name="index.html",
            context={
                "description": description,
                "filename": file.filename,
                "image_data": image_data_uri,
                "history": updated_history,
                "active_tab": "describer",
                "saved_id": new_id
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            request=request, 
            name="index.html",
            context={
                "error": f"An error occurred: {str(e)}",
                "history": history,
                "active_tab": "describer"
            }
        )

@app.post("/delete-history/{item_id}")
async def delete_history(request: Request, item_id: int):
    delete_history_record(item_id)
    return JSONResponse({"success": True})

@app.post("/clear-history")
async def clear_history(request: Request):
    clear_all_history_records()
    return JSONResponse({"success": True})

@app.get("/export-report/{item_id}")
async def export_report(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if not row:
            return PlainTextResponse("Record not found", status_code=404)
        
        item = dict(row)
    finally:
        conn.close()

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
