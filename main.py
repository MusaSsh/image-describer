import os
import sqlite3
import base64
import hashlib
import secrets
from datetime import datetime
from fastapi import FastAPI, Request, File, UploadFile, Form, Response
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
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/history.db"
else:
    os.makedirs("data", exist_ok=True)
    DB_PATH = "data/history.db"

# Password Hashing Utilities
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    return f"{salt}:{pwd_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, pwd_hash = stored_hash.split(":")
        check_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
        return secrets.compare_digest(pwd_hash, check_hash)
    except Exception:
        return False

# Database Initialization
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                description TEXT NOT NULL,
                image_data TEXT NOT NULL,
                snippet TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

init_db()

# Session Helper
def get_current_user(request: Request):
    user_id = request.cookies.get("visioniq_user_id")
    if not user_id:
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, email, full_name, created_at FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
    except Exception:
        pass
    return None

def get_user_history(user_id: int = None, search_query: str = ""):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if user_id:
            if search_query:
                cursor.execute("""
                    SELECT * FROM history 
                    WHERE user_id = ? AND (filename LIKE ? OR description LIKE ?)
                    ORDER BY id DESC
                """, (user_id, f"%{search_query}%", f"%{search_query}%"))
            else:
                cursor.execute("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,))
        else:
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

def insert_history_record(user_id: int, filename: str, timestamp: str, description: str, image_data: str, snippet: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (user_id, filename, timestamp, description, image_data, snippet)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, filename, timestamp, description, image_data, snippet))
        conn.commit()
        return cursor.lastrowid

def delete_history_record(item_id: int, user_id: int = None):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (item_id, user_id))
        else:
            cursor.execute("DELETE FROM history WHERE id = ?", (item_id,))
        conn.commit()

def clear_all_history_records(user_id: int = None):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("DELETE FROM history")
        conn.commit()

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
    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None
    history = get_user_history(user_id, q)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": history,
        "active_tab": tab,
        "search_query": q,
        "current_user": current_user
    })

@app.get("/analyze")
async def get_analyze_redirect():
    return RedirectResponse(url="/#describer", status_code=303)

@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None
    history = get_user_history(user_id)

    if not client:
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Gemini API key is not configured. Please add it to your .env file.",
            "history": history,
            "active_tab": "describer",
            "current_user": current_user
         })
         
    if not file.content_type.startswith("image/"):
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Uploaded file is not an image. Please upload a valid image file (PNG, JPG, JPEG, WEBP).",
            "history": history,
            "active_tab": "describer",
            "current_user": current_user
         })

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
            user_id=user_id,
            filename=file.filename,
            timestamp=timestamp,
            description=description,
            image_data=image_data_uri,
            snippet=snippet
        )
        
        updated_history = get_user_history(user_id)
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "description": description,
            "filename": file.filename,
            "image_data": image_data_uri,
            "history": updated_history,
            "active_tab": "describer",
            "saved_id": new_id,
            "current_user": current_user
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred: {str(e)}",
            "history": history,
            "active_tab": "describer",
            "current_user": current_user
        })

# Authentication Endpoints
@app.post("/signup")
async def signup(request: Request, full_name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    email_clean = email.strip().lower()
    full_name_clean = full_name.strip()
    
    if not email_clean or not password or not full_name_clean:
        history = get_user_history()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "auth_error": "All fields are required for sign up.",
            "history": history,
            "active_tab": "home"
        })
    
    pwd_hash = hash_password(password)
    now_str = datetime.now().strftime("%b %d, %Y")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (email, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (email_clean, full_name_clean, pwd_hash, now_str))
            conn.commit()
            user_id = cursor.lastrowid
            
        response = RedirectResponse(url="/#describer", status_code=303)
        response.set_cookie(key="visioniq_user_id", value=str(user_id), max_age=86400*30, httponly=True)
        return response
    except sqlite3.IntegrityError:
        history = get_user_history()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "auth_error": "An account with this email address already exists. Please Sign In.",
            "history": history,
            "active_tab": "home"
        })

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email_clean = email.strip().lower()
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
        row = cursor.fetchone()
        
        if row and verify_password(password, row["password_hash"]):
            response = RedirectResponse(url="/#describer", status_code=303)
            response.set_cookie(key="visioniq_user_id", value=str(row["id"]), max_age=86400*30, httponly=True)
            return response

    history = get_user_history()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "auth_error": "Invalid email or password. Please try again.",
        "history": history,
        "active_tab": "home"
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="visioniq_user_id")
    return response

@app.post("/delete-history/{item_id}")
async def delete_history(request: Request, item_id: int):
    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None
    delete_history_record(item_id, user_id)
    return JSONResponse({"success": True})

@app.post("/clear-history")
async def clear_history(request: Request):
    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None
    clear_all_history_records(user_id)
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
