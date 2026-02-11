from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import google.generativeai as genai
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import io
import json
import re
from typing import Optional, List, Dict, Tuple
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import firebase_admin
from firebase_admin import credentials, auth
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import zipfile
import tempfile
import pandas as pd

load_dotenv()

app = FastAPI(title="Enhanced ATS Resume Analyzer", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
# Make bearer auth optional so the API still works when Firebase isn't configured
security = HTTPBearer(auto_error=False)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
REPORT_DIR = Path("reports")
BATCH_DIR = Path("batch_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
BATCH_DIR.mkdir(exist_ok=True)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URL)
# Use your specified database name: Resume_Analyser
db = mongo_client.Resume_Analyser
users_collection = db.users
resumes_collection = db.resumes
analysis_collection = db.analysis
interviews_collection = db.interviews
rankings_collection = db.rankings

# Firebase Configuration
try:
    firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
    if os.path.exists(firebase_cred_path):
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred)
        FIREBASE_ENABLED = True
    else:
        FIREBASE_ENABLED = False
        print("Warning: Firebase credentials not found. Authentication disabled.")
except Exception as e:
    FIREBASE_ENABLED = False
    print(f"Warning: Firebase initialization failed: {e}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== AUTHENTICATION ====================
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify Firebase token (optional when Firebase is disabled)"""
    # If Firebase auth is disabled or no credentials are provided, use a demo user
    if not FIREBASE_ENABLED or credentials is None:
        return {"uid": "demo_user", "email": "demo@example.com"}

    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication: {str(e)}")


# ==================== HELPER FUNCTIONS ====================
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-.,@#$%&()]', '', text)
    return text.strip()


def extract_bullets(text: str) -> List[str]:
    """Extract bullet points from resume"""
    bullets = []
    patterns = [
        r'[•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]\s*(.+?)(?=[•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]|$)',
        r'[-–—]\s*(.+?)(?=[-–—]|\n|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        bullets.extend([match.strip() for match in matches if match.strip()])
    
    return bullets[:20]


def detect_weak_phrases(text: str) -> List[Dict]:
    """Detect weak phrases in resume"""
    weak_phrases_list = [
        "responsible for", "worked on", "helped with", "assisted with",
        "participated in", "various tasks", "duties included", "contributed to",
        "familiar with", "knowledge of", "exposure to"
    ]
    
    detected = []
    text_lower = text.lower()
    
    for phrase in weak_phrases_list:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            start, end = match.span()
            snippet_start = max(0, start - 30)
            snippet_end = min(len(text), end + 30)
            detected.append({
                "phrase": phrase,
                "start": start,
                "end": end,
                "snippet": text[snippet_start:snippet_end]
            })
    
    return detected


def compute_ats_score(resume_text: str, jd_text: str = "") -> Dict:
    """Compute comprehensive ATS scores"""
    resume_lower = resume_text.lower()
    
    # Section detection
    sections = {
        "Summary": bool(re.search(r'\b(summary|objective|profile)\b', resume_lower)),
        "Experience": bool(re.search(r'\b(experience|work history|employment)\b', resume_lower)),
        "Education": bool(re.search(r'\b(education|academic|qualification)\b', resume_lower)),
        "Skills": bool(re.search(r'\b(skills|technical skills|competencies)\b', resume_lower)),
        "Projects": bool(re.search(r'\b(projects|portfolio)\b', resume_lower)),
        "Certifications": bool(re.search(r'\b(certifications|certificates|licenses)\b', resume_lower)),
    }
    
    section_score = (sum(sections.values()) / len(sections)) * 100
    
    # Keyword matching with JD
    keyword_score = 0
    matched_keywords = []
    missing_keywords = []
    
    if jd_text:
        jd_words = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
        resume_words = set(re.findall(r'\b\w{3,}\b', resume_lower))
        
        # Filter out common words
        common_words = {'the', 'and', 'with', 'for', 'from', 'that', 'this', 'will', 'are', 'was', 'were'}
        jd_words = jd_words - common_words
        resume_words = resume_words - common_words
        
        matched = jd_words.intersection(resume_words)
        missing = jd_words - resume_words
        
        matched_keywords = list(matched)[:20]
        missing_keywords = list(missing)[:20]
        
        keyword_score = (len(matched) / len(jd_words)) * 100 if jd_words else 0
    
    # Extract bullets for scoring
    bullets = extract_bullets(resume_text)
    bullets_count = len(bullets)
    
    # Action verb score
    action_verbs = [
        'led', 'managed', 'developed', 'created', 'improved', 'increased', 'decreased',
        'achieved', 'delivered', 'designed', 'implemented', 'launched', 'built',
        'analyzed', 'optimized', 'reduced', 'generated', 'established', 'spearheaded',
        'orchestrated', 'streamlined', 'pioneered', 'transformed'
    ]
    action_count = sum(1 for bullet in bullets if any(verb in bullet.lower() for verb in action_verbs))
    action_score = (action_count / bullets_count * 100) if bullets_count > 0 else 0
    
    # Metric score (numbers, percentages, etc.)
    metric_pattern = r'\d+%|\$\d+|\d+\+|increased by \d+|reduced by \d+|\d+x'
    metric_count = sum(1 for bullet in bullets if re.search(metric_pattern, bullet, re.IGNORECASE))
    metric_score = (metric_count / bullets_count * 100) if bullets_count > 0 else 0
    
    # Word count and length score
    word_count = len(resume_text.split())
    length_score = 100 if 400 <= word_count <= 800 else max(0, 100 - abs(600 - word_count) / 10)
    
    # Calculate final score
    weights = {
        'section': 0.25,
        'keyword': 0.30 if jd_text else 0,
        'action': 0.20,
        'metric': 0.15,
        'length': 0.10
    }
    
    # Normalize weights if no JD provided
    if not jd_text:
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    
    final_score = (
        section_score * weights['section'] +
        keyword_score * weights['keyword'] +
        action_score * weights['action'] +
        metric_score * weights['metric'] +
        length_score * weights['length']
    )
    
    return {
        "final_score": round(final_score, 2),
        "section_score": round(section_score, 2),
        "keyword_score": round(keyword_score, 2),
        "action_score": round(action_score, 2),
        "metric_score": round(metric_score, 2),
        "length_score": round(length_score, 2),
        "word_count": word_count,
        "bullets_count": bullets_count,
        "sections_found": sections,
        "bullets": bullets,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords
    }


def generate_suggestions(ats_scores: Dict, weak_phrases: List[Dict], has_jd: bool) -> List[str]:
    """Generate improvement suggestions"""
    suggestions = []
    
    if ats_scores['section_score'] < 70:
        missing = [k for k, v in ats_scores['sections_found'].items() if not v]
        if missing:
            suggestions.append(f"❌ Missing important sections: {', '.join(missing)}")
    
    if has_jd and ats_scores['keyword_score'] < 60:
        suggestions.append("⚠️ Low keyword match with job description. Tailor your resume more closely to the JD.")
    
    if ats_scores['action_score'] < 50:
        suggestions.append("💪 Use more strong action verbs to start your bullet points (e.g., Led, Developed, Improved).")
    
    if ats_scores['metric_score'] < 40:
        suggestions.append("📊 Add more measurable achievements with numbers, percentages, or dollar amounts.")
    
    if ats_scores['word_count'] < 400:
        suggestions.append("📝 Your resume is too short. Add more details about your experience and achievements.")
    elif ats_scores['word_count'] > 800:
        suggestions.append("✂️ Your resume is too long. Focus on the most relevant and impactful information.")
    
    if weak_phrases:
        unique_phrases = list(set([wp['phrase'] for wp in weak_phrases]))
        suggestions.append(f"⚡ Replace weak phrases: {', '.join(unique_phrases[:5])}")
    
    if ats_scores['final_score'] >= 80:
        suggestions.append("✅ Excellent! Your resume is well-optimized for ATS systems.")
    elif ats_scores['final_score'] >= 60:
        suggestions.append("👍 Good start! A few improvements will make your resume even stronger.")
    else:
        suggestions.append("🔧 Significant improvements needed. Focus on sections, keywords, and action verbs.")
    
    return suggestions


async def analyze_with_gemini(resume_text: str, jd_text: str, prompt_type: str) -> str:
    """Analyze resume using Gemini AI"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    prompts = {
        "match": f"""
You are an experienced ATS scanner with deep understanding of tech, software engineering, data science, and big data fields.

Resume:
{resume_text}

Job Description:
{jd_text}

Provide a JSON response with:
{{
    "match_percentage": "X%",
    "missing_keywords": ["keyword1", "keyword2"],
    "profile_summary": "brief summary",
    "recommendations": ["recommendation1", "recommendation2"]
}}
""",
        
        "review": f"""
As a senior resume reviewer, analyze this resume:

Resume:
{resume_text}

Provide JSON response:
{{
    "rating": 8,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "ats_friendly": true
}}
""",
        
        "skills": f"""
Analyze this resume for skill improvements:

Resume:
{resume_text}

{f"Job Description: {jd_text}" if jd_text else ""}

Provide JSON response:
{{
    "current_skills": ["skill1", "skill2"],
    "skill_gaps": ["gap1", "gap2"],
    "trending_skills": ["trend1", "trend2"],
    "learning_resources": ["resource1", "resource2"],
    "presentation_tips": ["tip1", "tip2"]
}}
"""
    }
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompts[prompt_type])
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


async def generate_interview_questions(resume_text: str, difficulty: str = "mixed") -> List[Dict]:
    """Generate interview questions based on resume"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    prompt = f"""
Based on this resume, generate interview questions:

Resume:
{resume_text}

Generate 10 questions with the following distribution:
- 3 Easy questions (basic understanding)
- 4 Medium questions (application and analysis)
- 3 Hard questions (synthesis and evaluation)

Return as JSON array:
[
    {{
        "question": "question text",
        "difficulty": "easy|medium|hard",
        "category": "technical|behavioral|situational",
        "expected_keywords": ["keyword1", "keyword2"],
        "max_score": 10
    }}
]
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if json_match:
            questions = json.loads(json_match.group())
            return questions
        else:
            # Fallback to default questions if parsing fails
            return [
                {
                    "question": "Tell me about your most significant project.",
                    "difficulty": "easy",
                    "category": "behavioral",
                    "expected_keywords": ["project", "responsibility", "outcome"],
                    "max_score": 10
                }
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation error: {str(e)}")


async def evaluate_interview_answer(question: str, answer: str, expected_keywords: List[str]) -> Dict:
    """Evaluate interview answer using AI"""
    if not GOOGLE_API_KEY:
        return {"score": 0, "feedback": "AI evaluation unavailable"}
    
    prompt = f"""
Evaluate this interview answer:

Question: {question}
Answer: {answer}
Expected Keywords: {', '.join(expected_keywords)}

Provide JSON response:
{{
    "score": 0-10,
    "feedback": "detailed feedback",
    "strengths": ["strength1"],
    "improvements": ["improvement1"]
}}
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"score": 5, "feedback": "Evaluation unavailable"}
    except:
        return {"score": 5, "feedback": "Evaluation error"}


def generate_pdf_report(analysis_data: Dict, filename: str) -> str:
    """Generate PDF report for resume analysis"""
    filepath = REPORT_DIR / filename
    
    doc = SimpleDocTemplate(str(filepath), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("ATS Resume Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Metadata
    meta_data = [
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Overall ATS Score:", f"{analysis_data['ats_scores']['final_score']}%"],
        ["Status:", "Pass" if analysis_data['ats_scores']['final_score'] >= 60 else "Needs Improvement"]
    ]
    
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Score Breakdown
    story.append(Paragraph("Score Breakdown", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    scores = analysis_data['ats_scores']
    score_data = [
        ["Metric", "Score"],
        ["Sections", f"{scores['section_score']}%"],
        ["Keywords", f"{scores['keyword_score']}%"],
        ["Action Verbs", f"{scores['action_score']}%"],
        ["Metrics", f"{scores['metric_score']}%"],
        ["Length", f"{scores['length_score']}%"]
    ]
    
    score_table = Table(score_data, colWidths=[3*inch, 3*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(score_table)
    story.append(PageBreak())
    
    # Suggestions
    story.append(Paragraph("Improvement Suggestions", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    for suggestion in analysis_data['suggestions']:
        story.append(Paragraph(f"• {suggestion}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    return str(filepath)


# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_path = Path("static/index.html")
    if html_path.exists():
        with open(html_path, "r",encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>ATS System - Frontend not found</h1>")


@app.post("/api/analyze")
async def analyze_resume(
    resume_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(""),
    analysis_type: str = Form("basic"),
    user_data: dict = Depends(verify_token)
):
    """Analyze single resume"""
    try:
        # Extract resume text
        if resume_file:
            file_content = await resume_file.read()
            resume_text = extract_text_from_pdf(file_content)
            filename = resume_file.filename
        elif not resume_text:
            raise HTTPException(status_code=400, detail="No resume provided")
        else:
            filename = "pasted_text.txt"
        
        # Clean text
        resume_text = clean_text(resume_text)
        jd_text = clean_text(jd_text) if jd_text else ""
        
        # Basic analysis
        ats_scores = compute_ats_score(resume_text, jd_text)
        weak_phrases = detect_weak_phrases(resume_text)
        suggestions = generate_suggestions(ats_scores, weak_phrases, bool(jd_text))
        
        result = {
            "ats_scores": ats_scores,
            "weak_phrases": weak_phrases,
            "suggestions": suggestions,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # AI-powered analysis
        if analysis_type in ["match", "review", "skills"] and GOOGLE_API_KEY:
            ai_response = await analyze_with_gemini(resume_text, jd_text, analysis_type)
            
            try:
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    result["ai_analysis"] = ai_data
                else:
                    result["ai_analysis"] = {"raw_response": ai_response}
            except:
                result["ai_analysis"] = {"raw_response": ai_response}
        
        # Save to MongoDB
        analysis_doc = {
            "user_id": user_data.get("uid"),
            "filename": filename,
            "resume_text": resume_text[:1000],  # Store first 1000 chars
            "jd_text": jd_text[:500] if jd_text else "",
            "ats_scores": ats_scores,
            "analysis_type": analysis_type,
            "timestamp": datetime.now(),
            "result": result
        }
        
        await analysis_collection.insert_one(analysis_doc)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-analyze")
async def batch_analyze_resumes(
    files: List[UploadFile] = File(...),
    jd_text: Optional[str] = Form(""),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user_data: dict = Depends(verify_token)
):
    """Analyze multiple resumes and rank them"""
    try:
        jd_text = clean_text(jd_text) if jd_text else ""
        
        results = []
        
        for file in files:
            file_content = await file.read()
            resume_text = extract_text_from_pdf(file_content)
            resume_text = clean_text(resume_text)
            
            # Compute scores
            ats_scores = compute_ats_score(resume_text, jd_text)
            weak_phrases = detect_weak_phrases(resume_text)
            suggestions = generate_suggestions(ats_scores, weak_phrases, bool(jd_text))
            
            result = {
                "filename": file.filename,
                "ats_scores": ats_scores,
                "weak_phrases": weak_phrases,
                "suggestions": suggestions,
                "jd_match": ats_scores['keyword_score'] if jd_text else 0,
                "eligible_for_interview": ats_scores['keyword_score'] >= 60 if jd_text else False
            }
            
            results.append(result)
        
        # Rank by JD match score
        results.sort(key=lambda x: x['jd_match'], reverse=True)
        
        # Add rankings
        for idx, result in enumerate(results, 1):
            result['rank'] = idx
        
        # Save batch analysis
        batch_doc = {
            "user_id": user_data.get("uid"),
            "jd_text": jd_text[:500],
            "total_resumes": len(files),
            "results": results,
            "timestamp": datetime.now()
        }
        
        await rankings_collection.insert_one(batch_doc)
        
        return JSONResponse(content={
            "total_analyzed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interview/generate-questions")
async def generate_questions(
    resume_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    user_data: dict = Depends(verify_token)
):
    """Generate interview questions from resume"""
    try:
        if resume_file:
            file_content = await resume_file.read()
            resume_text = extract_text_from_pdf(file_content)
        elif not resume_text:
            raise HTTPException(status_code=400, detail="No resume provided")
        
        resume_text = clean_text(resume_text)
        
        questions = await generate_interview_questions(resume_text)
        
        # Save interview session
        interview_doc = {
            "user_id": user_data.get("uid"),
            "resume_text": resume_text[:1000],
            "questions": questions,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        inserted = await interviews_collection.insert_one(interview_doc)
        
        return JSONResponse(content={
            "interview_id": str(inserted.inserted_id),
            "questions": questions
        })
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interview/submit-answers")
async def submit_interview_answers(
    interview_id: str = Form(...),
    answers: str = Form(...),  # JSON string of answers
    user_data: dict = Depends(verify_token)
):
    """Submit and evaluate interview answers"""
    try:
        answers_data = json.loads(answers)
        
        # Get interview
        interview = await interviews_collection.find_one({"_id": ObjectId(interview_id)})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Evaluate each answer
        total_score = 0
        max_possible = 0
        evaluated_answers = []
        
        for idx, answer_data in enumerate(answers_data):
            question = interview['questions'][idx]
            evaluation = await evaluate_interview_answer(
                question['question'],
                answer_data['answer'],
                question['expected_keywords']
            )
            
            total_score += evaluation['score']
            max_possible += question['max_score']
            
            evaluated_answers.append({
                "question": question['question'],
                "answer": answer_data['answer'],
                "evaluation": evaluation
            })
        
        interview_score = (total_score / max_possible * 100) if max_possible > 0 else 0
        
        # Update interview
        await interviews_collection.update_one(
            {"_id": ObjectId(interview_id)},
            {
                "$set": {
                    "answers": evaluated_answers,
                    "score": interview_score,
                    "status": "completed",
                    "completed_at": datetime.now()
                }
            }
        )
        
        return JSONResponse(content={
            "interview_score": round(interview_score, 2),
            "total_score": total_score,
            "max_possible": max_possible,
            "evaluated_answers": evaluated_answers
        })
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rankings")
async def get_rankings(
    jd_text: str,
    user_data: dict = Depends(verify_token)
):
    """Get rankings based on JD match and interview performance"""
    try:
        # Get all resumes with their scores
        analyses = await analysis_collection.find({
            "user_id": user_data.get("uid"),
            "jd_text": {"$regex": jd_text[:50]}
        }).to_list(length=100)
        
        rankings = []
        
        for analysis in analyses:
            filename = analysis.get('filename', 'Unknown')
            ats_score = analysis['ats_scores']['final_score']
            jd_match = analysis['ats_scores']['keyword_score']
            
            # Get interview score if available
            interview = await interviews_collection.find_one({
                "user_id": user_data.get("uid"),
                "resume_text": {"$regex": analysis['resume_text'][:100]}
            })
            
            interview_score = interview.get('score', 0) if interview else 0
            
            # Combined score: 60% JD match + 40% interview performance
            combined_score = (jd_match * 0.6) + (interview_score * 0.4)
            
            rankings.append({
                "filename": filename,
                "ats_score": ats_score,
                "jd_match": jd_match,
                "interview_score": interview_score,
                "combined_score": round(combined_score, 2),
                "eligible": jd_match >= 60
            })
        
        # Sort by combined score
        rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add ranks
        for idx, item in enumerate(rankings, 1):
            item['rank'] = idx
        
        return JSONResponse(content={"rankings": rankings})
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-report/{analysis_id}")
async def download_report(
    analysis_id: str,
    user_data: dict = Depends(verify_token)
):
    """Download PDF report for an analysis"""
    try:
        analysis = await analysis_collection.find_one({"_id": ObjectId(analysis_id)})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        filename = f"report_{analysis_id}.pdf"
        filepath = generate_pdf_report(analysis['result'], filename)
        
        return FileResponse(
            filepath,
            media_type='application/pdf',
            filename=filename
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-batch-reports")
async def download_batch_reports(
    batch_id: str,
    user_data: dict = Depends(verify_token)
):
    """Download all reports from a batch analysis as ZIP"""
    try:
        batch = await rankings_collection.find_one({"_id": ObjectId(batch_id)})
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / f"batch_reports_{batch_id}.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for idx, result in enumerate(batch['results'], 1):
                    pdf_filename = f"report_{idx}_{result['filename']}.pdf"
                    pdf_path = generate_pdf_report(result, pdf_filename)
                    zipf.write(pdf_path, pdf_filename)
            
            return FileResponse(
                zip_path,
                media_type='application/zip',
                filename=f"batch_reports_{batch_id}.zip"
            )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export-rankings-csv")
async def export_rankings_csv(
    batch_id: str,
    user_data: dict = Depends(verify_token)
):
    """Export rankings to CSV"""
    try:
        batch = await rankings_collection.find_one({"_id": ObjectId(batch_id)})
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        df = pd.DataFrame(batch['results'])
        csv_path = REPORT_DIR / f"rankings_{batch_id}.csv"
        df.to_csv(csv_path, index=False)
        
        return FileResponse(
            csv_path,
            media_type='text/csv',
            filename=f"rankings_{batch_id}.csv"
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/history")
async def get_user_history(
    user_data: dict = Depends(verify_token)
):
    """Get user's analysis history"""
    try:
        analyses = await analysis_collection.find({
            "user_id": user_data.get("uid")
        }).sort("timestamp", -1).limit(50).to_list(length=50)
        
        history = []
        for analysis in analyses:
            history.append({
                "id": str(analysis['_id']),
                "filename": analysis.get('filename'),
                "score": analysis['ats_scores']['final_score'],
                "timestamp": analysis['timestamp'].isoformat()
            })
        
        return JSONResponse(content={"history": history})
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_configured": bool(GOOGLE_API_KEY),
        "firebase_enabled": FIREBASE_ENABLED,
        "mongodb_connected": mongo_client is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)