"""
model_utils.py — Cipher Resume Analyzer utilities
"""

import os
import re
import json
import io
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import docx
from rapidfuzz import fuzz

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- Embeddings ----------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder
def semantic_match_score(jd_phrases: List[str], resume_phrases: List[str], topk: int = 1) -> List[Dict[str, Any]]:
    """
    Compute semantic similarity between JD phrases and resume phrases.
    Returns a list of dicts with: jd, matched, sim
    """
    model = get_embedder()
    if not jd_phrases or not resume_phrases:
        return []
    jd_emb = model.encode(jd_phrases, convert_to_tensor=True)
    res_emb = model.encode(resume_phrases, convert_to_tensor=True)
    sims = util.cos_sim(jd_emb, res_emb).cpu().numpy()
    matches = []
    for i, jd in enumerate(jd_phrases):
        row = sims[i]
        best_idx = int(row.argmax())
        best_sim = float(row[best_idx])
        matches.append({
            "jd": jd,
            "matched": resume_phrases[best_idx],
            "sim": best_sim
        })
    return matches


# ---------------- Config ----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-1.5-mini"

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Failed to configure Gemini")

# ---------------- Text Extraction ----------------
def extract_text_from_pdf(path: str) -> str:
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text(path: str, filename: str = None) -> str:
    fname = filename or path
    ext = fname.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ("docx","doc"):
        return extract_text_from_docx(path)
    elif ext in ("txt","md"):
        try:
            with open(path,"r",encoding="utf-8",errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""

# ---------------- JD Parsing ----------------
def parse_jd_skills(jd_text: str) -> List[str]:
    jd_text_lower = (jd_text or "").replace("\r","\n")
    skills = []
    for line in jd_text_lower.split("\n"):
        if any(k in line for k in ["must have","must-have","skills:","required:","responsibilities:","good to have","nice to have"]):
            parts = re.split(r":|-", line, maxsplit=1)
            if len(parts) > 1:
                rest = parts[1]
                for s in re.split(r"[;,]", rest):
                    s = s.strip()
                    if len(s) > 1:
                        skills.append(s)
    if not skills:
        tokens = re.findall(r"\b[A-Za-z+#\.]{2,20}\b", jd_text_lower)
        common = ["python","sql","aws","docker","java","c++","ml","tensorflow","pytorch","react"]
        for t in tokens:
            if t.lower() in common and t not in skills:
                skills.append(t)
    return list(dict.fromkeys([s.strip().strip(".") for s in skills if s.strip()]))

# ---------------- Features & ATS ----------------
def fuzzy_match_score(jd_skill: str, text: str) -> float:
    try:
        return fuzz.token_set_ratio(jd_skill.lower(), text.lower())/100.0
    except Exception:
        return 0.0

def compute_simple_features(jd_skills: List[str], resume_text: str) -> Dict[str,Any]:
    res_lower = (resume_text or "").lower()
    num_skills = max(1, len(jd_skills))
    exact, fuzzy_total, missing = 0, 0.0, []
    for sk in jd_skills:
        if sk.lower() in res_lower:
            exact += 1
        else:
            score = fuzzy_match_score(sk,resume_text)
            fuzzy_total += score
            if score < 0.5:
                missing.append(sk)
    coverage = (exact + fuzzy_total)/num_skills
    return {"skills_coverage":coverage, "years_experience":0.0, "missing_skills":missing}

def compute_ats_score(resume_text: str) -> Dict[str,Any]:
    score, reasons = 0, []
    if re.search(r"[\w\.-]+@[\w\.-]+", resume_text): score += 25
    else: reasons.append("Missing email")
    if re.search(r"\+?\d[\d\s\-]{6,}\d", resume_text): score += 25
    else: reasons.append("Missing phone")
    if re.search(r"\bskills\b", resume_text.lower()): score += 25
    else: reasons.append("No skills section")
    if len(resume_text) > 200: score += 25
    else: reasons.append("Too short")
    return {"ats_score":score, "reasons":reasons}

def aggregate_components(features: Dict[str,Any], jd_text: str) -> Dict[str,float]:
    return {
        "skills": features.get("skills_coverage",0)*100,
        "experience": features.get("years_experience",0)*50,
        "projects": 75,
        "education": 80
    }

def ideal_profile_from_jd(jd_text: str) -> Dict[str,float]:
    return {"skills":100,"experience":100,"projects":100,"education":100}

# ---------------- Suggestions ----------------
def generate_ai_suggestions_rulebased(missing_skills: List[str], resume_text: str, jd_text: str) -> Dict[str,Any]:
    seeds = missing_skills or parse_jd_skills(jd_text)[:5]
    headline = f"Enhance your resume with {', '.join(seeds[:3])}"
    return {
        "headline": headline,
        "summary": "Add missing skills and highlight measurable results.",
        "improvements":[{"title":"Add keywords","why":"Missing skills","example":[f"Add {k} in Skills section"]} for k in seeds],
        "edits":["Refactor headline","Use bullet points"],
        "interview_questions":[{"question":f"Tell me about a project where you used {k}"} for k in seeds],
        "resources":[{"title":s,"url":"https://google.com/search?q="+s} for s in seeds],
        "overall_notes":"These are rule-based improvements."
    }

def _call_gemini(prompt: str) -> str:
    if not (GEMINI_API_KEY and genai):
        return None
    try:
        response = genai.generate_text(model=GEMINI_MODEL, prompt=prompt, temperature=0.2, max_output_tokens=700)
        return getattr(response,"text",str(response))
    except Exception as e:
        logger.exception("Gemini call failed: %s",e)
        return None

def generate_ai_suggestions_llm(jd_text: str, resume_text: str) -> Dict[str,Any]:
    schema = '{"headline":"...","summary":"...","improvements":[{"title":"...","why":"...","example":["..."]}],"edits":["..."],"interview_questions":[{"question":"..."}],"resources":[{"title":"...","url":"..."}],"overall_notes":"..."}'
    prompt = f"""You are a career coach. Return ONLY valid JSON in this schema:
{schema}

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    out = _call_gemini(prompt)
    if not out:
        return generate_ai_suggestions_rulebased([], resume_text, jd_text)
    txt = out.strip()
    jstart, jend = txt.find("{"), txt.rfind("}")
    if jstart!=-1 and jend!=-1:
        try:
            return json.loads(txt[jstart:jend+1])
        except Exception:
            return generate_ai_suggestions_rulebased([], resume_text, jd_text)
    return generate_ai_suggestions_rulebased([], resume_text, jd_text)

def generate_ai_suggestions(missing_skills,resume_text,jd_text):
    if GEMINI_API_KEY and genai:
        return generate_ai_suggestions_llm(jd_text,resume_text)
    return generate_ai_suggestions_rulebased(missing_skills or [],resume_text,jd_text)

# ---------------- PDF ----------------
def generate_pdf_suggestions_bytes(candidate_name: str, suggestions: Dict[str, Any], metrics: Dict[str, Any]) -> bytes:
    try:
        if canvas is None:
            raise RuntimeError("reportlab not available")
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.setFont("Helvetica-Bold",16)
        c.drawString(50,750,f"Cipher - Suggestions for {candidate_name}")
        c.setFont("Helvetica",12)
        y=720
        c.drawString(50,y,"Metrics:")
        y-=20
        for k,v in (metrics or {}).items():
            c.drawString(60,y,f"{k}: {v}")
            y-=14
        y-=10
        c.drawString(50,y,"AI Suggestions:")
        y-=20
        c.drawString(60,y,"Headline: "+suggestions.get("headline",""))
        y-=16
        c.drawString(60,y,"Summary: "+suggestions.get("summary","")[:100])
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()
    except Exception:
        return str(suggestions).encode()
