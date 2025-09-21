# app.py — Cipher multi-resume analyzer (keeps grey background + snowflakes UI)
import os
import time
import tempfile
import traceback
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# embeddings similarity
try:
    from sentence_transformers import util
except Exception:
    util = None

# model utilities (expected to exist; we fall back gently if not)
try:
    import model_utils as mu
    MU_AVAILABLE = True
except Exception:
    mu = None
    MU_AVAILABLE = False

st.set_page_config(page_title="Cipher — Resume Analyzer (multi)", layout="wide", initial_sidebar_state="auto")

# Visual polish: full-page abstract background + overlay + global snow (replace earlier blocks)
import streamlit.components.v1 as components

FAVICON_SVG = """
data:image/svg+xml;utf8,
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
  <defs>
    <linearGradient id='g' x1='0' x2='1'>
      <stop offset='0' stop-color='%2300b4ff'/>
      <stop offset='1' stop-color='%23703bff'/>
    </linearGradient>
  </defs>
  <rect width='100' height='100' rx='18' fill='url(%23g)'/>
  <text x='50' y='57' font-family='Segoe UI, Roboto, Arial' font-size='48' fill='white' text-anchor='middle' font-weight='700'>C</text>
</svg>
""".strip().replace("\n", "").replace(" ", "%20")

# Choose an abstract background (dark/tech). Replace URL if you want another.
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1600&q=80"

components.html(
    f"""
<link rel="icon" type="image/svg+xml" href="{FAVICON_SVG}">
<style>
  /* 1) Page-level background (full viewport). Use fixed so it doesn't scroll away. */
  html, body {{
    height: 100%;
    background: transparent !important;
  }}
  body::before {{
    content: "";
    position: fixed;
    inset: 0;
    background:
      linear-gradient(180deg, rgba(4,10,20,0.6), rgba(6,12,28,0.6)),
      url('{BACKGROUND_IMAGE}') center/cover no-repeat;
    background-size: cover;
    filter: saturate(1.05) contrast(1.02);
    opacity: 0.98;
    z-index: -100; /* keep very far behind UI */
    pointer-events: none;
  }}

  /* 2) animated subtle overlay to add motion */
  .cipher-gradient-overlay {{
    position: fixed;
    inset: 0;
    background: linear-gradient(270deg, rgba(11,132,255,0.02), rgba(124,58,237,0.02));
    mix-blend-mode: overlay;
    animation: cipher-slide 24s linear infinite;
    z-index: -90;
    pointer-events: none;
  }}
  @keyframes cipher-slide {{
    0% {{ transform: translateX(-8%); }}
    50% {{ transform: translateX(8%); }}
    100% {{ transform: translateX(-8%); }}
  }}

  /* 3) Snow canvas (behind content but above the background layers) */
  #snow-canvas {{
    position: fixed;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -50;
  }}

  /* 4) Make Streamlit content backgrounds transparent so the page background shows through */
  /* Cover known Streamlit wrappers and common generated classes */
  .stApp, .css-18e3th9, .reportview-container, .main, .block-container, .stSidebar, .sidebar-content {{
    background: transparent !important;
    box-shadow: none !important; /* remove inner shadows that block background */
  }}

  /* Many Streamlit elements use inner cards with background; target card-like classes */
  .stButton>button, .stDownloadButton>button, .stTextInput>div, .stFileUploader, .stMarkdown, .stMetric {{
    background: rgba(255,255,255,0.02) !important;
    border: 0 !important;
  }}

  /* If any panels still have a solid background, force them translucent to reveal background */
  .css-1d391kg, .css-1outpf7, .css-1l90r2p, .css-1adrfps {{
    background: transparent !important;
  }}

  /* Improve readability: add subtle translucent card look for content blocks */
  .stContainer, .block-container > div > div {{
    background: rgba(6,10,14,0.46) !important;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 8px 30px rgba(2,6,23,0.45);
  }}

  /* Ensure links and text contrast nicely on darker background */
  .markdown-text-container, .stMarkdown, .stText, .stExpanderHeader {{
    color: #e6eef5 !important;
  }}

  /* Buttons: slightly elevated for visual pop */
  .stButton>button, .stDownloadButton>button {{
    box-shadow: 0 6px 18px rgba(2,6,23,0.45);
  }}

</style>

<!-- gradient overlay element -->
<div class="cipher-gradient-overlay"></div>

<!-- Snow canvas -->
<canvas id="snow-canvas"></canvas>

<script>
(function() {{
  const canvas = document.getElementById('snow-canvas');
  const ctx = canvas.getContext('2d');
  let W = canvas.width = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  window.addEventListener('resize', () => {{
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    initFlakes();
  }});

  // Create snowflakes
  let flakes = [];
  function rand(min, max) {{ return Math.random() * (max - min) + min; }}

  function initFlakes() {{
    flakes = [];
    const count = Math.floor((W * H) / 90000) + 60; // density, tweak if needed
    for (let i = 0; i < count; i++) {{
      flakes.push({{
        x: Math.random() * W,
        y: Math.random() * H,
        r: rand(0.8, 4.2),
        d: rand(0.5, 1.6),
        tilt: rand(-0.8, 0.8),
        tiltAngle: rand(0, Math.PI*2),
        opacity: rand(0.45, 1.0)
      }});
    }}
  }}
  initFlakes();

  let angle = 0;
  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over';
    for (let i = 0; i < flakes.length; i++) {{
      const f = flakes[i];
      ctx.beginPath();
      const x = f.x + Math.sin(angle + f.tiltAngle) * 12;
      const y = f.y;
      ctx.globalAlpha = f.opacity * 0.95;
      ctx.shadowColor = 'rgba(255,255,255,0.12)';
      ctx.shadowBlur = 6;
      ctx.fillStyle = 'rgba(255,255,255,0.95)';
      ctx.arc(x, y, f.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1.0;
      f.tiltAngle += 0.01 * f.d;
      f.y += Math.pow(f.d, 1.15) + 0.45;
      f.x += Math.sin(angle) * 0.5 + f.tilt * 0.6;
      if (f.y > H + 10) {{
        f.y = -10;
        f.x = Math.random() * W;
      }}
      if (f.x > W + 20) f.x = -20;
      if (f.x < -20) f.x = W + 20;
    }}
    angle += 0.002;
    requestAnimationFrame(draw);
  }}

  canvas.style.opacity = '0';
  canvas.style.transition = 'opacity 900ms ease';
  requestAnimationFrame(() => {{ canvas.style.opacity = '1'; draw(); }});

}})();
</script>
""",
    height=0,
)

# --------- Helpers: wrappers that use model_utils if available, else simple fallbacks ----------
def safe_extract_text(path: str, filename: str = None) -> str:
    if MU_AVAILABLE and hasattr(mu, "extract_text"):
        try:
            return mu.extract_text(path, filename)
        except Exception:
            pass
    # fallback try read text
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def safe_parse_jd_skills(jd_text: str) -> List[str]:
    if MU_AVAILABLE and hasattr(mu, "parse_jd_skills"):
        try:
            return mu.parse_jd_skills(jd_text)
        except Exception:
            pass
    # basic fallback: look for common skill tokens
    commons = ['python','sql','java','c++','aws','docker','react','node','pandas','tensorflow','pytorch']
    jt = jd_text.lower()
    return [c for c in commons if c in jt]

def safe_compute_simple_features(jd_skills, resume_text):
    if MU_AVAILABLE and hasattr(mu, "compute_simple_features"):
        try:
            return mu.compute_simple_features(jd_skills, resume_text)
        except Exception:
            pass
    # fallback simple coverage
    res = resume_text.lower()
    if not jd_skills:
        return {"skills_coverage": 0.0, "missing_skills": []}
    matched = sum(1 for s in jd_skills if s.lower() in res)
    cov = matched / max(1, len(jd_skills))
    missing = [s for s in jd_skills if s.lower() not in res]
    return {"skills_coverage": float(cov), "missing_skills": missing}

def safe_compute_ats_score(resume_text):
    if MU_AVAILABLE and hasattr(mu, "compute_ats_score"):
        try:
            return mu.compute_ats_score(resume_text)
        except Exception:
            pass
    score = 0
    reasons = []
    if "@" in resume_text: score += 25
    else: reasons.append("Missing email")
    if any(ch.isdigit() for ch in resume_text): score += 25
    else: reasons.append("Missing phone-like tokens")
    if "skills" in resume_text.lower(): score += 25
    else: reasons.append("No skills section")
    if len(resume_text) > 200: score += 25
    else: reasons.append("Resume too short")
    return {"ats_score": score, "reasons": reasons}

def safe_aggregate_components(features, jd_text):
    if MU_AVAILABLE and hasattr(mu, "aggregate_components"):
        try:
            return mu.aggregate_components(features, jd_text)
        except Exception:
            pass
    return {"skills": min(100, features.get("skills_coverage",0)*100), "experience": 50.0, "projects": 60.0, "education": 70.0}

def safe_ideal_profile_from_jd(jd_text):
    if MU_AVAILABLE and hasattr(mu, "ideal_profile_from_jd"):
        try:
            return mu.ideal_profile_from_jd(jd_text)
        except Exception:
            pass
    return {"skills":100, "experience":100, "projects":100, "education":100}

def safe_semantic_match_score(jd_phrases, resume_phrases):
    if MU_AVAILABLE and hasattr(mu, "semantic_match_score"):
        try:
            return mu.semantic_match_score(jd_phrases, resume_phrases)
        except Exception:
            pass
    # try sentence-transformers if available
    if util is not None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            jd_emb = model.encode(jd_phrases, convert_to_tensor=True)
            res_emb = model.encode(resume_phrases, convert_to_tensor=True)
            sims = util.cos_sim(jd_emb, res_emb).cpu().numpy()
            matches = []
            for i, jd in enumerate(jd_phrases):
                row = sims[i]
                best_idx = int(row.argmax())
                matches.append({"jd": jd, "matched": resume_phrases[best_idx], "sim": float(row[best_idx])})
            return matches
        except Exception:
            pass
    # fallback simple substring match
    matches = []
    for jd in jd_phrases:
        best = ""
        sim = 0.0
        for r in resume_phrases:
            if jd.lower() in r.lower():
                best = r
                sim = 0.9
                break
            # token overlap
            jd_tokens = set(jd.lower().split())
            r_tokens = set(r.lower().split())
            if jd_tokens and r_tokens:
                inter = jd_tokens.intersection(r_tokens)
                score = len(inter)/max(len(jd_tokens),1)
                if score > sim:
                    sim = score
                    best = r
        matches.append({"jd": jd, "matched": best, "sim": float(sim)})
    return matches

def safe_generate_ai_suggestions(missing_skills, resume_text, jd_text):
    if MU_AVAILABLE and hasattr(mu, "generate_ai_suggestions"):
        try:
            return mu.generate_ai_suggestions(missing_skills, resume_text, jd_text)
        except Exception:
            pass
    # simple fallback suggestions
    headline = "Add or strengthen: " + ", ".join(missing_skills[:4]) if missing_skills else "General polish: quantify achievements"
    improvements = []
    for s in (missing_skills or [])[:6]:
        improvements.append({"title": f"Add {s} with measurable impact", "why": f"{s} mentioned in JD but not strongly present.", "example": [f"Implemented {s}, reduced X by 20%"]})
    return {"headline": headline, "summary": "Focus on adding missing skills and quantify impact.", "improvements": improvements, "edits": [], "interview_questions": [], "resources": [], "overall_notes": ""}

def safe_generate_pdf(name, suggestions, metrics):
    if MU_AVAILABLE and hasattr(mu, "generate_pdf_suggestions_bytes"):
        try:
            return mu.generate_pdf_suggestions_bytes(name, suggestions, metrics)
        except Exception:
            pass
    # fallback: simple text bytes
    text = f"Suggestions for {name}\\n\\nMetrics:\\n"
    for k,v in (metrics or {}).items():
        text += f"{k}: {v}\\n"
    text += "\\nSuggestions:\\n"
    text += suggestions.get("headline","") + "\\n\\n"
    for imp in suggestions.get("improvements", []):
        text += f"- {imp.get('title')}: {imp.get('why')}\\n"
    return text.encode("utf-8")

def safe_get_embedder():
    if MU_AVAILABLE and hasattr(mu, "get_embedder"):
        try:
            return mu.get_embedder()
        except Exception:
            pass
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

# -------------------------
# UI: Sidebar for uploads & controls
st.title("🔐 Cipher — Multi-Resume Analyzer")
st.markdown("Upload a Job Description and multiple Resumes. The app will score each candidate and highlight the best match.")

with st.sidebar:
    st.header("Upload")
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf","docx","txt"], help="Upload one JD", accept_multiple_files=False)
    resumes = st.file_uploader("Resumes (PDF/DOCX/TXT) — select multiple", type=["pdf","docx","txt"], accept_multiple_files=True, help="You can upload many resumes at once")
    st.divider()
    st.markdown("Tip: Upload several resumes (CTRL/CMD-click) to compare at once.")

if not jd_file:
    st.info("Please upload a Job Description first.")
    st.stop()

if not resumes:
    st.info("Please upload at least one resume (multiple allowed).")
    st.stop()

# Save JD to temp and extract
jd_temp = os.path.join(tempfile.gettempdir(), f"cipher_jd_{int(time.time())}_{jd_file.name}")
with open(jd_temp, "wb") as f:
    f.write(jd_file.read())
jd_text = safe_extract_text(jd_temp, jd_file.name)

# Parse JD skills / ideal
jd_skills = safe_parse_jd_skills(jd_text)
ideal = safe_ideal_profile_from_jd(jd_text)

# Process each resume
candidates = []  # list of dicts with metrics and suggestions
embedder = safe_get_embedder()

for res_file in resumes:
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), f"cipher_res_{int(time.time())}_{res_file.name}")
        with open(tmp_path, "wb") as f:
            f.write(res_file.read())
        res_text = safe_extract_text(tmp_path, res_file.name)

        # features & scores
        features = safe_compute_simple_features(jd_skills, res_text)
        ats = safe_compute_ats_score(res_text)
        comps = safe_aggregate_components(features, jd_text)

        # semantic matches (build phrases & sentences)
        jd_phrases = jd_skills if jd_skills else [s for s in jd_text.splitlines() if s.strip()][:40]
        res_sents = [s for s in res_text.splitlines() if s.strip()][:400]
        matches = safe_semantic_match_score(jd_phrases, res_sents)

        overall_score = 0.5 * comps.get("skills", 0) + 0.2 * comps.get("experience", 0) + 0.15 * comps.get("projects", 0) + 0.1 * ats.get("ats_score", 0)
        verdict = "High" if overall_score >= 80 else ("Medium" if overall_score >= 50 else "Low")

        suggestions = safe_generate_ai_suggestions(features.get("missing_skills"), res_text, jd_text)
        pdf_bytes = safe_generate_pdf(res_file.name.split(".")[0], suggestions, comps)

        candidates.append({
            "name": res_file.name,
            "text": res_text,
            "features": features,
            "ats": ats,
            "comps": comps,
            "matches": matches,
            "overall": overall_score,
            "verdict": verdict,
            "suggestions": suggestions,
            "pdf": pdf_bytes
        })
    except Exception as e:
        st.error(f"Failed to process {res_file.name}: {e}\n{traceback.format_exc()}")

if not candidates:
    st.error("No candidates processed.")
    st.stop()

# Build summary DataFrame
summary = pd.DataFrame([{"name": c["name"], "overall": c["overall"], "verdict": c["verdict"], "ats": c["ats"].get("ats_score", 0), "skills_cov": c["features"].get("skills_coverage", 0)} for c in candidates])
summary = summary.sort_values("overall", ascending=False).reset_index(drop=True)

best_candidate = candidates[summary.index[0]] if len(candidates) else None
best_name = summary.iloc[0]["name"]

# Top-level summary
st.markdown("## 📊 Summary — all uploaded candidates")
cols = st.columns([2,1,1,1])
cols[0].markdown("**Candidates**")
cols[0].write(summary[["name","verdict"]])
cols[1].metric("Top Overall", f"{summary.iloc[0]['overall']:.1f}")
cols[2].metric("Top ATS", f"{summary['ats'].max():.0f}")
cols[3].metric("Average Skills Coverage", f"{summary['skills_cov'].mean():.2f}")

st.divider()

# Charts comparing candidates
left, right = st.columns([2,1])

with left:
    st.subheader("Overall scores — candidates")
    fig_scores = px.bar(summary, x="name", y="overall", color="verdict", title="Overall relevance scores")
    st.plotly_chart(fig_scores, use_container_width=True)
    st.caption("Bar chart of relevance score per candidate (higher is better).")

    st.subheader("Component radar: overlay of candidates vs ideal")
    categories = list(best_candidate["comps"].keys())
    fig_rad = go.Figure()
    # ideal trace
    fig_rad.add_trace(go.Scatterpolar(r=[ideal.get(k,0) for k in categories], theta=categories, fill='toself', name='Ideal', line=dict(color='black', dash='dash')))
    # each candidate overlay (semi-transparent)
    colors = px.colors.qualitative.Plotly
    for i, c in enumerate(candidates):
        comp = c["comps"]
        fig_rad.add_trace(go.Scatterpolar(r=[comp.get(k,0) for k in categories], theta=categories, fill='toself', name=c["name"], opacity=0.5, line=dict(color=colors[i % len(colors)])))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])), showlegend=True)
    st.plotly_chart(fig_rad, use_container_width=True)
    st.caption("Radar overlay: each candidate vs ideal profile.")

    st.subheader("Skill coverage stacked bars (per JD skill across candidates)")
    # build DataFrame: rows per skill per candidate
    jd_phrases = jd_skills if jd_skills else [s for s in jd_text.splitlines() if s.strip()][:30]
    if jd_phrases:
        rows = []
        for c in candidates:
            for sk in jd_phrases:
                best = 0.0
                for m in c["matches"]:
                    if sk.lower() in (m.get("jd") or "").lower() or sk.lower() in (m.get("matched") or "").lower():
                        best = max(best, m.get("sim",0.0))
                rows.append({"candidate": c["name"], "skill": sk, "matched": best, "missing": max(0, 1-best)})
        df_sk = pd.DataFrame(rows)
        # stacked grouped bar per skill: for readability, choose top N skills or show interactive
        top_sk = jd_phrases[:12]
        df_top = df_sk[df_sk["skill"].isin(top_sk)]
        fig = px.bar(df_top, x="skill", y=["matched","missing"], color="candidate", barmode="group", title="Skill matched vs missing across candidates")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Stacked/grouped bars show how each candidate matches each JD skill.")
    else:
        st.info("No JD skills parsed — skill coverage chart not available.")

with right:
    st.subheader(f"🏆 Best candidate: {best_name}")
    st.write(f"Score: **{best_candidate['overall']:.1f}** — Verdict: **{best_candidate['verdict']}**")
    st.markdown("**Download suggestions PDF for best candidate**")
    st.download_button("Download Best Suggestions", data=best_candidate["pdf"], file_name=f"{best_candidate['name']}_suggestions.pdf")

    st.markdown("### Candidate table (select to inspect)")
    select_name = st.selectbox("Select candidate to inspect", [c["name"] for c in candidates], index=0)
    selected = next(c for c in candidates if c["name"] == select_name)

    st.markdown("**Metrics**")
    st.write(pd.DataFrame([{"component": k, "score": v} for k,v in selected["comps"].items()]))
    st.metric("Overall score", f"{selected['overall']:.1f}")
    st.metric("Verdict", selected["verdict"])
    st.metric("ATS score", selected["ats"].get("ats_score", 0))

st.divider()

# Detailed per-candidate panels
for c in candidates:
    with st.expander(f"{c['name']} — Score: {c['overall']:.1f} — {c['verdict']}", expanded=(c is best_candidate)):
        st.markdown("**Preview (first 800 chars)**")
        st.text(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""))
        st.markdown("**Component breakdown**")
        comp_df = pd.DataFrame({"component": list(c["comps"].keys()), "score": list(c["comps"].values())})
        fig_comp = px.bar(comp_df, x="component", y="score", title=f"{c['name']} component scores")
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("**Top matches (examples)**")
        for m in c["matches"][:6]:
            st.markdown(f"- **JD:** {m['jd'][:160]}")
            st.write(f"  - matched → {m['matched'][:160]}  • sim={m['sim']:.2f}")

        st.markdown("**AI Suggestions**")
        sugg = c["suggestions"]
        st.markdown(f"### {sugg.get('headline','Suggestions')}")
        st.write(sugg.get("summary",""))
        for imp in sugg.get("improvements", [])[:6]:
            with st.expander(imp.get("title","Improvement")):
                st.write("Why:", imp.get("why",""))
                for ex in imp.get("example", [])[:4]:
                    st.code(ex)

        st.markdown("**Interview questions / resources**")
        for q in sugg.get("interview_questions", [])[:6]:
            if isinstance(q, dict):
                st.write("Q:", q.get("question"))
                for p in q.get("expected_points", [])[:4]:
                    st.write("-", p)
            else:
                st.write("-", q)
        for r in sugg.get("resources", [])[:6]:
            st.write("-", r.get("title",""), r.get("url",""))

        st.download_button(f"Download {c['name']} suggestions (PDF)", data=c["pdf"], file_name=f"{c['name']}_suggestions.pdf", mime="application/pdf")

# final notes
st.write("---")
st.caption("Cipher • Multi-resume analysis — demo. Best candidate chosen by highest overall score (weighted).")
