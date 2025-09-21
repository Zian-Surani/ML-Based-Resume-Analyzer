# 🔐 Cipher — Multi-Resume Analyzer

Cipher is a **Streamlit-powered Resume Relevance Analyzer** built for hackathons and placement teams.  
It automates the evaluation of multiple resumes against a single Job Description (JD) using **AI-driven semantic matching, ATS checks, and rich data visualizations**.  

With Cipher, recruiters can **shortlist faster, more consistently, and at scale** — while students receive **personalized AI-powered feedback** on how to improve their resumes.

---

## ✨ Features

- 📂 **Upload one JD and multiple resumes** (PDF/DOCX/TXT).
- 🤖 **Hybrid scoring**: 
  - Hard match (keywords, ATS-style checks)
  - Soft match (semantic similarity using sentence-transformers or fallback rules).
- 📊 **Rich analytics**:
  - Candidate vs. ideal profile radar
  - Stacked skill coverage bars
  - ATS breakdowns
  - Similarity heatmaps
  - Overall score & verdict (High / Medium / Low)
- 🏆 **Best candidate detection** with weighted scoring.
- 💡 **AI Suggestions**:
  - Missing skills, edits, interview prep questions
  - Downloadable **PDF reports per candidate**
- ❄️ **Premium UI**:
  - Grey tech-themed background
  - Animated snowflakes
  - Subtle gradients and shadows
  - Responsive layout

---

## 📷 Demo Screenshot

![Cipher Screenshot](docs/screenshot.png)

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **ML/NLP**: [SentenceTransformers](https://www.sbert.net/), [HuggingFace models](https://huggingface.co/)  
- **Visualization**: [Plotly](https://plotly.com/python/)  
- **Backend Utilities**: Python + custom `model_utils.py` for parsing, ATS scoring, semantic analysis, and suggestions  
- **Export**: ReportLab / PDF utils for suggestion downloads  

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/cipher.git
cd cipher
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # on Windows
# OR
source venv/bin/activate  # on Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will be available at 👉 [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```
cipher/
│
├── app.py                # Main Streamlit app (UI + logic)
├── model_utils.py        # Utility functions (text extraction, ATS, suggestions, embeddings)
├── requirements.txt      # Python dependencies
├── sample_data/          # Example resumes & job descriptions
└── docs/
    └── screenshot.png    # Demo screenshot for README
```

---

## ⚙️ Configuration

- **Multiple resumes** supported via `st.file_uploader` with `accept_multiple_files=True`.
- **Best candidate** = highest weighted score (default weights: skills 50%, experience 20%, projects 15%, ATS 10%).  
  You can tweak these in `app.py`.
- If no LLM API key is provided, fallback **rule-based suggestions** are generated.  
- To use **Gemini/OpenAI/Claude** for richer suggestions, extend `model_utils.generate_ai_suggestions()` with your API key.

---

## 📊 Scoring Logic

The final **Overall Score (0–100)** is a weighted combination of:
- ✅ Skills coverage (50%)
- ✅ Experience (20%)
- ✅ Projects (15%)
- ✅ ATS score (10%)

Verdicts:
- **High Fit**: ≥ 80
- **Medium Fit**: 50–79
- **Low Fit**: < 50

---

## 🤝 Contributors

- **Rahul Seervi**  
- **Ayush Thakur**  
- **Zian Surani** *(Lead Developer)*  

---

## 📜 License

MIT License © 2025 Cipher Project Contributors

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/) for rapid prototyping  
- [SentenceTransformers](https://www.sbert.net/) for embeddings  
- [Plotly](https://plotly.com/python/) for visualizations  
- [Innomatics Research Labs](https://www.innomatics.in/) — Hackathon problem inspiration  
