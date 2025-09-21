Cipher - Full Demo Package
==========================

This package is a plug-and-play demo for Cipher (Streamlit). It includes:
- app.py : Streamlit UI and orchestration
- model_utils.py : text extraction, semantic matching, ATS scoring, LLM suggestions (OpenAI/Grok hooks), PDF export
- requirements.txt : python dependencies
- sample_data/ : sample JD and resume files
- .env.example : example env file

IMPORTANT:
- Do NOT commit your real API keys to source control.
- Create a `.env` file in the project root (same folder as app.py) containing your API keys if you want LLM suggestions:
    GROK_API_KEY=your_grok_key_here
    GROK_API_URL=https://api.grok.example/v1/generate
    OPENAI_API_KEY=your_openai_key_here

Run:
1. python -m venv venv
2. source venv/bin/activate   # mac/linux
   venv\Scripts\Activate    # windows powershell
3. pip install -r requirements.txt
4. streamlit run app.py
