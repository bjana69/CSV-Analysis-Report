# FastAPI Automated Report Service

This repository contains a FastAPI backend that accepts a CSV dataset, performs a simple statistical analysis and optional machine‑learning model, and then generates a PDF report summarising the findings.  It is paired with a lightweight HTML/JavaScript front‑end (`../frontend.html`) that allows users to upload a CSV file, choose the task type (classification or regression), select the target column, optionally request LLM‑style enrichment, and download the generated report.

## Project Structure

```
fastapi_project/
│   Dockerfile          # Container definition for deployment
│   requirements.txt    # Python dependencies
│   .env                # Environment variables (e.g. API keys)
│
├── app/                # Main application package
│   ├── __init__.py
│   ├── main.py         # FastAPI entry point
│   ├── analysis.py     # Data loading and modelling routines
│   ├── pdf_generator.py# Create PDF reports from analysis results
│   └── llm.py          # Lightweight text summarisation (LLM placeholder)
│
├── tests/              # Placeholder for API tests
│   ├── __init__.py
│   └── test_main.py    # Example test using FastAPI TestClient
│
└── training/           # Reserved for future model training utilities
    └── __init__.py
```

## Running Locally

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI development server:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open `frontend.html` in your browser and point the API requests to `http://localhost:8000` (the default base path is `/api`).  Upload a CSV, select the target, and download the generated PDF.

## Docker

If you prefer to run the application inside a container:

```bash
docker build -t fastapi-report .
docker run -p 8000:8000 fastapi-report
```

## Notes

* The included LLM summarisation is a placeholder that constructs a summary from the computed metrics and feature importances.  It does not call any external API.  To integrate a real language model (e.g. OpenAI), you would implement the logic in `app/llm.py` and store your API key in `.env`.
* Generated reports and temporary files are stored in `app/reports` and `app/temp` respectively.  These folders are created automatically when the API runs.