"""
Main entry point for the FastAPI application.

This module defines the API endpoints that the front‑end interacts with.  It
provides a single POST endpoint to accept a CSV file, performs the requested
analysis, generates a PDF report, and returns the download link.  A separate
GET endpoint serves the generated PDF back to the client.

The API is intentionally simple: no authentication is included, and
calculations run synchronously.  For large datasets or heavy models, you may
want to offload processing to a background task queue (e.g. Celery, RQ) and
implement progress reporting using WebSockets or Server‑Sent Events.
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi import Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, ValidationError
from typing import Literal
from .analysis import analyze_dataset
from .pdf_generator import create_pdf
from .llm import summarize
from . import REPORTS_DIR, TEMP_DIR

app = FastAPI(title="Automated Statistical Report API")

# Configure CORS based on an environment variable.  Production deployments
# should set CORS_ORIGINS to a comma‑separated list of allowed origins.
cors_raw = os.environ.get("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    """
    Schema for the analyse endpoint request body.  Using pydantic
    validators ensures early feedback if the payload is missing required
    fields or contains invalid values.  The `task_type` field is
    constrained to either "classification" or "regression" via a
    `Literal` type.

    The `target` validator additionally strips any surrounding quotes
    and whitespace so that the API is forgiving when users select
    quoted column names from the front end.
    """
    file_name: str
    content: str  # Base64 encoded file contents
    task_type: Literal['classification', 'regression']
    target: str
    use_llm: bool = False

    @validator('file_name')
    def file_name_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('file_name must be provided')
        return v.strip()

    @validator('content')
    def content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('content must be provided and base64 encoded')
        return v.strip()

    @validator('target')
    def target_strip_and_non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError('target column must be provided')
        # Remove surrounding quotes and whitespace
        v_clean = str(v).strip()
        if (v_clean.startswith('"') and v_clean.endswith('"')) or (v_clean.startswith("'") and v_clean.endswith("'")):
            v_clean = v_clean[1:-1]
        return v_clean.strip()


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest = Body(...)):
    """
    Accept a JSON payload containing a base64‑encoded CSV and analysis parameters.
    Perform a statistical analysis and optional machine‑learning model, then
    generate a PDF report.

    The request body should conform to the `AnalyzeRequest` schema.  The
    response includes a unique task identifier and a URL to download the
    generated PDF.
    """
    # Use pydantic‑validated fields.  The `target` has already been
    # stripped of surrounding quotes via the validator.
    task_type = req.task_type
    # Decode the base64 content into bytes; validate parameter ensures early
    # detection of invalid base64 input.
    import base64
    try:
        content_bytes = base64.b64decode(req.content, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 file content: {exc}")
    # Generate a unique task ID and determine where to save the file
    task_id = str(uuid.uuid4())
    temp_filename = f"{task_id}_{req.file_name}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    # Write the decoded bytes to disk; create the directory if needed
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(content_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write temp file: {exc}")
    # Read the CSV header to determine available columns.  Only read the
    # first row for efficiency.
    import pandas as pd
    import difflib
    try:
        header_df = pd.read_csv(temp_path, nrows=0)
        df_cols = list(header_df.columns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV read error: {exc}")
    # Use the cleaned target value directly from the request
    target_req = req.target
    # Attempt to match the target column: exact match, case‑insensitive, then fuzzy
    matched_target: str | None = None
    if target_req in df_cols:
        matched_target = target_req
    else:
        lower_map = {col.lower().strip(): col for col in df_cols}
        key = target_req.lower().strip()
        if key in lower_map:
            matched_target = lower_map[key]
        else:
            candidates = difflib.get_close_matches(target_req, df_cols, n=1, cutoff=0.8)
            if candidates:
                matched_target = candidates[0]
    if matched_target is None:
        available = ", ".join(df_cols[:50])
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target}' not found. Available columns: {available}"
        )
    # Perform the analysis.  Any errors (e.g. non‑numeric target for regression)
    # are surfaced as 400 Bad Request.
    try:
        report_data = analyze_dataset(temp_path, task_type, matched_target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}")
    # Optionally summarise the results using the LLM stub.  Failures here
    # should not break the overall request; instead include an error string.
    summary_text: str | None = None
    if req.use_llm:
        try:
            summary_text = summarize(report_data)
        except Exception as exc:
            summary_text = f"LLM summarisation failed: {exc}"
    # Create the PDF.  Errors at this stage are internal server errors (500).
    pdf_filename = f"{task_id}_report.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    try:
        create_pdf(report_data, summary_text, pdf_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {exc}")
    # Construct the response payload.  The PDF is served via a separate
    # endpoint to avoid sending large binary data inline.
    return JSONResponse({
        'task_id': task_id,
        'pdf_url': f'/api/download/{task_id}',
        'file_name': pdf_filename,
    })


@app.get("/api/download/{task_id}")
def download_report(task_id: str):
    """
    Serve a previously generated PDF given a task identifier.  If the file
    does not exist, return a 404 error.
    """
    # The report filename is derived from the task ID
    pdf_filename = f"{task_id}_report.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_filename)


# Health check endpoint useful for monitoring and readiness probes.  It
# simply returns a JSON object indicating that the service is running.
@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}