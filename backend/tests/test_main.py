"""
Example tests for the FastAPI report service.

These tests use FastAPI's TestClient to simulate HTTP requests to the API
endpoints without running a live server.  They verify that the `/api/analyze`
endpoint accepts a CSV file and returns a response containing a task
identifier and download URL.

To run these tests locally install pytest (`pip install pytest`) and run

    pytest fastapi_project/tests
"""

import os
import json
from fastapi.testclient import TestClient
from fastapi_project.app.main import app


client = TestClient(app)


def test_analyze_endpoint(tmp_path):
    """Ensure that the analyze endpoint processes a small CSV correctly."""
    # Create a simple CSV for testing
    csv_content = "a,b,c\n1,2,3\n4,5,6\n"
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(csv_content)
    import base64
    # Read and encode the CSV file
    content_bytes = csv_path.read_bytes()
    content_b64 = base64.b64encode(content_bytes).decode('utf-8')
    payload = {
        'file_name': 'test.csv',
        'content': content_b64,
        'task_type': 'regression',
        'target': 'c',
        'use_llm': False
    }
    response = client.post(
        "/api/analyze",
        json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert 'task_id' in data
    assert data['pdf_url'].startswith('/api/download/')