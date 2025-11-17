"""
Application package initialisation.

This module ensures that the `app` directory is recognised as a Python package.
It also creates persistent directories used for storing uploaded files and generated
reports when the module is imported.
"""

import os

# Create directories for temporary files and reports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

for path in [TEMP_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)