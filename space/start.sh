#!/usr/bin/env bash
# FastAPI (agent, logs, feedback) inside the container; Streamlit is the public page on port 7860
set -e
uvicorn neurograph.app.api:app --host 127.0.0.1 --port 8000 &
exec streamlit run src/neurograph/app/dashboard.py --server.port 7860 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
