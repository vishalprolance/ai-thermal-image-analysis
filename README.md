# Thermal AI: Generative AI System for Manufacturing Equipment Analysis

## Overview
This is an advanced GenAI/agentic AI solution for analyzing thermal imaging videos and IoT sensor data from manufacturing equipment (focused on motors and bearings). It detects anomalies (hotspots, patterns), forecasts failures using ML, retrieves contextual insights from equipment manuals and historical logs via RAG, generates actionable recommendations, and provides outputs via dashboard, reports, and alerts. The system is scalable, ethical (privacy/bias checks), and integrates real-time IoT.

### Key Features
- Video processing with OpenCV and PyTorch (Autoencoder for anomalies, LSTM for forecasting).
- Dual Chroma DB for RAG (manuals and logs).
- LangChain ReAct agent for reasoning and insight generation.
- FastAPI for API endpoints (uploads, insights).
- Gradio dashboard for interactive UI.
- Local MQTT mock with AWS MQ/IoT integration.
- Ethics: PII anonymization, validation pipelines.
- Dockerized for deployment; Locust for load testing.

### Tech Stack
- Python 3.10+
- LangChain (agentic/RAG)
- Chroma DB (vector stores)
- OpenCV (video)
- PyTorch (ML)
- FastAPI (API)
- Gradio (dashboard)
- Paho-MQTT / Boto3 (IoT/AWS)
- Pydantic (models)
- Docker (containerization)

## File Structure
```
.
├── README.md
├── requirements.txt
├── docker-compose.yml
├── api/
│   ├── main.py
│   └── Dockerfile
├── ml/
│   ├── models.py
│   └── Dockerfile
├── rag/
│   ├── rag.py
│   └── Dockerfile
├── agent/
│   ├── core.py
│   └── Dockerfile
├── dashboard/
│   ├── app.py
│   └── Dockerfile
├── utils/
│   ├── models.py
│   └── ethics.py
├── iot/
│   └── mqtt_client.py
├── data/          # Input videos, logs, manuals
├── outputs/       # Insights, reports, audits
├── vector_dbs/    # Chroma indices
├── docs/
│   ├── user_guide.md
│   └── ethics_guidelines.md
└── tests/
    ├── validate.py
    └── load_test.py
```

## Prerequisites
- Python 3.10+
- Docker and Docker Compose
- GROQ API key (for LLM; set as env var)
- AWS credentials (optional for IoT)
- GPU (optional for ML; NVIDIA Docker for CUDA)

## Installation
1. Clone or download the project.
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```
   export GROQ_API_KEY=your_groq_key
   # For AWS IoT (optional)
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```
4. Place sample data:
   - Videos in `data/videos/`
   - Manuals (PDFs) in `data/manuals/`
   - Run ingestion: `python rag/rag.py` (call setup_manual_db if needed).

## Run Steps

### Local Run (No Docker)
1. Start MQTT broker (optional for IoT):
   ```
   docker run -d -p 1883:1883 eclipse-mosquitto:2
   ```
2. Run services in terminals:
   - API: `cd api && uvicorn main:app --reload --port 8000`
   - Dashboard: `cd dashboard && python app.py` (opens http://localhost:7860)
   - IoT mock: `cd iot && python mqtt_client.py`
   - RAG/Agent/ML: Run as needed or integrate in API.
3. Access:
   - API docs: http://localhost:8000/docs
   - Dashboard: http://localhost:7860

### Docker Run
1. Build and start:
   ```
   docker-compose up --build -d
   ```
2. Access:
   - API: http://localhost:8000/docs
   - Dashboard: http://localhost:7860
   - MQTT: localhost:1883
3. Stop: `docker-compose down`

### AWS IoT Integration
- Update iot/mqtt_client.py with your AWS IoT endpoint/certs.
- Run `client.start_aws()` for publishing/subscribing.

## Usage Examples
- **Video Upload (API)**:
  ```
  curl -X POST "http://localhost:8000/upload/video" -F "file=@sample.mp4" -F 'metadata={"equipment_id": "MOTOR-001"}'
  ```
- **Log Analysis (Dashboard)**: Paste JSON in "Log Analysis" tab.
- **RAG Query**: "bearing hotspot" in dashboard.
- **Load Test**: `locust -f tests/load_test.py --host=http://localhost:8000 --users=50 --spawn-rate=5`

## Testing
- Ethics/ML validation: `python utils/ethics.py`
- Unit tests: `pytest tests/validate.py`
- Load test: See above.

## Documentation
- User Guide: docs/user_guide.md
- Ethics Guidelines: docs/ethics_guidelines.md
- API: /docs endpoint

## Scalability and Deployment
- Dockerized for Kubernetes/AWS ECS.
- Scale ML with Ray; API with Gunicorn.
- Monitor with Prometheus (add to compose).

For issues, check logs or run validation scripts.